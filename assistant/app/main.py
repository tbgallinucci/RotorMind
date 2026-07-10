import os
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
from engine.rotordynamics.schema import RunParams, RunResult
from . import agent, tools, wiki_logic

app = FastAPI(title="RotorMind — Rotordynamics Copilot")

# Mount static files (path anchored to the package so boot is cwd-independent)
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve wiki assets (run plots, HTML reports) so run pages render inline in the UI
app.mount("/wiki-files", StaticFiles(directory=str(wiki_logic.WIKI_DIR)), name="wiki-files")

# OpenAI client for LLM. Guarded so the app (and its tests) can boot in
# environments where the HTTP client can't be constructed; chat just reports
# the problem instead of the whole API dying on import.
try:
    client = AsyncOpenAI(base_url=wiki_logic.LLM_BASE_URL, api_key=wiki_logic.LLM_API_KEY)
except Exception as _e:  # pragma: no cover - environment-specific
    client = None
    _client_error = str(_e)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(STATIC_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/pages")
async def get_pages():
    index = wiki_logic.load_wiki_index()
    pages = []
    for slug, desc in index.items():
        pages.append({"slug": slug, "description": desc})
    return pages

@app.get("/api/tree")
async def get_tree():
    tree, _ = wiki_logic.build_index_tree()
    return tree

@app.get("/api/search")
async def search_wiki(q: str):
    return wiki_logic.full_text_search(q)

@app.get("/api/pages/{slug:path}")
async def get_page(slug: str):
    content = wiki_logic.load_page(slug)
    if content is None:
        raise HTTPException(status_code=404, detail="Page not found")
    return {"slug": slug, "content": content}

@app.get("/api/download-pdf/{slug:path}")
async def download_pdf(slug: str):
    _, slug_to_pdf = wiki_logic.build_index_tree()
    pdf_rel_path = slug_to_pdf.get(slug)
    if not pdf_rel_path:
        raise HTTPException(status_code=404, detail="Source PDF mapping not found")
    
    # Path in raw/ should match the heading subject path
    # e.g. "Manuals / Flow Meter and Prover / Datasheet.pdf"
    # We need to handle the spaces and slashes
    pdf_path = wiki_logic.RAW_DIR
    parts = [p.strip() for p in pdf_rel_path.split("/")]
    for part in parts:
        pdf_path = pdf_path / part

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_rel_path}")
        
    return FileResponse(pdf_path, filename=parts[-1], media_type="application/pdf")

@app.post("/api/run", response_model=RunResult)
def run_analysis_endpoint(params: RunParams):
    """Manual FEA run from the web UI - no LLM involved. Validates the
    parameters, runs the engine, ingests the report into the wiki, and
    returns the structured result (sync def -> FastAPI threadpool)."""
    try:
        return tools.run_rotordynamic_analysis(params)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    user_input = request.message
    history = [msg.dict() for msg in request.history]
    
    # 1. Load Index
    index = wiki_logic.load_wiki_index()
    
    # 2. Build Context (cheap keyword retrieval up front; the agent can pull
    #    more via the search_knowledge tool if this isn't enough)
    context = wiki_logic.build_context(user_input, index, wiki_logic.CONTEXT_BUDGET)
    
    # 3. Trim History
    trimmed_history = wiki_logic.trim_history(history, wiki_logic.HISTORY_TOKENS)
    
    # 4. Prepare Messages
    context_message = {
        "role": "system",
        "content": (
            "=== WIKI LIBRARY CONTEXT (use ONLY this to answer factual questions) ===\n\n"
            + context
            + "\n\n=== END OF WIKI CONTEXT ==="
        ),
    }

    messages = [
        {"role": "system", "content": wiki_logic.SYSTEM_PROMPT + agent.AGENT_PROMPT_SUFFIX},
        context_message,
        *trimmed_history,
        {"role": "user", "content": user_input},
    ]

    # 5. Stream Completion through the tool-calling agent loop (Phase 3).
    #    run_agent executes search_knowledge / run_rotordynamic_analysis as the
    #    model requests them, then streams the final grounded answer.
    async def stream_generator():
        if client is None:
            yield f"Error: LLM client unavailable: {_client_error}"
            return
        try:
            async for chunk in agent.run_agent(client, messages):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
