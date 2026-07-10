# Rotordynamics Copilot

**An engineering copilot that reasons over rotordynamics theory *and* runs the physics to back up its answers.**

Ask it a theory or standards question and it answers from a curated knowledge base with a citation for every claim. Ask it something numeric — *"what's the first critical speed for this shaft?"* — and it runs a finite-element rotordynamic analysis, files the report into its own knowledge base, and answers with cited, computed results. Every answer, whether from a reference page or a simulation you just ran, carries a source.

It runs 100% locally against any OpenAI-compatible LLM (e.g. LM Studio) — nothing leaves your machine.

This project merges two of my earlier repos into one product:

- **[Rotordynamic FEA](https://github.com/tbgallinucci/rotordynamics)** — finite-element analysis of shaft-disk-bearing systems (hydrodynamic bearing coefficients, critical speeds, mode shapes, Campbell diagrams).
- **AlexandrIA** — a citation-first document assistant using deterministic keyword retrieval (no vector DB).

## How it works

```
                 ┌──────────────────────────── assistant/ (FastAPI + LLM) ─┐
   you ──chat──▶ │  agent loop ──┬─▶ search_knowledge ─▶ build_context ─┐  │
                 │               └─▶ run_rotordynamic_analysis          │  │
                 └───────────────────────────│──────────────────────────┘  │
                                             ▼                              ▼
                                     engine/ (FEA)                    assistant/wiki/
                                RotordynamicAnalysis ──build_wiki_page──▶ runs/*.md
                                                                     (theory + past runs)
```

The assistant depends on the engine; the engine has no knowledge of the assistant (clean one-way dependency). Simulation reports are written into the same `wiki/` the retriever reads, so a run you executed five minutes ago is citable exactly like a textbook page.

## Layout

```
engine/       Rotordynamic FEA, imported as a library
  rotordynamics/
    analysis.py     RotordynamicAnalysis (the solver)
    report.py       report builders, decoupled from the UI
    schema.py       Pydantic RunParams / RunResult contracts
    streamlit_app.py legacy standalone UI (still works)
assistant/    AlexandrIA, the chat / RAG layer
  app/
    main.py         FastAPI app + endpoints
    wiki_logic.py   keyword retrieval (build_context)
    tools.py        tool registry + FEA tool wrapper
    agent.py        tool-calling loop around the LLM
  static/           single-page UI (sidebar + wiki + chat)
  wiki/             knowledge base (theory + ingested run reports)
knowledge/raw/  source documents
docs/architecture.md   full design & build plan
```

## Status

All build phases are complete: the agent loop answers theory questions from the seeded rotordynamics wiki and runs the FEA for numeric questions, ingesting every run report back into the wiki as a citable page (`wiki/runs/<date>-run-NNN`). Design and integration points are in [`docs/architecture.md`](docs/architecture.md); the overnight build journal is in [`docs/build-log.md`](docs/build-log.md).

- [x] **Phase 0** — unified repo, both apps run side by side
- [x] **Phase 1** — decouple the report builder from Streamlit (`report.py`: `build_report`, `build_wiki_page`)
- [x] **Phase 2** — parameter schema + FEA tool wrapper (`tools.py`: `search_knowledge`, `run_rotordynamic_analysis`)
- [x] **Phase 3** — agent loop; runs ingested into the wiki (`agent.py: run_agent`, wired into `/api/chat`)
- [x] **Phase 4** — seed rotordynamics corpus (bearing theory, Reynolds equation, critical speeds & separation margin, glossary) + tests

Run the tests with `pytest` (they need no network and no LLM — the model is mocked).

## Quickstart

**One click:** double-click `run.bat` (Windows) or run `./run.sh` (macOS/Linux) — it creates the venv, installs, starts the server, and opens http://localhost:8000.

Or manually:

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .

# assistant (chat + wiki UI) — needs a local LLM for chat; browsing works without one
python -m uvicorn assistant.app.main:app --host 0.0.0.0 --port 8000

# engine (standalone FEA), either the library or the legacy Streamlit UI
python -c "from engine.rotordynamics.analysis import RotordynamicAnalysis; RotordynamicAnalysis().run_analysis()"
pip install -e ".[ui]" && streamlit run engine/rotordynamics/streamlit_app.py
```

LLM endpoint is configured via `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME` (defaults target LM Studio). Never commit real keys.

## Licensing

Code is MIT (see `LICENSE`). Sample content inherits its original license; only redistribute documents you have the right to. See `docs/architecture.md` §6 for seeding the rotordynamics corpus.
