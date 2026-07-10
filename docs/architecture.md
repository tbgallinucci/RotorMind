# Rotordynamics Copilot — Architecture & Integration Plan

*Merging **Rotordynamic FEA** and **AlexandrIA** into one agentic engineering assistant.*

---

## 1. The idea in one sentence

Fuse your citation-first document assistant (**AlexandrIA**) with your finite-element rotordynamics simulator (**Rotordynamic FEA**) into a single **agentic engineering copilot**: a chat assistant that answers rotordynamics theory and standards questions *with citations*, and — when a question needs numbers — **runs the actual FEA simulation, ingests the resulting report into its own knowledge base, and answers with cited, computed results.**

This turns two decent-but-separate student-grade repos into one portfolio piece that demonstrates, in a single coherent product: numerical engineering (FEA, Reynolds equation, eigen-analysis), RAG / retrieval design (the keyword-scoring rationale is already a strong differentiator), and agentic tool-use / LLM orchestration. That combination is rare and reads as senior-level.

---

## 2. What each project brings

**Rotordynamic FEA** — Python + Streamlit. A `RotordynamicAnalysis` class with a clean entry point (`run_analysis()`), plus `calculate_bearing_coefficients()`, `calculate_system_response()`, `calculate_critical_speeds()`, `plot_results()`. It already produces a self-contained HTML report (all parameters, critical speeds, bearing reactions, embedded plots). References Childs, Vance, API 684.

**AlexandrIA** — FastAPI + local LLM. Turns a `raw/` document folder into a cited `wiki/` knowledge base. Retrieval is deterministic keyword scoring behind a clean `build_context()` seam; endpoints include `/api/chat` (streaming), `/api/search`, `/api/pages`, `/api/tree`. Ships a `wiki-ingest.skill` for adding documents.

The seams line up almost perfectly: FEA has a callable analysis class and an HTML report; AlexandrIA has a pluggable retrieval layer, a streaming chat loop, and an ingestion pipeline. The copilot is built by connecting those seams with a **tool layer** and a **report-to-wiki bridge**.

---

## 3. Target architecture

```
rotordynamics-copilot/                 # new unified monorepo
├── README.md                          # the story: one product, three capabilities
├── docs/
│   └── architecture.md                # this document
├── engine/                            # ← Rotordynamic FEA, imported as a library
│   ├── rotordynamics/
│   │   ├── analysis.py                # RotordynamicAnalysis (was main_rotordynamic.py)
│   │   ├── report.py                  # HTML report builder (extracted from streamlit_app)
│   │   └── schema.py                  # Pydantic in/out models for a run
│   └── tests/
├── assistant/                         # ← AlexandrIA, imported as the chat/RAG layer
│   ├── app/
│   │   ├── main.py                    # FastAPI app + endpoints
│   │   ├── wiki_logic.py              # retrieval (build_context) unchanged
│   │   ├── tools.py                   # NEW: tool registry + FEA tool wrapper
│   │   └── agent.py                   # NEW: tool-calling loop around the LLM
│   ├── static/                        # existing SPA (sidebar + wiki + chat)
│   └── wiki/                          # knowledge base (theory + ingested run reports)
├── knowledge/
│   └── raw/                           # rotordynamics source docs (theory, API 684 notes…)
├── pyproject.toml                     # single install; both packages editable
└── docker-compose.yml                 # optional: one command to run the whole thing
```

Two Python packages (`engine`, `assistant`) in one repo, installed together. The assistant depends on the engine; the engine has no idea the assistant exists (clean one-way dependency).

---

## 4. How the agentic loop works

The core new piece is a **tool-calling loop** wrapped around AlexandrIA's existing chat. The LLM is given two tools and decides which to use per question.

**Tool 1 — `search_knowledge(query)`** — the existing `build_context()`. Retrieves cited wiki sections. Used for theory/standards ("what does API 684 say about the separation margin?").

**Tool 2 — `run_rotordynamic_analysis(params)`** — a thin wrapper over `RotordynamicAnalysis`. Accepts a validated parameter set (shaft geometry, disk, bearings, speed range), runs the sim, returns structured results **and** writes the HTML report into `wiki/runs/` so it becomes citable.

Flow for a numeric question:

```
User: "First critical speed for a 13mm × 747mm steel shaft, 2.3kg disk, journal bearings?"
  │
  ├─ LLM recognises numbers needed → calls run_rotordynamic_analysis(params)
  │     ├─ engine runs analysis.run_analysis()
  │     ├─ report.py writes wiki/runs/2026-07-10-run-001.md (+ embedded plots)
  │     └─ returns {critical_speeds, bearing_reactions, report_slug}
  │
  ├─ LLM may also call search_knowledge("critical speed separation margin")
  │     └─ returns cited theory from wiki/
  │
  └─ LLM answers: computed numbers + theory, every claim cited
        "(run: 2026-07-10-run-001)"  and  "(wiki: api-684-separation-margin)"
```

Because report ingestion writes into the same `wiki/` that `build_context()` reads, **past runs become part of the knowledge base automatically** — the assistant can later answer "what was bearing 2's reaction in the 13 mm run?" purely from retrieval, no re-computation. This is the elegant part: the two systems share one substrate (the wiki), and citations work identically for a textbook page and a simulation you ran five minutes ago.

---

## 5. Key integration points (concrete)

**A. Extract the report builder from Streamlit.** Today the HTML report is generated inside `streamlit_app.py` behind the "Generate Full Report" button. Pull that into `engine/rotordynamics/report.py` as a pure function `build_report(analysis) -> (html, plots)` callable without Streamlit. This is the single most important refactor — it decouples the report from the UI so both the Streamlit app and the agent tool can produce it. Add a Markdown variant `build_wiki_page(analysis) -> md` so runs slot straight into `wiki/`.

**B. Define a parameter schema.** Create `engine/rotordynamics/schema.py` with a Pydantic `RunParams` (shaft, disk, bearing, speed range) mirroring the `__init__` attributes of `RotordynamicAnalysis`, and a `RunResult`. This gives the LLM tool a validated, self-documenting contract and prevents it from passing garbage into the physics.

**C. Add the tool layer.** `assistant/app/tools.py` registers the two tools with JSON schemas (derive `run_rotordynamic_analysis`'s schema from `RunParams`). `assistant/app/agent.py` replaces the single-shot chat call in `/api/chat` with a loop: send tools → if the model calls one, execute it, append the result, continue → stream the final grounded answer. AlexandrIA already streams, so this is an extension of the existing generator, not a rewrite.

**D. Wire runs into retrieval.** Reuse the `wiki-ingest.skill` pattern: after a run, write `wiki/runs/<slug>.md` and add an index row so `rank_pages()` / `build_context()` pick it up with zero other changes. Run reports get a `## Run / <date>` category in `index.md`.

**E. Unify config.** One `pyproject.toml`; keep AlexandrIA's env-var LLM config (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL_NAME`) as the single source of truth. Keep the "runs 100% local against LM Studio" promise — the agent loop works the same against a local or cloud OpenAI-compatible endpoint.

---

## 6. Seed the knowledge base

For the copilot to be impressive on first clone, `knowledge/raw/` + `wiki/` should ship with a small, **openly-licensed** rotordynamics corpus so the citations are real, not empty. Candidates: your own theory notes/derivations (you own them), a summary of API 684 concepts *in your own words* (don't redistribute the standard itself), bearing-theory and Reynolds-equation notes, and a glossary. Mirror AlexandrIA's honesty about licensing — cite sources, only redistribute what's yours or openly licensed. Replace the demo cookbook recipes with rotordynamics pages so the whole repo reads as one domain.

---

## 7. Build phases

**Phase 0 — Repo setup.** Create `rotordynamics-copilot`, drop both projects in under `engine/` and `assistant/`, single `pyproject.toml`, get both running side by side unchanged. *Milestone: `pip install -e .` then both the Streamlit app and the FastAPI app start.*

**Phase 1 — Decouple the report (integration point A).** Extract `report.py`, add the Markdown variant, unit-test that a run produces a valid wiki page. *Milestone: `build_wiki_page(run)` writes an ingestible `.md`.*

**Phase 2 — Schema + tool wrapper (B, C).** Add `RunParams`/`RunResult`, wrap the engine as `run_rotordynamic_analysis`, register both tools. *Milestone: calling the tool directly returns structured results and a report slug.*

**Phase 3 — Agent loop (C, D).** Convert `/api/chat` to the tool-calling loop; wire run reports into the wiki. *Milestone: a numeric question triggers a sim, ingests the report, and answers with mixed run+wiki citations.*

**Phase 4 — Seed corpus + polish (integration point 6).** Real rotordynamics wiki pages, a README that tells the story, a demo GIF/screenshot, tests green. *Milestone: fresh clone → ask a theory question and a numeric question, both answered with citations.*

Phases 0–1 are low-risk refactors; the genuinely new engineering is Phases 2–3. Each phase is independently demoable, so the repo looks finished at every checkpoint.

---

## 8. What makes this strong on GitHub

The README can lead with a single narrative — *"an engineering copilot that reasons over rotordynamics theory and runs the physics to back up its answers"* — supported by a demo showing one chat that both cites a standard and reports a computed critical speed. Reviewers see, in one repo: real numerical methods (not a CRUD app), a *defended* retrieval-design decision (the keyword-vs-embeddings section is already excellent), and a working agentic tool-use loop. The one-way `engine → report → wiki → retrieval` data flow is clean enough to draw in a single diagram, which signals architectural maturity. Keep both original repos as archived predecessors and link forward to the merged one so the evolution is visible.

---

## 9. Risks & mitigations

The main risk is **scope creep** — the agent loop can balloon. Mitigate by shipping Phase 1 first (useful even alone) and treating Phases 2–3 as the real deliverable, everything past that optional. Second risk: **LLM passing invalid physics parameters** — mitigated by the Pydantic `RunParams` gate and sane defaults, so a partial spec still runs. Third: **report generation coupled to matplotlib/Streamlit state** — the Phase 1 extraction is exactly what removes that coupling; do it early and test it in isolation. Keep the engine dependency-free of the assistant so the FEA remains independently usable and testable.
