# Build Log — overnight autonomous run (2026-07-10)

Running Phases 1-4 of docs/architecture.md unattended. Decisions and assumptions logged here as I go.

## Environment
* Sandboxed Linux (Python 3.10.12), venv at /tmp/venv in the sandbox, `pip install -e ".[dev]"` succeeded (fastapi, openai, tiktoken, numpy/scipy/matplotlib, pytest).
* Network is proxied: PyPI works, but `openaipublic.blob.core.windows.net` (tiktoken encodings) is blocked -> the offline tokenizer guard is exercised for real here.

## Tooling issues found & routed around
* **Stale committed `__pycache__`**: both packages ship `.pyc` files that shadowed edited sources. Can't delete files (hard constraint), so all Python runs use `PYTHONPYCACHEPREFIX=/tmp/pycache` and I `touch`ed every `.py` to invalidate the old caches. TODO for Thiago: delete the `__pycache__` dirs and add them to .gitignore.
* **File-sync truncation/corruption**: large writes/edits through the editor-side tools intermittently truncated files on the mount (report.py cut at 1.8 KB; wiki_logic.py lost its tail; one edit left NUL-byte padding). Routed around by writing all files via shell heredocs and verifying with `wc`/import after every write. wiki_logic.py was fully reconstructed (original logic + tokenizer guard); verified by importing and running retrieval against the demo wiki.
* Leftover probe file `engine/rotordynamics/report_part2.tmp.py` (used to test the sync path). Safe to delete — I'm not allowed to delete anything tonight.

## Phase 1 — report builder decoupled (DONE, tests green)
* `engine/rotordynamics/report.py`: ported the five `build_*_fig` builders out of `streamlit_app.py` as pure module-level functions; `matplotlib.use("Agg")` for headless use; added `build_plots`, `build_report(analysis) -> (html, plots)` (self-contained HTML, base64-embedded PNGs) and `build_wiki_page(analysis, run_id) -> (slug, markdown)`.
* Wiki page format mirrors the existing pages: YAML frontmatter (type/title/description/tags/timestamp) + `## Overview / Parameters / Critical Speeds / Bearing Reactions / Plots / Provenance`; slug = `runs/<date>-<run_id>`; citation token `(run: <date>-<run_id>)`.
* Assumption: component axial positions for the layout figure come from the FE mesh (nodes 3/15) and `d1` (disk); the Streamlit UI kept them in session state, which a pure function can't see.
* Decision: report text uses ASCII (R1, rho, um) instead of the original unicode glyphs — the sync layer corrupted multi-byte writes once; not risking it in generated artifacts.
* `wiki_logic.py`: tokenizer init wrapped in try/except with a len()//4 fallback so imports/tests never fail offline (verified: falls back here, retrieval still works).
* Tests: `engine/tests/test_report.py` (3 passed) — runs a real shortened analysis (10-1000 rad/s), asserts valid HTML+PNGs, wiki page structure/params/reactions, and that the 1st critical (~160 rad/s) is physical.

## Phase 2 — FEA tool wrapper (DONE, tests green)
* `assistant/app/tools.py`: `search_knowledge` delegates to `wiki_logic.build_context`; `run_rotordynamic_analysis` validates with `RunParams`, maps params onto the engine, runs it, writes `wiki/runs/<page>.md` + `wiki/runs/<page>/{*.png,report.html}`, appends an index row under `## Runs / <date>`, returns `RunResult`.
* Decision: the FE mesh is hand-defined for the 747 mm reference shaft, so `_apply_params` scales node coordinates (and d1/d2) axially by `length_m/0.747` — shaft length stays a meaningful input; bearing/disk stations keep relative positions. Both bearings share RunParams' single bearing spec (mirrors the schema).
* Decision: tool JSON schema inlines pydantic's `$defs`/`$ref` (local models often don't resolve refs).
* `wiki_logic.load_page` got a recursive-glob fallback and `full_text_search` uses rglob, so pages in `wiki/runs/` are retrievable even though the index stores bare slugs. Run IDs auto-increment per day (run-001, run-002...).
* Tests: `assistant/tests/test_tools.py` (4 passed) — end-to-end run against a temp wiki: results + slug returned, files written, index row added, retrievable via search_knowledge; validation rejects bad params; stiffer shaft raises the 1st critical.

## Phase 3 — agent loop (DONE, tests green)
* `assistant/app/agent.py`: `run_agent(client, messages)` — async generator; loops up to `max_tool_rounds` non-streaming calls with `tools=TOOL_SCHEMAS`, dispatches tool calls (errors are returned to the model as `ERROR: ...` text instead of raising), and emits the final answer; if the budget is exhausted it makes one last *streaming* call with no tools. The LLM client is injected, so tests pass a fake and nothing ever dials out.
* `main.py`: `/api/chat` keeps its context-building + trimming + StreamingResponse shape, but the single-shot completion is now `agent.run_agent`. System prompt = existing SYSTEM_PROMPT + `AGENT_PROMPT_SUFFIX` (tool usage + `(run: ...)` citation rule). Env-var LLM config untouched.
* Hardening: `AsyncOpenAI(...)` construction is wrapped in try/except (in this sandbox it throws for proxy reasons); static dir is now anchored to the package path so the app boots from any cwd.
* Tests: `assistant/tests/test_agent.py` (4 passed) — scripted FakeLLM asks for the FEA tool, the run is *really* executed and ingested into a temp wiki, and the final answer carries `(run: <real-page-id>)` + `(wiki: ...)`; tool errors surface as messages; budget exhaustion falls back to streaming; `/api/chat` via FastAPI TestClient streams the same answer end-to-end.

## Phase 4 — seed corpus + polish (DONE)
* Seeded 4 original wiki pages (sources cited, nothing copyrighted reproduced): `journal-bearing-theory`, `reynolds-equation`, `critical-speeds-and-separation-margin` (API 684 concepts in my own words, explicitly non-normative), `rotordynamics-glossary`. All ASCII-safe, matching the existing frontmatter + `##` section format.
* Demo cookbook pages: rows removed from `wiki/index.md`; the 3 files overwritten with "Retired" notices pointing to the new corpus (per instructions, nothing deleted). `index.md` rewritten around the rotordynamics library; `log.md` appended; `knowledge/raw/README.md` documents the licensing stance.
* README: status checkboxes all done, phase markers cleaned up, pointer to this log, note that tests need no network/LLM.
* New test `assistant/tests/test_corpus.py` (index integrity, page structure, retrieval by theory questions).
* Demo artifact: executed one real reference-rig run against the live wiki -> `wiki/runs/2026-07-10-run-001` (criticals 165.0 and 807.7 rad/s; reactions R1=17.86 N, R2=4.71 N; page + 5 plots + report.html, indexed under `## Runs / 2026-07-10` and confirmed retrievable and served by the API).

---

# Summary

**Done.** Phases 1-4 complete; full suite: **14 passed** (engine/tests/test_report.py 3, assistant/tests/test_tools.py 4, test_agent.py 4, test_corpus.py 3). Verified end to end with the LLM mocked: numeric question -> FEA runs -> report ingested -> answer cites `(run: ...)` + `(wiki: ...)`; theory questions retrieve the seeded corpus; a real ingested demo run ships in `wiki/runs/`.

**How to run**
```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest                                             # no network, no LLM needed
python -m uvicorn assistant.app.main:app --port 8000   # chat needs LM Studio (env: LLM_BASE_URL/LLM_API_KEY/LLM_MODEL_NAME)
```

**TODOs for Thiago**
1. Delete the stale `__pycache__` dirs in `engine/` and `assistant/` and gitignore them — they shadowed edited code all night (I worked around it with PYTHONPYCACHEPREFIX; you won't need to).
2. Delete `engine/rotordynamics/report_part2.tmp.py` (my sync probe; I wasn't allowed to delete anything).
3. First live test with LM Studio up: ask one theory + one numeric question through the UI; the loop is only proven against the mocked client so far. If your local model ignores tools, check it supports OpenAI function-calling.
4. Optional polish from architecture.md §8: demo GIF/screenshot for the README; consider whether the Streamlit app should now call `report.build_report` instead of its own copy (left untouched to keep the legacy UI stable).
5. Design note to review: `run_rotordynamic_analysis` scales the hand-defined FE mesh axially by `shaft.length_m/0.747`, and both bearings share one spec (that's all RunParams offers). Fine for the reference-rig family; revisit if you want arbitrary geometry.

## Post-run additions (2026-07-10, interactive session)
* `run.bat` / `run.sh` one-click launchers (run.bat finds Python 3.10+ via the py launcher and rebuilds wrong-version venvs).
* Wiki assets served at `/wiki-files`; run pages now embed their plots inline and link the full HTML report (old demo run migrated).
* Manual FEA panel in the web UI: "Run FEA analysis" form in the left sidebar -> new `POST /api/run` endpoint (validates RunParams, runs the engine, ingests the report, opens the new run page). Works with no LLM running. Suite now 16 passed.
* FEA Workbench (replaces the cramped sidebar panel): "Wiki | Analysis" tabs in the center pane; full-parameter form grouped by Shaft / Disk / Bearing 1 / Bearing 2 / Journal oil film / Positions / Speed sweep, results card with critical-speed table + reactions + "open report page". RunParams extended to full engine coverage: per-bearing kind, per-bearing ball K/C (incl. cross terms in the schema), shared journal-film geometry (engine limitation, documented), optional axial stations (mesh remapped piecewise-linearly; reactions recomputed from actual arms). Legacy single-"bearing" specs still accepted. Suite: 19 passed.
* Amplitude investigation (user question): verified no unit error — solver is SI internally (quasi-static disk deflection 2.51e-4 m matches 2.37e-4 m hand calc), FRF plots correctly convert to mm. Two look-big effects documented instead of "fixed": (1) Z-channel harmonic force includes static weight (Fe = me·w² − W, MATLAB parity kept by user's choice) so Z curves carry static sag; (2) linear model resonance peaks exceed bearing clearance. Both now noted in the HTML report FRF section and a new "Interpretation Notes" section on every run wiki page — so the LLM retrieves the caveats too.
