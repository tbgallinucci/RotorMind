"""
Tool registry for the agentic loop.

Integration points B and C (docs/architecture.md). The assistant exposes two
tools to the LLM and lets it choose per question:

  1. search_knowledge(query)           -> cited wiki sections (existing build_context)
  2. run_rotordynamic_analysis(params) -> runs the FEA, ingests the report, returns RunResult

`run_rotordynamic_analysis` is the bridge between the two projects: it
validates the model's parameters with engine.rotordynamics.schema.RunParams,
runs the simulation, writes a citable wiki page (plus plots and the full HTML
report) into `assistant/wiki/runs/`, adds an index row so retrieval picks the
run up, and returns structured results the model answers from.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np

from engine.rotordynamics.analysis import RotordynamicAnalysis
from engine.rotordynamics.report import build_report, build_wiki_page
from engine.rotordynamics.schema import RunParams, RunResult

from . import wiki_logic


# ----------------------------------------------------------------------
# Tool 1 — knowledge search (delegates to the existing retriever)
# ----------------------------------------------------------------------

def search_knowledge(query: str) -> str:
    """Retrieve cited wiki context for a query via wiki_logic.build_context."""
    index = wiki_logic.load_wiki_index()
    return wiki_logic.build_context(query, index, wiki_logic.CONTEXT_BUDGET)


# ----------------------------------------------------------------------
# Tool 2 — FEA run wrapper
# ----------------------------------------------------------------------

# Reference mesh: analysis.coord is hand-meshed for a 747 mm shaft with
# bearing 1 / disk / bearing 2 stations at these nodes.
_REFERENCE_LENGTH_M = 747e-3
_B1_NODE, _DISK_NODE, _B2_NODE = 2, 5, 14  # 0-based rows in analysis.coord


def _apply_ball_params(analysis, num: int, ball) -> None:
    prefix = f"ball_bearing{num}_"
    setattr(analysis, prefix + "kxx", ball.kxx_n_m)
    setattr(analysis, prefix + "kyy", ball.kyy_n_m)
    setattr(analysis, prefix + "kxy", ball.kxy_n_m)
    setattr(analysis, prefix + "kyx", ball.kyx_n_m)
    setattr(analysis, prefix + "cxx", ball.cxx_ns_m)
    setattr(analysis, prefix + "cyy", ball.cyy_ns_m)
    setattr(analysis, prefix + "cxy", ball.cxy_ns_m)
    setattr(analysis, prefix + "cyx", ball.cyx_ns_m)


def _apply_params(analysis: RotordynamicAnalysis, p: RunParams) -> None:
    """Map validated RunParams onto the engine's attributes and rebuild the model.

    Mesh handling: node coordinates are hand-defined for the 747 mm reference
    shaft. By default they are scaled axially with the shaft length. When
    explicit component positions are given, every node is remapped
    piecewise-linearly so the bearing-1, disk, and bearing-2 nodes land exactly
    on the requested stations.
    """
    # Shaft / material
    analysis.E = p.shaft.youngs_modulus_pa
    analysis.rho = p.shaft.density_kg_m3
    analysis.de = p.shaft.diameter_m
    analysis.Ae = (np.pi * analysis.de ** 2) / 4
    analysis.Ie = (np.pi * analysis.de ** 4) / 64
    analysis.le = p.shaft.length_m

    # Disk
    analysis.dd = p.disk.diameter_m
    analysis.ld = p.disk.length_m
    analysis.md = p.disk.mass_kg
    analysis.W = analysis.md * analysis.g
    analysis.me = p.disk.unbalance_kg_m
    analysis.e = analysis.me / analysis.md
    analysis.Ip = (analysis.md * analysis.dd ** 2) / 8
    analysis.Id = (analysis.md * analysis.dd ** 2) / 16 + (analysis.md * analysis.ld ** 2) / 12

    # Bearings: per-bearing type + ball coefficients; shared journal film
    analysis.bearing1_type = p.bearing1.kind.lower().strip()
    analysis.bearing2_type = p.bearing2.kind.lower().strip()
    _apply_ball_params(analysis, 1, p.bearing1.ball)
    _apply_ball_params(analysis, 2, p.bearing2.ball)
    analysis.D = p.journal_film.diameter_m
    analysis.C = p.journal_film.length_m
    analysis.delta = p.journal_film.radial_clearance_m
    analysis.mi = p.journal_film.viscosity_pa_s

    # Mesh geometry: remap node coordinates
    ref = analysis.coord[:, 1].copy()
    if p.positions is not None:
        xp = [0.0, ref[_B1_NODE], ref[_DISK_NODE], ref[_B2_NODE], _REFERENCE_LENGTH_M]
        fp = [0.0, p.positions.bearing1_m, p.positions.disk_m,
              p.positions.bearing2_m, p.shaft.length_m]
        analysis.coord[:, 1] = np.interp(ref, xp, fp)
    else:
        analysis.coord[:, 1] = ref * (p.shaft.length_m / _REFERENCE_LENGTH_M)

    # Static reactions from the actual stations (d1 = disk-b1, d2 = b2-b1)
    b1 = analysis.coord[_B1_NODE, 1]
    analysis.d1 = analysis.coord[_DISK_NODE, 1] - b1
    analysis.d2 = analysis.coord[_B2_NODE, 1] - b1
    analysis.FM2 = analysis.W * analysis.d1 / analysis.d2
    analysis.FM1 = analysis.W - analysis.FM2

    # Speed range (inclusive of the stop value)
    analysis.omega = np.arange(p.speed.start_rad_s,
                               p.speed.stop_rad_s + p.speed.step_rad_s / 2,
                               p.speed.step_rad_s)
    analysis.n = len(analysis.omega)

    # Rebuild dependent state: coefficient arrays + FE matrices.
    analysis.initialize_arrays()
    analysis.Tgeo = np.array([
        [analysis.Am, analysis.Ae],
        [analysis.Im, analysis.Ie],
        [analysis.dm, analysis.de],
    ])
    analysis.build_global_matrices()


def _next_run_id(runs_dir) -> str:
    """First free run-NNN id for today (run-001, run-002, ...)."""
    today = date.today().isoformat()
    taken = set()
    if runs_dir.exists():
        for f in runs_dir.glob(f"{today}-run-*.md"):
            m = re.search(r"run-(\d+)$", f.stem)
            if m:
                taken.add(int(m.group(1)))
    n = 1
    while n in taken:
        n += 1
    return f"run-{n:03d}"


def _add_index_row(page_id: str, description: str) -> None:
    """Append the run to wiki/index.md under a '## Runs' heading (create it if needed)."""
    index_file = wiki_logic.INDEX_FILE
    content = index_file.read_text(encoding="utf-8") if index_file.exists() else "# Wiki Index\n"
    row = f"* [Rotordynamic Analysis {page_id}](runs/{page_id}.md) - {description}"
    if row in content:
        return
    heading = f"## Runs / {date.today().isoformat()}"
    if heading in content:
        content = content.replace(heading, f"{heading}\n{row}", 1)
        # keep the row directly under the heading; formatting stays simple
        content = content.replace(f"{heading}\n{row}\n{row}", f"{heading}\n{row}", 1)
    else:
        if not content.endswith("\n"):
            content += "\n"
        content += f"\n{heading}\n\n{row}\n"
    index_file.write_text(content, encoding="utf-8")


def _flatten_keys(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            out.extend(_flatten_keys(v, path))
        else:
            out.append(f"{path}={v}")
    return out


def run_rotordynamic_analysis(params: dict[str, Any] | RunParams | None = None) -> RunResult:
    """Validate params, run the FEA, ingest the report into the wiki, return results."""
    if params is None:
        params = {}
    p = params if isinstance(params, RunParams) else RunParams.model_validate(params)

    # What did the caller actually specify? (everything else is a default)
    provided = _flatten_keys(p.model_dump(exclude_unset=True))

    analysis = RotordynamicAnalysis()
    _apply_params(analysis, p)
    if not analysis.run_analysis():
        raise RuntimeError("Rotordynamic analysis failed")

    # Build artifacts
    runs_dir = wiki_logic.WIKI_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = _next_run_id(runs_dir)
    slug, md = build_wiki_page(analysis, run_id=run_id)
    page_id = slug.rsplit("/", 1)[-1]

    html, plots = build_report(analysis)

    # Persist: markdown page, plots folder, self-contained HTML report
    (runs_dir / f"{page_id}.md").write_text(md, encoding="utf-8")
    plot_dir = runs_dir / page_id
    plot_dir.mkdir(exist_ok=True)
    for key, png in plots.items():
        (plot_dir / f"{key}.png").write_bytes(png)
    (plot_dir / "report.html").write_text(html, encoding="utf-8")

    # Make it retrievable
    crit = [float(c) for c in getattr(analysis, "critical_speeds", [])]
    first = f"{crit[0]:.1f} rad/s ({crit[0] / (2 * np.pi):.1f} Hz)" if crit else "n/a"
    description = (
        f"FEA run - {analysis.de * 1e3:.0f} mm x {analysis.le * 1e3:.0f} mm shaft, "
        f"{analysis.md:.2f} kg disk, {analysis.bearing1_type}/{analysis.bearing2_type} bearings; "
        f"first critical speed {first}."
    )
    _add_index_row(page_id, description)

    if provided:
        assumptions = (
            f"User-specified parameters: {'; '.join(provided)}. "
            "ALL OTHER parameters used the reference test-rig defaults "
            "(13 mm x 747 mm steel shaft, 2.3 kg disk, journal bearings, "
            "90 um clearance) - state this to the user."
        )
    else:
        assumptions = (
            "No parameters were specified - the ENTIRE run used the reference "
            "test-rig defaults (13 mm x 747 mm steel shaft, 2.3 kg disk, "
            "journal bearings, 90 um clearance) - state this to the user."
        )
    summary = (
        f"Critical speeds: {', '.join(f'{c:.1f} rad/s' for c in crit) or 'none in range'}. "
        f"Bearing reactions: R1={analysis.FM1:.2f} N, R2={analysis.FM2:.2f} N. "
        f"{assumptions} "
        f"The full parameter table is on the ingested wiki page '{slug}' "
        f"(cite as '(run: {page_id})')."
    )
    return RunResult(
        critical_speeds_rad_s=crit,
        bearing_reactions_n=[float(analysis.FM1), float(analysis.FM2)],
        speed_points=int(analysis.n),
        report_slug=slug,
        summary=summary,
    )


# ----------------------------------------------------------------------
# Tool definitions handed to the LLM (OpenAI-compatible function-calling format)
# ----------------------------------------------------------------------

def _run_params_json_schema() -> dict[str, Any]:
    """Inline the $defs from RunParams.model_json_schema() for maximum
    compatibility with local models that don't resolve $ref."""
    schema = RunParams.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"].rsplit("/", 1)[-1]
                merged = {**defs.get(ref, {}), **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search the rotordynamics knowledge base (theory, standards notes, "
            "glossary, and past simulation runs) and return cited source sections.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rotordynamic_analysis",
            "description": "Run a finite-element rotordynamic analysis of a shaft-disk-bearing "
            "system and return critical speeds, bearing reactions, and a citable report. "
            "All parameters are optional; omitted ones use the reference test-rig defaults.",
            "parameters": _run_params_json_schema(),
        },
    },
]

TOOL_DISPATCH = {
    "search_knowledge": lambda args: search_knowledge(**args),
    "run_rotordynamic_analysis": lambda args: run_rotordynamic_analysis(args),
}
