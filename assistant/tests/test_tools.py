"""Phase 2 — FEA tool wrapper: direct call returns results + slug, wiki file exists,
index row added, and the ingested run is retrievable. No LLM involved."""

import pytest

from assistant.app import tools, wiki_logic
from engine.rotordynamics.schema import RunResult


@pytest.fixture()
def tmp_wiki(monkeypatch, tmp_path):
    """Point the whole wiki at a temp folder so tests never touch the real one."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    index = wiki / "index.md"
    index.write_text("# Wiki Index\n\n**Summary**: test index.\n", encoding="utf-8")
    monkeypatch.setattr(wiki_logic, "WIKI_DIR", wiki)
    monkeypatch.setattr(wiki_logic, "INDEX_FILE", index)
    return wiki


# Short speed sweep to keep the test fast but still catch the 1st critical (~160 rad/s).
FAST_SPEED = {"start_rad_s": 10, "stop_rad_s": 800, "step_rad_s": 20}


def test_run_tool_end_to_end(tmp_wiki):
    result = tools.run_rotordynamic_analysis({"speed": FAST_SPEED})

    assert isinstance(result, RunResult)
    assert result.report_slug.startswith("runs/")
    page_id = result.report_slug.rsplit("/", 1)[-1]

    # physics sanity: default rig -> 1st critical near 160 rad/s, R1+R2 = W = 22.56 N
    assert result.critical_speeds_rad_s, "no critical speeds returned"
    assert 100 < result.critical_speeds_rad_s[0] < 260
    r1, r2 = result.bearing_reactions_n
    assert abs((r1 + r2) - 2.3 * 9.81) < 1e-6
    assert result.speed_points == 40
    assert "(run: " in result.summary

    # artifacts on disk
    page = tmp_wiki / "runs" / f"{page_id}.md"
    assert page.exists(), "wiki run page not written"
    md = page.read_text(encoding="utf-8")
    assert "## Critical Speeds" in md and "## Bearing Reactions" in md
    plot_dir = tmp_wiki / "runs" / page_id
    assert (plot_dir / "system-layout.png").exists()
    assert (plot_dir / "report.html").exists()

    # index row -> retrievable by the existing retriever
    index = wiki_logic.load_wiki_index()
    assert page_id in index
    ctx = tools.search_knowledge("critical speed of the shaft run")
    assert f"WIKI PAGE: {page_id}" in ctx


def test_run_ids_increment(tmp_wiki):
    r1 = tools.run_rotordynamic_analysis({"speed": FAST_SPEED})
    r2 = tools.run_rotordynamic_analysis({"speed": FAST_SPEED})
    assert r1.report_slug != r2.report_slug
    assert r1.report_slug.endswith("run-001")
    assert r2.report_slug.endswith("run-002")


def test_params_are_validated(tmp_wiki):
    with pytest.raises(Exception):
        tools.run_rotordynamic_analysis({"shaft": {"diameter_m": -1}})
    with pytest.raises(ValueError):
        tools.run_rotordynamic_analysis({"bearing": {"kind": "magnetic"}})


def test_custom_params_change_the_answer(tmp_wiki):
    base = tools.run_rotordynamic_analysis({"speed": FAST_SPEED})
    stiff = tools.run_rotordynamic_analysis({
        "speed": FAST_SPEED,
        "shaft": {"diameter_m": 20e-3},   # stiffer shaft -> higher 1st critical
    })
    assert stiff.critical_speeds_rad_s[0] > base.critical_speeds_rad_s[0]


def test_per_bearing_types_and_ball_coefficients(tmp_wiki):
    """Mixed journal/ball system with custom ball stiffness reaches the engine."""
    from assistant.app.tools import RotordynamicAnalysis, _apply_params
    from engine.rotordynamics.schema import RunParams

    p = RunParams.model_validate({
        "bearing1": {"kind": "journal"},
        "bearing2": {"kind": "ball", "ball": {"kxx_n_m": 5e7, "kyy_n_m": 6e7,
                                              "cxx_ns_m": 2e3, "cyy_ns_m": 3e3}},
        "speed": FAST_SPEED,
    })
    a = RotordynamicAnalysis()
    _apply_params(a, p)
    assert a.bearing1_type == "journal" and a.bearing2_type == "ball"
    assert a.ball_bearing2_kxx == 5e7 and a.ball_bearing2_kyy == 6e7
    assert a.ball_bearing2_cxx == 2e3 and a.ball_bearing2_cyy == 3e3

    result = tools.run_rotordynamic_analysis(p)
    assert result.critical_speeds_rad_s


def test_custom_positions_remap_mesh_and_reactions(tmp_wiki):
    from assistant.app.tools import RotordynamicAnalysis, _apply_params
    from engine.rotordynamics.schema import RunParams

    p = RunParams.model_validate({
        "positions": {"bearing1_m": 0.050, "disk_m": 0.300, "bearing2_m": 0.700},
        "speed": FAST_SPEED,
    })
    a = RotordynamicAnalysis()
    _apply_params(a, p)
    # anchor nodes land exactly on the requested stations
    assert abs(a.coord[2, 1] - 0.050) < 1e-12
    assert abs(a.coord[5, 1] - 0.300) < 1e-12
    assert abs(a.coord[14, 1] - 0.700) < 1e-12
    # mesh stays monotonic and inside the shaft
    import numpy as np
    assert np.all(np.diff(a.coord[:, 1]) > 0)
    assert abs(a.coord[-1, 1] - a.le) < 1e-12
    # static reactions follow the new arms: d1=0.25, d2=0.65
    assert abs(a.d1 - 0.250) < 1e-12 and abs(a.d2 - 0.650) < 1e-12
    assert abs((a.FM1 + a.FM2) - a.W) < 1e-9


def test_invalid_positions_rejected(tmp_wiki):
    with pytest.raises(Exception):
        tools.run_rotordynamic_analysis({
            "positions": {"bearing1_m": 0.5, "disk_m": 0.2, "bearing2_m": 0.7}})


def test_mcs_hz_flows_from_input_to_result_and_report(tmp_wiki):
    """MCS is an input (so the agent never has to be told it per-question) and
    must survive into the tool result and the printed run report."""
    default_run = tools.run_rotordynamic_analysis({"speed": FAST_SPEED})
    assert default_run.mcs_hz == 60.0

    custom_run = tools.run_rotordynamic_analysis({"speed": FAST_SPEED, "mcs_hz": 50.0})
    assert custom_run.mcs_hz == 50.0
    assert "MCS (maximum allowable continuous speed): 50.0 Hz" in custom_run.summary

    page_id = custom_run.report_slug.rsplit("/", 1)[-1]
    md = (tmp_wiki / "runs" / f"{page_id}.md").read_text(encoding="utf-8")
    assert "MCS (max. continuous speed) | 50.0 Hz" in md
    assert "NEW pump" in md


def test_summary_carries_precomputed_api610_screen_verdict(tmp_wiki):
    """The tool result must hand the model a ready-made screen verdict so it
    never does the floor comparison itself (the polarity-inversion incident).
    Default rig: first critical ~25.5 Hz < 72 Hz floor -> INCONCLUSIVE."""
    r = tools.run_rotordynamic_analysis({"speed": FAST_SPEED})
    assert "PRECOMPUTED" in r.summary
    assert "INCONCLUSIVE" in r.summary
    assert "72.0 Hz floor (1.20 x MCS)" in r.summary
    # and the ingested page carries the same verdict section
    page_id = r.report_slug.rsplit("/", 1)[-1]
    md = (tmp_wiki / "runs" / f"{page_id}.md").read_text(encoding="utf-8")
    assert "## API 610 Classically-Stiff Screen (SS5.2.4.1.1)" in md
    assert "**Screen verdict: INCONCLUSIVE (both bases).**" in md


def test_summary_discloses_assumed_defaults(tmp_wiki):
    r = tools.run_rotordynamic_analysis({"shaft": {"diameter_m": 0.020}, "speed": FAST_SPEED})
    assert "shaft.diameter_m=0.02" in r.summary
    assert "reference test-rig defaults" in r.summary

    r2 = tools.run_rotordynamic_analysis({})
    assert "No parameters were specified" in r2.summary
