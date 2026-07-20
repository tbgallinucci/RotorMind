"""Phase 1 — report builders produce a valid HTML report and wiki page.

Runs a *real* (slightly shortened) analysis; no Streamlit, no display, no LLM.
"""

import re

import numpy as np
import pytest

from engine.rotordynamics.analysis import RotordynamicAnalysis
from engine.rotordynamics.report import api610_screen, build_report, build_wiki_page


@pytest.fixture(scope="module")
def analysis():
    a = RotordynamicAnalysis()
    # Shorten the sweep (10..1000 rad/s, coarse) to keep the test quick while
    # still catching the first critical speed (~160 rad/s).
    a.omega = np.arange(10, 1001, 25)
    a.n = len(a.omega)
    a.initialize_arrays()
    assert a.run_analysis() is True
    return a


def test_build_report_html_and_plots(analysis):
    html, plots = build_report(analysis)

    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "Finite Element Rotordynamic Analysis Report" in html
    assert "System Parameters" in html
    assert "data:image/png;base64," in html  # embedded plots

    # journal/journal default -> every plot incl. the bearing locus
    for key in ("system-layout", "bearing-coefficients", "bearing-locus",
                "frequency-response", "mode-shapes"):
        assert key in plots, f"missing plot: {key}"
        assert plots[key][:8] == b"\x89PNG\r\n\x1a\n", f"{key} is not a PNG"


def test_build_wiki_page_structure(analysis):
    slug, md = build_wiki_page(analysis, run_id="run-test")

    assert re.fullmatch(r"runs/\d{4}-\d{2}-\d{2}-run-test", slug)

    # frontmatter matching the existing wiki page format
    assert md.startswith("---\n")
    assert "type: Analysis Run" in md
    assert "description:" in md

    # required sections
    for heading in ("## Overview", "## Parameters", "## Critical Speeds",
                    "## Bearing Reactions", "## Plots", "## Provenance"):
        assert heading in md, f"missing section: {heading}"

    # parameters present
    assert "Shaft diameter, d | 13.0 mm" in md
    assert "Shaft length, L | 747.0 mm" in md
    assert "Disk mass | 2.300 kg" in md

    # computed numbers present
    assert f"| Bearing 1, R1 | {analysis.FM1:.2f} |" in md
    assert f"| Bearing 2, R2 | {analysis.FM2:.2f} |" in md
    crit = analysis.critical_speeds
    assert len(crit) > 0
    assert f"{crit[0]:.1f}" in md

    # plot references and citation token
    assert re.search(r"!\[.*\]\(/wiki-files/runs/\d{4}-\d{2}-\d{2}-run-test/system-layout\.png\)", md)
    assert "report.html" in md
    assert re.search(r"\(run: \d{4}-\d{2}-\d{2}-run-test\)", md)


def test_wiki_page_first_critical_is_physical(analysis):
    """The default rig's first critical speed is known to sit near 160 rad/s."""
    first = float(analysis.critical_speeds[0])
    assert 100.0 < first < 260.0


# ----------------------------------------------------------------------
# API 610 classically-stiff screen — the arithmetic the LLM must never do.
# A real incident had the chat model conclude "no lateral analysis required"
# for rotors whose first critical sat far BELOW the 1.2x MCS floor; these
# tests pin the verdict polarity deterministically.
# ----------------------------------------------------------------------

def test_screen_below_floor_is_inconclusive_never_a_pass():
    s = api610_screen(first_critical_hz=19.2, mcs_hz=60.0)
    assert s["verdict"] == "INCONCLUSIVE"
    assert s["wet_floor_hz"] == pytest.approx(72.0)
    assert s["dry_floor_hz"] == pytest.approx(78.0)
    assert s["clears_wet"] is False and s["clears_dry"] is False
    # the summary must carry the anti-inversion instruction verbatim
    assert "BELOW" in s["summary"]
    assert "no lateral analysis required" in s["summary"]  # as a prohibition
    assert "NOT a pass" in s["summary"]


def test_screen_above_wet_floor_only_passes_wet_basis():
    s = api610_screen(first_critical_hz=75.0, mcs_hz=60.0)  # 72 <= 75 < 78
    assert s["verdict"] == "PASSES (wet-running-only basis)"
    assert s["clears_wet"] is True and s["clears_dry"] is False
    assert "inconclusive" in s["summary"]  # the run-dry basis stays open


def test_screen_above_both_floors_passes_both():
    s = api610_screen(first_critical_hz=90.0, mcs_hz=60.0)
    assert s["verdict"] == "PASSES"
    assert s["clears_wet"] is True and s["clears_dry"] is True
    assert "no lateral analysis required" in s["summary"]


def test_screen_without_critical_speed_is_not_evaluated():
    s = api610_screen(first_critical_hz=None, mcs_hz=60.0)
    assert s["verdict"] == "NOT EVALUATED"
    assert s["clears_wet"] is None and s["clears_dry"] is None


def test_wiki_page_carries_precomputed_screen_section(analysis):
    """The default rig (first critical ~25 Hz, MCS 60 Hz) sits below the
    72 Hz floor -> the page must print the INCONCLUSIVE verdict, spelled
    out, so the chat model quotes instead of computing."""
    _slug, md = build_wiki_page(analysis, run_id="run-test")
    assert "## API 610 Classically-Stiff Screen (SS5.2.4.1.1)" in md
    assert "**Screen verdict: INCONCLUSIVE (both bases).**" in md
    assert "| Wet-running-only floor = 1.20 x MCS | 72.0 Hz |" in md
    assert "floor NOT cleared" in md
