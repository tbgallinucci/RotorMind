"""Phase 1 — report builders produce a valid HTML report and wiki page.

Runs a *real* (slightly shortened) analysis; no Streamlit, no display, no LLM.
"""

import re

import numpy as np
import pytest

from engine.rotordynamics.analysis import RotordynamicAnalysis
from engine.rotordynamics.report import build_report, build_wiki_page


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
