"""Phase 4 — the seeded rotordynamics corpus is indexed and retrievable."""

import re

from assistant.app import tools, wiki_logic

CORPUS = {
    "journal-bearing-theory",
    "reynolds-equation",
    "critical-speeds-and-separation-margin",
    "rotordynamics-glossary",
}


def test_corpus_indexed_and_demo_pages_retired():
    idx = wiki_logic.load_wiki_index()
    assert CORPUS <= set(idx)
    for retired in ("tomato-juice", "sweet-potato-soup", "buckwheat-ginger-banana-cake"):
        assert retired not in idx


def test_corpus_pages_have_required_structure():
    for slug in CORPUS:
        md = wiki_logic.load_page(slug)
        assert md is not None, slug
        assert md.startswith("---\n"), f"{slug}: missing frontmatter"
        assert "## Overview" in md
        assert "## Sources" in md, f"{slug}: sources must be cited"


def test_theory_questions_retrieve_the_right_pages():
    ctx = tools.search_knowledge("separation margin required by API 610 for critical speeds")
    assert "critical-speeds-and-separation-margin" in ctx
    ctx = tools.search_knowledge("journal bearing oil whirl cross-coupled stiffness")
    assert "journal-bearing-theory" in ctx
    ctx = tools.search_knowledge("Reynolds equation short bearing assumptions")
    assert "reynolds-equation" in ctx


def test_run_specific_compliance_question_still_retrieves_the_standard():
    """A query naming a specific run by its date/id (a realistic "does this
    run comply with API 610" question) used to have its date digits (2026,
    07, 19, the run number) drown every run report's sections in a false
    numeric-match boost, crowding the actual standard out of the context
    budget entirely (see wiki_logic.extract_query_numbers). The standard
    must still make it into context alongside the run's own data."""
    index = wiki_logic.load_wiki_index()
    run_slugs = [s for s in index if re.match(r"\d{4}-\d{2}-\d{2}-run-\d+$", s)]
    assert run_slugs, "expected at least one ingested run report to test against"
    slug = run_slugs[0]

    ctx = tools.search_knowledge(
        f"is lateral analysis required for {slug} running at 60hz rated speed?"
    )
    assert "api-610-lateral-analysis" in ctx, (
        "the standard's own page must be retrieved, not just the run's data"
    )
    assert slug in ctx
