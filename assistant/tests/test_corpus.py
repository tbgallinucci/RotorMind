"""Phase 4 — the seeded rotordynamics corpus is indexed and retrievable."""

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
    ctx = tools.search_knowledge("separation margin required by API 684 for critical speeds")
    assert "critical-speeds-and-separation-margin" in ctx
    ctx = tools.search_knowledge("journal bearing oil whirl cross-coupled stiffness")
    assert "journal-bearing-theory" in ctx
    ctx = tools.search_knowledge("Reynolds equation short bearing assumptions")
    assert "reynolds-equation" in ctx
