"""Precision/recall regression gate for the wiki retrievers, against the
labeled query set in eval_retrieval.py.

Thresholds are set from the harness's actual current numbers (see that
module's docstring for how to rerun it standalone), with headroom for noise
-- not aspirational targets. The point is to catch a real regression (a
ranking change that starts missing pages it used to find, or one that starts
flooding context with irrelevant runs), not to enforce a research-grade bar.

Recall matters far more than precision here: missing the one page that
grounds the answer produces "not available in context" or a fabrication:
pulling in a few extra pages just costs context budget. So the recall floor
is strict and the precision floor is loose.
"""

import pytest

from assistant.app import wiki_vector
from assistant.tests.eval_retrieval import EVAL_SET, evaluate, macro_average


def test_lexical_retrieval_precision_recall():
    results = evaluate("lexical")
    precision, recall = macro_average(results)
    assert recall >= 0.85, f"recall regressed: {recall:.3f} (see per-query dump below)\n" + \
        "\n".join(f"  R={r.recall:.2f} missing={sorted(r.query.relevant - r.retrieved)} {r.query.text!r}"
                   for r in results if r.recall < 1.0)
    assert precision >= 0.30, f"precision regressed: {precision:.3f}"


def test_lexical_retrieval_never_misses_every_relevant_page():
    """A query where recall hits 0 means the retriever found NONE of the
    pages that should have grounded the answer -- worse than merely noisy,
    the answer has nothing to cite and will either refuse or fabricate."""
    results = evaluate("lexical")
    zero_recall = [r for r in results if r.recall == 0.0]
    assert not zero_recall, [r.query.text for r in zero_recall]


@pytest.mark.skipif(not wiki_vector.is_available(), reason="vector backend not installed")
def test_vector_retrieval_precision_recall():
    results = evaluate("vector")
    precision, recall = macro_average(results)
    assert recall >= 0.55, f"recall regressed: {recall:.3f}"
    assert precision >= 0.05, f"precision regressed: {precision:.3f}"


def test_eval_set_is_labeled_with_real_slugs():
    """Guards the eval set itself: a typo'd expected slug would silently
    make every query about that page score 0 recall forever."""
    from assistant.app import wiki_logic
    index = wiki_logic.load_wiki_index()
    for q in EVAL_SET:
        assert q.relevant <= set(index), f"unknown slug(s) in: {q.text!r}"
