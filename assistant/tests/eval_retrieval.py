"""Retrieval evaluation harness — precision/recall for the wiki retrievers.

A labeled query set against the real, static corpus (the 5 theory/practice/
reference pages; run reports are excluded because they're generated and
change over time, see test_corpus.py for the "does it still find the
standard" case that covers them separately).

For a query, `relevant` is every page slug that would actually ground a
correct answer — not just the "best" one. Pages routinely share ground truth
(e.g. a separation-margin question is legitimately answered by both the
practice note and the glossary entry), so this is set-based, not top-1.

Metrics are computed against what the retriever actually returns (a set of
whole pages pulled into context, not a fixed top-k list), so precision here
measures "how much of what got pulled into context was actually relevant"
and recall measures "of the pages that should have grounded the answer, how
many made it into context".

Run standalone for a human-readable report:
    python -m assistant.tests.eval_retrieval [lexical|vector]
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from assistant.app import tools, wiki_logic

WIKI_PAGE_RE = re.compile(r"### WIKI PAGE: (\S+)")


@dataclass(frozen=True)
class Query:
    text: str
    relevant: frozenset[str]


EVAL_SET: list[Query] = [
    Query("What is oil whirl and oil whip in journal bearings?",
          frozenset({"journal-bearing-theory", "rotordynamics-glossary"})),
    Query("How is eccentricity ratio and attitude angle defined for a journal bearing?",
          frozenset({"journal-bearing-theory", "rotordynamics-glossary"})),
    Query("What are the linearized stiffness and damping coefficients Kij Cij for a journal bearing?",
          frozenset({"journal-bearing-theory", "rotordynamics-glossary"})),
    Query("Why does bearing modelling dominate rotordynamic predictions, comparing hydrodynamic to ball bearings?",
          frozenset({"journal-bearing-theory"})),
    Query("Derive the Reynolds equation assumptions for hydrodynamic lubrication",
          frozenset({"reynolds-equation"})),
    Query("What is the short bearing Ocvirk closed-form solution used by the engine?",
          frozenset({"reynolds-equation"})),
    Query("How does the wedge term and squeeze term appear in the Reynolds equation?",
          frozenset({"reynolds-equation"})),
    Query("How does film thickness and viscosity relate to load capacity in a journal bearing?",
          frozenset({"reynolds-equation"})),
    Query("What is a critical speed and how does the Campbell diagram find it?",
          frozenset({"critical-speeds-and-separation-margin", "rotordynamics-glossary"})),
    Query("How is the amplification factor AF computed by the half-power method?",
          frozenset({"critical-speeds-and-separation-margin", "rotordynamics-glossary"})),
    Query("What separation margin does API 610 require between a critical speed and the operating range?",
          frozenset({"api-610-lateral-analysis", "rotordynamics-glossary"})),
    Query("Is a lateral analysis required under API 610 SS5.2.4.1.1 classically stiff screen?",
          frozenset({"api-610-lateral-analysis"})),
    Query("What is the difference between API 610's dry critical speed and wet critical speed definitions SS1.4.7/1.4.8?",
          frozenset({"api-610-lateral-analysis"})),
    Query("What are the Appendix I allowable displacement and shop-verification criteria?",
          frozenset({"api-610-lateral-analysis"})),
    Query("Define gyroscopic effect and mode shape",
          frozenset({"rotordynamics-glossary"})),
    Query("What is an FRF, frequency response function?",
          frozenset({"rotordynamics-glossary"})),
    Query("What is unbalance and how is it quantified?",
          frozenset({"rotordynamics-glossary"})),
    Query("What is the Sommerfeld-type family of closed-form Reynolds equation solutions?",
          frozenset({"reynolds-equation", "rotordynamics-glossary"})),
]


def retrieved_slugs(context: str) -> frozenset[str]:
    return frozenset(WIKI_PAGE_RE.findall(context))


def precision_recall(retrieved: frozenset[str], relevant: frozenset[str]) -> tuple[float, float]:
    hits = len(retrieved & relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(relevant) if relevant else 1.0
    return precision, recall


@dataclass(frozen=True)
class Result:
    query: Query
    retrieved: frozenset[str]
    precision: float
    recall: float


def evaluate(mode: str = "lexical") -> list[Result]:
    results = []
    for q in EVAL_SET:
        ctx = tools.search_knowledge(q.text, mode=mode)
        retrieved = retrieved_slugs(ctx)
        precision, recall = precision_recall(retrieved, q.relevant)
        results.append(Result(q, retrieved, precision, recall))
    return results


def macro_average(results: list[Result]) -> tuple[float, float]:
    n = len(results)
    return (sum(r.precision for r in results) / n,
            sum(r.recall for r in results) / n)


def _report(mode: str) -> None:
    results = evaluate(mode)
    print(f"\n=== Retrieval eval — mode={mode} ===")
    for r in results:
        print(f"P={r.precision:.2f} R={r.recall:.2f}  {r.query.text[:70]!r}")
        print(f"    expected={sorted(r.query.relevant)} retrieved={sorted(r.retrieved)}")
    p, rec = macro_average(results)
    print(f"\nmacro precision={p:.3f}  macro recall={rec:.3f}  (n={len(results)})")


if __name__ == "__main__":
    modes = sys.argv[1:] or ["lexical"]
    for m in modes:
        _report(m)
