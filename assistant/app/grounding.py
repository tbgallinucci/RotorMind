"""
Post-generation grounding guard.

The agent can retrieve the right page and still assert a figure that is not on
it — e.g. "API 610 requires a separation margin of at least 30%" dressed up
with a real-looking (wiki: ...) citation — because the model knows the number
from pre-training. Prompt rules lower how often that happens; they cannot
guarantee it away on a small local model. This module is the deterministic
backstop that runs between the model and the user.

Rule enforced:
    For every sentence that carries a (wiki: slug, Section) citation: the
    slug must resolve to a real page, and every number in that sentence must
    appear on that page.

Design choices that matter:
* We check numbers against the page the model *cited*, sourced first from the
  text retrieved this turn and, failing that, loaded from disk via a loader
  that walks the whole wiki tree. A loader that runs and confirms the slug
  doesn't exist anywhere is itself a provable fabrication and IS flagged
  (a hallucinated citation to a page that was never retrieved and doesn't
  exist is exactly the kind of "real-looking but fake" citation this guard
  exists to catch). We only stay silent when we genuinely can't check either
  way — no loader was given, or the loader itself errored — never when a
  disk lookup came back empty-handed.
* Numbers inside the (wiki: ...) / (run: ...) tokens themselves are ignored, so
  section names, run dates, and run ids never trigger a false hit.
* Only *sourced* numbers are checked. A number the model computed and did not
  attribute to a wiki page (e.g. a separation margin it worked out) is not
  pinned to a source, so it is out of scope here.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_WIKI_CITE = re.compile(r"[(\[]wiki:\s*([^,>)\]]+)\s*[,>]\s*([^)\]]+)[)\]]")
_RUN_CITE = re.compile(r"[(\[]run:\s*([^)\]]+)[)\]]")
# Splits on sentence-ending punctuation AND on any line break. Markdown
# answers are full of bullets, headings, and code spans with no terminal
# "." — without the newline split, an entire multi-line block (e.g. a
# bulleted formula followed, lines later, by a numeric citation) reads as
# one giant "sentence", so numbers from an uncited bullet get checked
# against a citation that belongs to a completely different line.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_PAGE_MARK = re.compile(r"###\s*WIKI PAGE:\s*(\S+)")


def pages_from_context(*context_blocks: str) -> dict[str, str]:
    """Reconstruct slug -> retrieved text from the '### WIKI PAGE: <slug>'
    blocks that were actually placed in the model's context this turn (the
    up-front context message plus any search_knowledge tool results)."""
    pages: dict[str, str] = {}
    for block in context_blocks:
        if not block:
            continue
        parts = _PAGE_MARK.split(block)
        # parts = [pre, slug1, body1, slug2, body2, ...]
        for i in range(1, len(parts) - 1, 2):
            slug = parts[i].strip()
            body = parts[i + 1]
            pages[slug] = pages.get(slug, "") + "\n" + body
    return pages


_PAGE_NOT_FOUND = object()  # sentinel: loader ran and confirmed no such page exists


def _page_text(slug: str,
               pages: dict[str, str],
               loader: Callable[[str], str | None] | None):
    """Text of the cited page: retrieved context first, then disk.
    Returns None if we genuinely cannot verify either way (no loader given,
    or the loader itself errored) - the caller must not flag in that case.
    Returns _PAGE_NOT_FOUND if a loader was available and it confirms no such
    page exists anywhere in the wiki - that IS a provable fabrication (a
    citation to a slug that was never retrieved and doesn't exist on disk
    either), unlike a genuinely unreadable page."""
    if slug in pages and pages[slug].strip():
        return pages[slug]
    if loader is None:
        return None
    try:
        text = loader(slug)
    except Exception:
        return None
    return text if text else _PAGE_NOT_FOUND


def _num_on_page(num: str, page: str) -> bool:
    """Whole-number containment check: '20' must not match inside '120.5' or
    '20.1' just because it's a substring."""
    return re.search(rf"(?<!\d){re.escape(num)}(?!\d)", page) is not None


def find_violations(answer: str,
                    pages: dict[str, str],
                    loader: Callable[[str], str | None] | None = None) -> list[str]:
    """Return human-readable grounding violations found in `answer`.
    Empty list means every sourced number we could verify checked out."""
    problems: list[str] = []
    for sentence in _SENT_SPLIT.split(answer):
        cites = _WIKI_CITE.findall(sentence)
        if not cites:
            continue
        # numbers in the sentence, excluding the citation tokens themselves
        body = _RUN_CITE.sub("", _WIKI_CITE.sub("", sentence))
        numbers = _NUM.findall(body)
        for slug, _section in cites:
            slug = slug.strip()
            page = _page_text(slug, pages, loader)
            if page is None:
                continue  # cannot verify either way -> stay silent
            if page is _PAGE_NOT_FOUND:
                problems.append(
                    f"(wiki: {slug}, ...) cites a page that does not exist in the knowledge base"
                )
                continue
            for num in numbers:
                if not _num_on_page(num, page):
                    problems.append(
                        f"figure '{num}' is cited to '{slug}' but does not appear on that page"
                    )
    # de-duplicate, preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for p in problems:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered
