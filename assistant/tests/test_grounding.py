"""Grounding guard — fully mocked, no network, no LLM.

Covers the exact failure that motivated it: the model retrieves the right page,
then asserts a figure that is NOT on it ("API 684 requires 30%") wrapped in a
real-looking (wiki: ...) citation. The guard must catch it.

The guard flags rather than rewrites: an earlier version ran a second,
non-streaming LLM call to self-correct the answer, but on a small local model
that routinely discarded perfectly good, correctly-cited content along with
the flagged claim, and the extra round trip was itself a failure point. The
current design never calls the model twice — it emits one `flag` event
alongside whatever was already streamed, and leaves that text untouched.

Also covers that the answer genuinely streams (multiple `delta` events)
before the gate ever runs.
"""

import asyncio
import json

from assistant.app import agent, grounding
from assistant.tests.streaming_fakes import content_stream


# ---------------------------------------------------------------- unit tests

def test_flags_number_not_on_cited_page():
    pages = {"gloss": "Separation margin: required distance; grows with AF "
                      "(see API 684 concepts)."}
    answer = "API 684 requires at least 30% margin (wiki: gloss, Terms)."
    problems = grounding.find_violations(answer, pages)
    # 684 is on the page, 30 is not -> exactly the fabricated figure is flagged.
    assert any("30" in p and "gloss" in p for p in problems)
    assert not any("684" in p for p in problems)


def test_passes_number_present_on_cited_page():
    pages = {"cs": "Indicative margins are 15-16 percent below and 20-26 above."}
    answer = "Indicative margins run about 15 to 16 percent (wiki: cs, Margins)."
    assert grounding.find_violations(answer, pages) == []


def test_skips_when_page_text_unavailable():
    answer = "API 684 requires 30% (wiki: gloss, Terms)."
    assert grounding.find_violations(answer, {}, loader=None) == []


def test_flags_square_bracket_citation_too():
    """The frontend (script.js parseAssistantResponse) tolerates [wiki: ...]
    as well as (wiki: ...) — the guard must not be stricter than what's
    actually allowed to reach the user, or a bracket-formatted fabrication
    sails through unchecked."""
    pages = {"gloss": "no numbers here at all"}
    answer = "API 684 requires at least 30% margin [wiki: gloss, Terms]."
    problems = grounding.find_violations(answer, pages)
    assert any("30" in p and "gloss" in p for p in problems)


def test_flags_citation_to_a_page_that_does_not_exist():
    """A hallucinated slug (typo'd or entirely made up) must not get a free
    pass just because there's nothing to load — a loader that runs and
    confirms the page doesn't exist is itself the fabrication signal."""
    def loader(slug):
        return {"real-page": "Some real content with 42 in it."}.get(slug)

    answer = "This machine complies with the standard (wiki: fake-page, Section)."
    problems = grounding.find_violations(answer, {}, loader=loader)
    assert any("fake-page" in p and "does not exist" in p for p in problems)


def test_does_not_flag_a_real_page_reached_only_via_loader():
    def loader(slug):
        return {"real-page": "Some real content with 42 in it."}.get(slug)

    answer = "The figure is 42 (wiki: real-page, Section)."
    assert grounding.find_violations(answer, {}, loader=loader) == []


def test_number_check_is_not_fooled_by_substring_matches():
    """'20' must not be considered 'on the page' just because it's a
    character substring of '120.5' or '1204' elsewhere on that page."""
    pages = {"cs": "Values recorded: 120.5 and 1204."}
    answer = "The figure is 20 (wiki: cs, Section)."
    problems = grounding.find_violations(answer, pages)
    assert any("20" in p and "cs" in p for p in problems)


def test_uncited_bullet_is_not_blamed_on_an_unrelated_citation_later_in_the_block():
    """Reproduces a real incident: the model stated the API 610 20%/30%
    formula in an uncited bullet list, then several lines later cited a run
    report for an unrelated figure. Without a newline-aware sentence split,
    the whole multi-line block reads as one "sentence", so the formula's
    numbers (never attributed to any source) got checked against — and
    wrongly flagged as missing from — the run's page, which was never about
    them in the first place."""
    pages = {"run-010": "Critical Speeds: 175.4 rad/s, 27.9 Hz."}
    answer = (
        "The threshold depends on the rotor type:\n"
        "* 20% margin for wet-running-only: `dry_critical_speed >= 1.20 x MCS`\n"
        "* 30% margin for rotors that can run dry: `dry_critical_speed >= 1.30 x MCS`\n"
        "\n"
        "From the run report (wiki: run-010, Critical Speeds), the first critical "
        "speed is 175.4 rad/s."
    )
    problems = grounding.find_violations(answer, pages)
    assert problems == [], (
        "the uncited formula numbers (20, 1.20, 30, 1.30) must not be checked "
        f"against the run-010 citation on a later line, got: {problems}"
    )


def test_newline_still_separates_two_different_citations_correctly():
    """The flip side: a genuinely bad figure on its own line must still be
    caught even though nothing terminates that line with '.', '!', or '?'."""
    pages = {"gloss": "no numbers here at all"}
    answer = (
        "Some unrelated intro text.\n"
        "API 684 requires at least 30% margin (wiki: gloss, Terms)"
    )
    problems = grounding.find_violations(answer, pages)
    assert any("30" in p and "gloss" in p for p in problems)


def test_ignores_numbers_inside_citation_tokens():
    pages = {"cs": "no numbers here at all"}
    answer = "See the concept (wiki: cs, Section 2.5) and (run: 2026-07-19-run-003)."
    assert grounding.find_violations(answer, pages) == []


# --------------------------------------------------- integration (run_agent)

CONTEXT = (
    "=== WIKI LIBRARY CONTEXT ===\n\n"
    "### WIKI PAGE: gloss\n## Terms\n"
    "Separation margin: required distance between a critical and the operating "
    "range; grows with AF (see API 684 concepts).\n\n"
    "### WIKI PAGE: cs\n## Separation margin\n"
    "Indicative API 684 figures: 15-16 percent below the minimum, 20-26 percent "
    "above the maximum, increasing with AF.\n"
)

BAD = "API 684 requires a separation margin of at least 30% (wiki: gloss, Terms)."
GOOD = ("The exact API 684 margin is not stated in the notes; the indicative "
        "figures are 15-16 percent below and 20-26 percent above "
        "(wiki: cs, Separation margin).")


class OneShotLLM:
    """Streams a single final-answer round — no correction call exists to
    mock anymore, so this is just a plain content stream."""
    def __init__(self, text: str):
        self.text = text
        self.calls = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return content_stream(self.text)


def _run(llm):
    messages = [
        {"role": "system", "content": agent.wiki_logic.SYSTEM_PROMPT},
        {"role": "system", "content": CONTEXT},
        {"role": "user", "content": "Is run-003 compliant with API 684 at 60 Hz?"},
    ]
    async def go():
        return [chunk async for chunk in agent.run_agent(llm, messages)]
    return asyncio.run(go())


def _events(raw_lines):
    return [json.loads(s) for line in raw_lines for s in line.splitlines() if s.strip()]


def _reconstruct(raw_lines):
    displayed = ""
    for evt in _events(raw_lines):
        if evt["type"] == "delta":
            displayed += evt["text"]
        elif evt["type"] == "replace":
            displayed = evt["text"]
    return displayed


def test_answer_streams_before_the_gate_checks_it():
    llm = OneShotLLM(BAD)
    raw = _run(llm)
    delta_events = [e for e in _events(raw) if e["type"] == "delta"]
    assert len(delta_events) > 1, "the answer should stream token by token"
    streamed_text = "".join(e["text"] for e in delta_events)
    assert "30%" in streamed_text


def test_flags_a_fabrication_without_touching_displayed_text():
    llm = OneShotLLM(BAD)
    raw = _run(llm)
    events = _events(raw)

    # the fabricated figure is exactly what gets flagged...
    flags = [e for e in events if e["type"] == "flag"]
    assert len(flags) == 1
    assert "30" in flags[0]["text"] and "gloss" in flags[0]["text"]

    # ...but the displayed answer itself is never rewritten or replaced
    assert not any(e["type"] == "replace" for e in events)
    assert "30%" in _reconstruct(raw)

    # and, crucially, the model was called exactly once - no correction round trip
    assert len(llm.calls) == 1


def test_clean_answer_emits_done_with_no_flag_and_one_call():
    llm = OneShotLLM(GOOD)
    raw = _run(llm)
    events = _events(raw)
    assert any(e["type"] == "done" for e in events)
    assert not any(e["type"] == "flag" for e in events)
    assert len(llm.calls) == 1
