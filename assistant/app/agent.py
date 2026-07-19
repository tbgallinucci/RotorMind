"""
Tool-calling loop around the LLM, with true token-by-token streaming.

Every round is requested with stream=True. As chunks arrive we inspect each
delta:
  * if it carries `tool_calls`, this round is a TOOL round — we accumulate the
    call(s) (name + incrementally-streamed arguments, exactly how providers
    actually stream function calls) across chunks. Nothing is shown to the
    user; once the stream ends we execute the tool(s) and loop.
  * if it carries `content`, this round is the FINAL ANSWER — each token is
    forwarded immediately as a `delta` event (optimistic display) while we
    also buffer the full text.

Once a final-answer round finishes, the buffered text is checked against its
sources (grounding.find_violations) — the same rule as before: every number in
a sentence carrying a (wiki: slug, Section) citation must appear on that page.
Two outcomes:
  * clean -> emit `done`; what the user already saw stays as-is.
  * violation -> emit `flag` describing the specific unverifiable claim(s).
    The displayed text is NOT touched or rewritten.

An earlier version rewrote the answer with a second, non-streaming LLM call
when a violation was found. That traded one problem for a worse one: on a
small local model, "rewrite this" routinely over-corrected and discarded
perfectly good, correctly-cited content along with the flagged claim, and the
extra round trip was itself a new failure point (a network hiccup there used
to blank out an already-good answer). Flagging costs nothing beyond the
regex check already being run, never touches text the user already read, and
trusts the user to weigh a called-out claim rather than trusting a second
generation to fix it.

Wire format: every yielded string is one JSON object followed by "\n":
    {"type": "delta",   "text": "..."}   - append to the displayed answer
    {"type": "replace", "text": "..."}   - discard what was displayed, show this
                                            (used only to clear an abandoned
                                            tool-call preamble, see _stream_round)
    {"type": "flag",    "text": "..."}   - unverifiable claim(s); text unchanged
    {"type": "done"}                      - stream finished, nothing to flag
    {"type": "error",   "text": "..."}   - something went wrong
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from pydantic import BaseModel

from . import grounding, wiki_logic
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS

# Appended to the base SYSTEM_PROMPT when the agent loop is active.
AGENT_PROMPT_SUFFIX = """

TOOLS:
You can call tools before answering.
* Use `search_knowledge` for ANY theory, standards, glossary, definition, or
  past-run question — INCLUDING compliance questions (e.g. "is this compliant
  to API 610?"). If a question touches a standard or a concept and you have not
  just retrieved a wiki page about it, call `search_knowledge` FIRST. Never
  answer a standards/theory/compliance question from your own prior knowledge —
  only from retrieved wiki context.
* Use `run_rotordynamic_analysis` when the user needs computed numbers (critical
  speeds, bearing reactions, response) for a specific machine. Pass only the
  parameters the user gave; omitted parameters fall back to the reference rig.
* After a simulation, cite its numbers with the run token from the tool result,
  e.g. (run: 2026-07-10-run-001). Cite knowledge-base facts — definitions,
  concepts, standards requirements, and compliance criteria, NOT just numbers —
  as (wiki: [slug], [Section]). EVERY factual claim taken from the context needs
  one of the two tokens; naming a standard in prose (e.g. "API 610") is not a
  citation.
* The slug in a (wiki: slug, Section) tag must be copied EXACTLY from a
  "### WIKI PAGE: <slug>" marker you actually saw in the retrieved context this
  turn. Never guess, abbreviate, or invent a slug (e.g. "api-610" when the
  marker says "api-610-lateral-analysis") — a wrong slug is treated as citing a
  page that does not exist at all, which fails this answer's grounding check.
* A simulation run report is cited as (run: ...), NEVER as (wiki: ...) — it is
  not a wiki page even though it happens to live in the same wiki folder.
* "IS A LATERAL ANALYSIS REQUIRED" questions (API 610 SS5.2.4.1.1): retrieve
  the api-610-lateral-analysis page and get the exact margin multiplier FROM
  IT — do not state it from memory, and do not restate it without citing
  (wiki: api-610-lateral-analysis, Section) on the sentence that gives it.
  The comparison is the rotor's DRY critical speed against that margin
  applied to the pump's MAXIMUM ALLOWABLE CONTINUOUS SPEED (MCS) — both
  properties of the physical pump, neither of which is a run's speed-sweep
  range, its "rated"/"operating" speed, or its computed (wet) critical speed.
  Never substitute the user's stated operating/rated speed for MCS, and never
  substitute a run's wet critical speed for the dry one — they are different
  numbers by definition, not interchangeable. If the user hasn't given you
  both an MCS and a dry critical speed, say plainly that this project's
  engine does not currently compute a dry critical speed or accept an MCS
  input, so this question cannot be answered from a run alone — do not
  invent a substitute comparison using numbers you do have. The margin in the
  retrieved page is a floor RAISED ABOVE MCS — never compute "MCS minus the
  margin" as a lowered minimum, and never treat it as a two-sided band.
* NUMERIC REQUIREMENTS: never state a threshold, percentage, limit, or required
  margin unless that exact figure appears in the retrieved context. If a
  compliance question needs a criterion the context does not contain, say the
  criterion is not in the knowledge base and give ONLY the indicative figures
  that ARE present, each cited. Never supply a figure from general knowledge,
  and never attach a (wiki: ...) tag to a number you did not read on that page.
* DEFAULTS ARE ASSUMPTIONS: any parameter the user did not give falls back to
  the reference test rig. The tool result lists what was user-specified vs
  defaulted - ALWAYS summarize the assumed defaults in your answer (one short
  sentence is enough). If the user is clearly asking about a specific real
  machine and key parameters are missing (shaft size, disk mass, bearing
  type), ask for them FIRST instead of running with defaults. For generic or
  exploratory questions, run with defaults and state the assumptions.
"""

def _event(type_: str, **kw) -> str:
    kw["type"] = type_
    return json.dumps(kw) + "\n"


def _tool_result_to_str(result) -> str:
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    return str(result)


def _execute_tool_call(name: str, arguments: str) -> str:
    """Dispatch one tool call; errors are returned as text so the model can recover."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'"
    try:
        args = json.loads(arguments) if arguments else {}
        if not isinstance(args, dict):
            return "ERROR: tool arguments must be a JSON object"
        return _tool_result_to_str(fn(args))
    except Exception as e:  # noqa: BLE001 — surfaced to the model on purpose
        return f"ERROR: {type(e).__name__}: {e}"


def _retrieved_pages(convo: list[dict]) -> dict[str, str]:
    """Slug -> text for every wiki page actually placed in the model's context
    this turn (the up-front context message + any search_knowledge results)."""
    blocks = [
        m["content"] for m in convo
        if isinstance(m.get("content"), str) and "### WIKI PAGE:" in m["content"]
    ]
    return grounding.pages_from_context(*blocks)


async def _vet_and_finalize(convo, buffered_text: str) -> AsyncIterator[str]:
    """Check the finished answer against its sources. Yields exactly one
    `done` (clean) or `flag` event. Never calls the model again and never
    touches `buffered_text` — see the module docstring for why."""
    pages = _retrieved_pages(convo)
    violations = grounding.find_violations(buffered_text, pages, loader=wiki_logic.load_page)
    if not violations:
        yield _event("done")
        return
    yield _event("flag", text="; ".join(violations))


async def _stream_round(client, convo, model, tools, temperature, max_tokens,
                        holder: dict) -> AsyncIterator[str]:
    """Perform one streaming round.

    Yields `delta` SSE events for content tokens as they arrive (forward these
    to the user immediately). When the round ends, writes the outcome into
    `holder`:
        holder["kind"]    = "content" | "tool_calls"
        holder["payload"] = full text  | list of {"id","name","arguments"}

    If the round emitted some content before switching to tool_calls (a model
    "thinking out loud" before invoking a tool), that partial content is
    discarded and a `replace` event with empty text is yielded first, so the
    frontend clears what it optimistically displayed — it was never the final
    answer.
    """
    stream = await client.chat.completions.create(
        model=model,
        messages=convo,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    tool_calls_acc: dict[int, dict] = {}
    content_parts: list[str] = []
    saw_tool_calls = False

    async for chunk in stream:
        delta = chunk.choices[0].delta
        tc = getattr(delta, "tool_calls", None)
        if tc:
            saw_tool_calls = True
            for call_delta in tc:
                idx = getattr(call_delta, "index", 0)
                slot = tool_calls_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if getattr(call_delta, "id", None):
                    slot["id"] = call_delta.id
                fn = getattr(call_delta, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
            continue

        text = getattr(delta, "content", None)
        if text:
            content_parts.append(text)
            yield _event("delta", text=text)

    if saw_tool_calls:
        if content_parts:
            # We optimistically streamed preamble text before finding out this
            # round was actually a tool call — it was never a final answer, so
            # clear it from the display. The text itself is NOT discarded (see
            # holder["preamble"] below): small models routinely reason out loud
            # in prose and only issue the tool call at the very end of that
            # same round, so the reasoning is usually genuine work the model
            # did (e.g. "I need the max continuous speed for this — let me
            # check the knowledge base") rather than a throwaway false start.
            # Dropping it from the conversation too would make the model
            # re-derive that reasoning from scratch next round, which is how a
            # rich partial answer used to end up replaced by a curt "I don't
            # have enough information" once the tool budget ran out.
            yield _event("replace", text="")
        holder["kind"] = "tool_calls"
        holder["payload"] = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
        holder["preamble"] = "".join(content_parts) or None
    else:
        holder["kind"] = "content"
        holder["payload"] = "".join(content_parts)


async def run_agent(
    client,
    messages: list[dict],
    model: str | None = None,
    max_tool_rounds: int = 4,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """Yield SSE-style JSON-line events, executing tool calls as the model
    requests them and streaming the final answer token-by-token.

    `client` is any AsyncOpenAI-compatible client (chat.completions.create).
    """
    model = model or wiki_logic.LLM_MODEL_NAME
    convo = list(messages)

    for _round in range(max_tool_rounds):
        holder: dict = {}
        async for ev in _stream_round(client, convo, model, TOOL_SCHEMAS,
                                      temperature, max_tokens, holder):
            yield ev

        if holder.get("kind") == "content":
            async for ev in _vet_and_finalize(convo, holder["payload"]):
                yield ev
            return

        calls = holder.get("payload", [])
        convo.append({
            "role": "assistant",
            "content": holder.get("preamble"),
            "tool_calls": [
                {
                    "id": c["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for i, c in enumerate(calls)
            ],
        })
        for c in calls:
            result = _execute_tool_call(c["name"], c["arguments"])
            convo.append({
                "role": "tool",
                "tool_call_id": c["id"] or "call_0",
                "name": c["name"],
                "content": result,
            })

    # Tool budget exhausted: force one final round with no tools available.
    holder = {}
    async for ev in _stream_round(client, convo, model, None, temperature, max_tokens, holder):
        yield ev
    async for ev in _vet_and_finalize(convo, holder.get("payload", "")):
        yield ev
