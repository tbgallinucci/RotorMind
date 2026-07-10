"""
Tool-calling loop around the LLM.

Integration points C/D (docs/architecture.md). AlexandrIA's /api/chat used to
make a single grounded LLM call; `run_agent` wraps that in a loop:

    send messages + TOOL_SCHEMAS
      -> while the model requests tool calls, execute them (tools.py),
         append the results as tool messages, and continue
      -> once the model answers without a tool call, stream the text through

Because search_knowledge and run_rotordynamic_analysis both ultimately feed
the same wiki, citations are uniform: '(wiki: <slug>, <Section>)' for theory
pages and '(run: <page-id>)' for simulation runs the assistant just executed.

The LLM client is injected (an AsyncOpenAI-compatible object), so tests can
pass a mock and no live endpoint is ever required.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from pydantic import BaseModel

from . import wiki_logic
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS

# Appended to the base SYSTEM_PROMPT when the agent loop is active.
AGENT_PROMPT_SUFFIX = """

TOOLS:
You can call tools before answering.
* Use `search_knowledge` for theory, standards, glossary, and past-run questions.
* Use `run_rotordynamic_analysis` when the user needs computed numbers (critical
  speeds, bearing reactions, response) for a specific machine. Pass only the
  parameters the user gave; omitted parameters fall back to the reference rig.
* After a simulation, cite its numbers with the run token from the tool result,
  e.g. (run: 2026-07-10-run-001). Cite knowledge-base facts as
  (wiki: [slug], [Section]). Every numeric claim needs one of the two.
* DEFAULTS ARE ASSUMPTIONS: any parameter the user did not give falls back to
  the reference test rig. The tool result lists what was user-specified vs
  defaulted - ALWAYS summarize the assumed defaults in your answer (one short
  sentence is enough). If the user is clearly asking about a specific real
  machine and key parameters are missing (shaft size, disk mass, bearing
  type), ask for them FIRST instead of running with defaults. For generic or
  exploratory questions, run with defaults and state the assumptions.
"""


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
        return _tool_result_to_str(fn(args) if name == "run_rotordynamic_analysis"
                                   else fn(args))
    except Exception as e:  # noqa: BLE001 — surfaced to the model on purpose
        return f"ERROR: {type(e).__name__}: {e}"


async def run_agent(
    client,
    messages: list[dict],
    model: str | None = None,
    max_tool_rounds: int = 4,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """Yield answer chunks, executing tool calls as the model requests them.

    `client` is any AsyncOpenAI-compatible client (chat.completions.create).
    """
    model = model or wiki_logic.LLM_MODEL_NAME
    convo = list(messages)

    for _round in range(max_tool_rounds):
        response = await client.chat.completions.create(
            model=model,
            messages=convo,
            tools=TOOL_SCHEMAS,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            # Final answer reached inside the tool loop — emit it and stop.
            if msg.content:
                yield msg.content
            return

        convo.append({
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            result = _execute_tool_call(tc.function.name, tc.function.arguments)
            convo.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result,
            })

    # Tool budget exhausted: force a final streamed answer with no more tools.
    stream = await client.chat.completions.create(
        model=model,
        messages=convo,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content
