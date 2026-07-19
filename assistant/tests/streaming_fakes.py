"""Fake OpenAI-compatible streaming primitives, shared across test modules.

Mimics the real shape closely enough for agent.py's _stream_round to work
against it: content chunks carry `delta.content`; tool-call chunks carry
`delta.tool_calls`, a list of deltas each with `.index`, optionally `.id`,
and `.function.name` / `.function.arguments` (arguments streamed as
incremental fragments, exactly like real providers do it — name arrives on
the first chunk for that index, arguments accumulate across chunks).
"""

from __future__ import annotations

import types


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Chunk:
    def __init__(self, delta):
        self.choices = [types.SimpleNamespace(delta=delta)]


class _NonStreamResponse:
    """Shape returned for stream=False calls (used only by the one-shot
    self-correction request in agent._vet_and_finalize)."""
    def __init__(self, content):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=content, tool_calls=None))]


async def content_stream(text: str, chunk_size: int = 12):
    """Fake stream of content-only chunks for a final-answer round."""
    for i in range(0, len(text), chunk_size):
        yield _Chunk(_Delta(content=text[i:i + chunk_size]))


async def tool_call_stream(name: str, arguments_json: str, call_id: str = "call_1"):
    """Fake stream for a tool-call round: name arrives on chunk 1 (empty
    arguments), the arguments JSON arrives as fragment(s) after."""
    yield _Chunk(_Delta(tool_calls=[
        types.SimpleNamespace(index=0, id=call_id,
                              function=types.SimpleNamespace(name=name, arguments=""))
    ]))
    mid = max(1, len(arguments_json) // 2)
    for frag in (arguments_json[:mid], arguments_json[mid:]):
        if frag:
            yield _Chunk(_Delta(tool_calls=[
                types.SimpleNamespace(index=0, id=None,
                                      function=types.SimpleNamespace(name=None, arguments=frag))
            ]))


async def content_then_tool_call_stream(text: str, name: str, arguments_json: str,
                                        call_id: str = "call_1", chunk_size: int = 12):
    """Fake stream for a round where the model reasons in prose first and
    only issues the tool call at the end of the same round — the case that
    used to make agent.py discard the reasoning entirely instead of just
    hiding it from the display."""
    for i in range(0, len(text), chunk_size):
        yield _Chunk(_Delta(content=text[i:i + chunk_size]))
    async for chunk in tool_call_stream(name, arguments_json, call_id):
        yield chunk
