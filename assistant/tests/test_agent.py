"""Phase 3 — agent loop with the LLM fully mocked (true streaming). Proves:
  * a numeric question triggers run_rotordynamic_analysis,
  * the run report is ingested into the wiki,
  * the final answer streams token-by-token and carries both (run: ...) and
    (wiki: ...) citations,
  * /api/chat streams JSON-line events (no live endpoint / LLM anywhere).
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from assistant.app import agent, main, tools, wiki_logic
from assistant.tests.streaming_fakes import (
    _NonStreamResponse, content_stream, content_then_tool_call_stream, tool_call_stream,
)


def _collect(gen):
    async def go():
        return [chunk async for chunk in gen]
    return asyncio.run(go())


def _reconstruct(raw_lines: list[str]) -> str:
    """Mirror the frontend: apply delta/replace events to get what the user
    would actually see at the end."""
    displayed = ""
    for line in raw_lines:
        for sub in line.splitlines():
            if not sub.strip():
                continue
            evt = json.loads(sub)
            if evt["type"] == "delta":
                displayed += evt["text"]
            elif evt["type"] == "replace":
                displayed = evt["text"]
    return displayed


@pytest.fixture()
def tmp_wiki(monkeypatch, tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    index = wiki / "index.md"
    index.write_text("# Wiki Index\n", encoding="utf-8")
    # Minimal real page so FakeLLMs below can cite it without tripping the
    # "citation to a page that doesn't exist" grounding check — this fixture
    # is an isolated sandbox, not a copy of the real wiki corpus.
    (wiki / "critical-speeds-and-separation-margin.md").write_text(
        "# Critical Speeds and Separation Margin\n\n"
        "API 684 recommends operating away from a critical speed by a "
        "separation margin.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiki_logic, "WIKI_DIR", wiki)
    monkeypatch.setattr(wiki_logic, "INDEX_FILE", index)
    return wiki


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_numeric_question_runs_fea_and_cites_both_sources(tmp_wiki):
    class FakeLLM:
        """Round 1: streams a tool call. Round 2: streams the grounded answer,
        built from the real tool result it was handed back."""
        def __init__(self):
            self.calls = []
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            messages = kwargs["messages"]
            tool_results = [m for m in messages if m.get("role") == "tool"]
            if not tool_results:
                args = json.dumps({"speed": {"start_rad_s": 10, "stop_rad_s": 800,
                                             "step_rad_s": 20}})
                return tool_call_stream("run_rotordynamic_analysis", args)
            run = json.loads(tool_results[-1]["content"])
            page_id = run["report_slug"].rsplit("/", 1)[-1]
            first = run["critical_speeds_rad_s"][0]
            text = (
                f"The first critical speed is {first:.1f} rad/s (run: {page_id}). "
                f"API 684 recommends operating away from it by a separation margin "
                f"(wiki: critical-speeds-and-separation-margin, Separation margin)."
            )
            return content_stream(text)

    llm = FakeLLM()
    messages = [
        {"role": "system", "content": wiki_logic.SYSTEM_PROMPT + agent.AGENT_PROMPT_SUFFIX},
        {"role": "user", "content": "What is the first critical speed of the 13 mm rig?"},
    ]
    raw = _collect(agent.run_agent(llm, messages))
    answer = _reconstruct(raw)

    # tool actually executed and the report was ingested
    runs = list((tmp_wiki / "runs").glob("*.md"))
    assert len(runs) == 1, "run report was not ingested into the wiki"
    page_id = runs[0].stem
    assert page_id in wiki_logic.load_wiki_index(), "index row missing"

    # answer carries both citation styles, with the real run id
    assert f"(run: {page_id})" in answer
    assert "(wiki: " in answer
    assert "rad/s" in answer

    # the tool round-trip really went through the LLM protocol
    assert len(llm.calls) == 2
    assert any(m.get("role") == "tool" for m in llm.calls[1]["messages"])
    tool_names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert set(tool_names) == {"search_knowledge", "run_rotordynamic_analysis"}

    # genuinely streamed: more than one delta event reached the caller
    delta_events = sum(1 for line in raw for sub in line.splitlines() if sub.strip()
                       and json.loads(sub)["type"] == "delta")
    assert delta_events > 1, "final answer should stream in multiple chunks, not one blob"


def test_tool_errors_are_fed_back_not_raised(tmp_wiki):
    out = agent._execute_tool_call("run_rotordynamic_analysis",
                                   json.dumps({"shaft": {"diameter_m": -5}}))
    assert out.startswith("ERROR:")
    out2 = agent._execute_tool_call("nonexistent_tool", "{}")
    assert out2.startswith("ERROR: unknown tool")


def test_exhausted_tool_budget_falls_back_to_streaming(tmp_wiki):
    class LoopingLLM:
        def __init__(self):
            self.calls = []
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("tools"):  # still has a tool budget -> keep looping
                args = json.dumps({"query": "critical speed"})
                return tool_call_stream("search_knowledge", args, call_id=f"c{len(self.calls)}")
            # forced final round: tools=None -> plain content, no citations
            return content_stream("final streamed answer")

    llm = LoopingLLM()
    raw = _collect(agent.run_agent(llm, [{"role": "user", "content": "hi"}],
                                   max_tool_rounds=2))
    answer = _reconstruct(raw)
    assert answer == "final streamed answer"
    # last call is the forced round with no tools
    assert llm.calls[-1].get("tools") is None
    assert llm.calls[-1].get("stream") is True


def test_preamble_before_tool_call_is_hidden_but_not_forgotten(tmp_wiki):
    """A round that reasons in prose then calls a tool should clear the
    display (the prose was never a final answer) but must NOT erase that
    reasoning from the conversation history — otherwise the model has to
    start over from nothing next round, which is how a well-reasoned partial
    answer used to get replaced by an uninformative one-liner once the tool
    budget ran out (the model lost track of its own earlier reasoning)."""
    preamble = "Let me check the maximum continuous speed for this rotor first."

    class FakeLLM:
        def __init__(self):
            self.calls = []
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return content_then_tool_call_stream(
                    preamble, "search_knowledge",
                    json.dumps({"query": "maximum continuous speed"}))
            return content_stream("final answer")

    llm = FakeLLM()
    raw = _collect(agent.run_agent(llm, [{"role": "user", "content": "hi"}],
                                   max_tool_rounds=2))
    answer = _reconstruct(raw)

    # the preamble never survives as the *displayed* answer
    assert answer == "final answer"
    assert preamble not in answer

    # but it does survive in the conversation the model sees on the next call
    second_call_messages = llm.calls[1]["messages"]
    assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
    assert any(m.get("content") == preamble for m in assistant_msgs), (
        "the model's own reasoning before the tool call must stay in its "
        "context, even though it was hidden from the user"
    )


def test_api_chat_streams_agent_answer(tmp_wiki, monkeypatch):
    class FakeLLM:
        def __init__(self):
            self.calls = []
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            messages = kwargs["messages"]
            tool_results = [m for m in messages if m.get("role") == "tool"]
            if not tool_results:
                args = json.dumps({})
                return tool_call_stream("run_rotordynamic_analysis", args)
            run = json.loads(tool_results[-1]["content"])
            page_id = run["report_slug"].rsplit("/", 1)[-1]
            text = (f"First critical speed noted (run: {page_id}). "
                    f"(wiki: critical-speeds-and-separation-margin, Separation margin)")
            return content_stream(text)

    llm = FakeLLM()
    monkeypatch.setattr(main, "client", llm)

    with TestClient(main.app) as tc:
        resp = tc.post("/api/chat", json={"message": "First critical speed of the rig?",
                                          "history": []})
    assert resp.status_code == 200
    answer = _reconstruct([resp.text])
    assert "(run: " in answer and "(wiki: " in answer
