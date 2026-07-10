"""Phase 3 — agent loop with the LLM fully mocked. Proves:
  * a numeric question triggers run_rotordynamic_analysis,
  * the run report is ingested into the wiki,
  * the final answer carries both (run: ...) and (wiki: ...) citations,
  * /api/chat streams the agent's answer (no live endpoint / LLM anywhere).
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from assistant.app import agent, main, tools, wiki_logic


# ----------------------------------------------------------------------
# Minimal OpenAI-compatible fakes
# ----------------------------------------------------------------------

class FakeFunction:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class FakeToolCall:
    def __init__(self, id_, name, arguments):
        self.id, self.type = id_, "function"
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Choice:
    def __init__(self, message=None, delta=None):
        self.message, self.delta = message, delta


class _Response:
    def __init__(self, message):
        self.choices = [_Choice(message=message)]


class _Delta:
    def __init__(self, content):
        self.content = content


async def _stream(chunks):
    for c in chunks:
        yield type("Chunk", (), {"choices": [_Choice(delta=_Delta(c))]})()


class FakeLLM:
    """Scripted local-LLM stand-in: first requests the FEA tool, then answers
    with citations pulled from the real tool result it was handed back."""

    def __init__(self):
        self.calls = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]

        if kwargs.get("stream"):
            return _stream(["streamed ", "fallback ", "answer"])

        tool_results = [m for m in messages if m.get("role") == "tool"]
        if not tool_results:
            # Round 1: the model decides the question needs numbers.
            args = json.dumps({"speed": {"start_rad_s": 10, "stop_rad_s": 800,
                                         "step_rad_s": 20}})
            return _Response(FakeMessage(
                tool_calls=[FakeToolCall("call_1", "run_rotordynamic_analysis", args)]))

        # Round 2: ground the answer in the tool result, like a good model would.
        run = json.loads(tool_results[-1]["content"])
        page_id = run["report_slug"].rsplit("/", 1)[-1]
        first = run["critical_speeds_rad_s"][0]
        return _Response(FakeMessage(content=(
            f"The first critical speed is {first:.1f} rad/s (run: {page_id}). "
            f"API 684 recommends operating away from it by a separation margin "
            f"(wiki: critical-speeds-and-separation-margin, Separation margin)."
        )))


@pytest.fixture()
def tmp_wiki(monkeypatch, tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    index = wiki / "index.md"
    index.write_text("# Wiki Index\n", encoding="utf-8")
    monkeypatch.setattr(wiki_logic, "WIKI_DIR", wiki)
    monkeypatch.setattr(wiki_logic, "INDEX_FILE", index)
    return wiki


def _collect(gen):
    async def go():
        return [chunk async for chunk in gen]
    return asyncio.run(go())


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_numeric_question_runs_fea_and_cites_both_sources(tmp_wiki):
    llm = FakeLLM()
    messages = [
        {"role": "system", "content": wiki_logic.SYSTEM_PROMPT + agent.AGENT_PROMPT_SUFFIX},
        {"role": "user", "content": "What is the first critical speed of the 13 mm rig?"},
    ]
    answer = "".join(_collect(agent.run_agent(llm, messages)))

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


def test_tool_errors_are_fed_back_not_raised(tmp_wiki):
    out = agent._execute_tool_call("run_rotordynamic_analysis",
                                   json.dumps({"shaft": {"diameter_m": -5}}))
    assert out.startswith("ERROR:")
    out2 = agent._execute_tool_call("nonexistent_tool", "{}")
    assert out2.startswith("ERROR: unknown tool")


def test_exhausted_tool_budget_falls_back_to_streaming(tmp_wiki):
    class LoopingLLM(FakeLLM):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("stream"):
                return _stream(["final ", "streamed"])
            args = json.dumps({"query": "critical speed"})
            return _Response(FakeMessage(
                tool_calls=[FakeToolCall(f"c{len(self.calls)}", "search_knowledge", args)]))

    llm = LoopingLLM()
    chunks = _collect(agent.run_agent(llm, [{"role": "user", "content": "hi"}],
                                      max_tool_rounds=2))
    assert "".join(chunks) == "final streamed"
    assert llm.calls[-1].get("stream") is True


def test_api_chat_streams_agent_answer(tmp_wiki, monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr(main, "client", llm)

    with TestClient(main.app) as tc:
        resp = tc.post("/api/chat", json={"message": "First critical speed of the rig?",
                                          "history": []})
    assert resp.status_code == 200
    assert "(run: " in resp.text and "(wiki: " in resp.text
