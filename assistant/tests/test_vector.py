"""Vector retrieval (the UI's Lexical/Vector toggle), with embeddings fully
faked — same test philosophy as the mocked LLM: no torch, no model download,
no network. The fake embedder is a deterministic bag-of-words hasher, which is
enough to exercise everything that is OURS (chunking, caching, ranking
mechanics, budget fill, output contract, fallback wiring) without testing the
semantics of someone else's model."""

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from assistant.app import agent, main, tools, wiki_logic, wiki_vector
from assistant.tests.streaming_fakes import content_stream, tool_call_stream


# ----------------------------------------------------------------------
# Deterministic fake embedder: hash each token into one of `dim` buckets and
# count. Texts sharing words get similar vectors — no semantics, but stable
# and offline, which is all these tests need.
# ----------------------------------------------------------------------

def fake_embed(texts):
    dim = 64
    out = []
    for t in texts:
        v = np.zeros(dim, dtype=np.float32)
        for tok in re.findall(r"\b\w+\b", t.lower()):
            bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
            v[bucket] += 1.0
        norm = np.linalg.norm(v)
        out.append(v / norm if norm else v)
    return np.asarray(out, dtype=np.float32)


class CountingEmbed:
    """Wraps fake_embed and records what got embedded, to prove cache reuse."""
    def __init__(self):
        self.batches = []

    def __call__(self, texts):
        self.batches.append(list(texts))
        return fake_embed(texts)


@pytest.fixture()
def tmp_wiki(monkeypatch, tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    index = wiki / "index.md"
    index.write_text(
        "# Wiki Index\n\n## Pages\n"
        "* [Journal Bearings](journal-bearings.md) - oil film journal bearing theory\n"
        "* [Rotor Balancing](rotor-balancing.md) - balancing grades and procedure\n",
        encoding="utf-8",
    )
    (wiki / "journal-bearings.md").write_text(
        "# Journal Bearings\n\n## Oil film stiffness\n"
        "The hydrodynamic oil film acts as a spring and damper; its stiffness "
        "depends on viscosity, clearance, and speed.\n",
        encoding="utf-8",
    )
    (wiki / "rotor-balancing.md").write_text(
        "# Rotor Balancing\n\n## Procedure\n"
        "Balance to grade G2.5 by adding correction masses in two planes.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(wiki_logic, "WIKI_DIR", wiki)
    monkeypatch.setattr(wiki_logic, "INDEX_FILE", index)
    return wiki


# ----------------------------------------------------------------------
# Ranking + output contract
# ----------------------------------------------------------------------

def test_relevant_page_ranks_first_and_contract_matches_lexical(tmp_wiki):
    ctx = wiki_vector.build_context(
        "oil film stiffness of a journal bearing", 4000, embed_fn=fake_embed)

    # the on-topic page is present and ahead of the off-topic one (if the
    # off-topic one made it past the similarity floor at all)
    assert "### WIKI PAGE: journal-bearings" in ctx
    if "rotor-balancing" in ctx:
        assert ctx.index("journal-bearings") < ctx.index("rotor-balancing")

    # same contract as the lexical retriever: the grounding guard and the
    # citation rules key off this exact marker, so vector context must be a
    # drop-in replacement
    assert "### WIKI PAGE: " in ctx


def test_off_topic_query_falls_back_to_index(tmp_wiki):
    # everything below the similarity floor -> index fallback, same behaviour
    # as the lexical retriever when nothing matches. The bag-of-words fake
    # can't guarantee near-zero similarity (hash collisions), so this test
    # injects an embedder that makes the query exactly orthogonal to every
    # chunk - testing the floor logic itself, not hash luck.
    def orthogonal_embed(texts):
        vecs = np.zeros((len(texts), 2), dtype=np.float32)
        for i, t in enumerate(texts):
            vecs[i] = [0.0, 1.0] if t == "off topic query" else [1.0, 0.0]
        return vecs

    ctx = wiki_vector.build_context("off topic query", 4000,
                                    embed_fn=orthogonal_embed)
    assert "WIKI INDEX" in ctx


def test_token_budget_is_respected(tmp_wiki):
    ctx = wiki_vector.build_context(
        "oil film stiffness of a journal bearing", 4000, embed_fn=fake_embed)
    assert wiki_logic.count_tokens(ctx) <= 4000 + 50  # + banner/separators


def test_embeddings_are_cached_and_reused(tmp_wiki):
    counter = CountingEmbed()
    wiki_vector.build_context("oil film", 4000, embed_fn=counter)
    # first call: one batch for all chunks + one for the query
    assert len(counter.batches) == 2
    n_chunks = len(counter.batches[0])
    assert n_chunks >= 2

    counter2 = CountingEmbed()
    wiki_vector.build_context("bearing clearance", 4000, embed_fn=counter2)
    # second call: chunk vectors come from the on-disk cache; only the query
    # gets embedded
    assert counter2.batches == [["bearing clearance"]]


def test_new_page_triggers_incremental_embedding_only(tmp_wiki):
    wiki_vector.build_context("oil film", 4000, embed_fn=fake_embed)

    # ingest a new page (what a fresh FEA run report does)
    (tmp_wiki / "whirl.md").write_text(
        "# Oil Whirl\n\n## Instability\nSubsynchronous whirl at ~0.48x speed.\n",
        encoding="utf-8")
    index = wiki_logic.INDEX_FILE
    index.write_text(index.read_text(encoding="utf-8") +
                     "* [Oil Whirl](whirl.md) - oil whirl instability\n",
                     encoding="utf-8")

    counter = CountingEmbed()
    wiki_vector.build_context("whirl instability", 4000, embed_fn=counter)
    # only the NEW page's sections + the query got embedded, not the corpus
    embedded_chunks = counter.batches[0]
    assert all("whirl" in t.lower() for t in embedded_chunks)
    assert counter.batches[-1] == ["whirl instability"]


# ----------------------------------------------------------------------
# Fallback wiring (tools + endpoint)
# ----------------------------------------------------------------------

def test_search_knowledge_vector_mode_falls_back_when_unavailable(tmp_wiki, monkeypatch):
    monkeypatch.setattr(wiki_vector, "is_available", lambda: False)
    vec = tools.search_knowledge("oil film journal bearing", mode="vector")
    lex = tools.search_knowledge("oil film journal bearing", mode="lexical")
    assert vec == lex


def test_search_knowledge_vector_mode_falls_back_on_error(tmp_wiki, monkeypatch):
    monkeypatch.setattr(wiki_vector, "is_available", lambda: True)

    def boom(*a, **kw):
        raise RuntimeError("model download failed")
    monkeypatch.setattr(wiki_vector, "build_context", boom)

    out = tools.search_knowledge("oil film journal bearing", mode="vector")
    assert out == tools.search_knowledge("oil film journal bearing", mode="lexical")


def test_agent_tool_dispatch_routes_search_to_vector(tmp_wiki, monkeypatch):
    """The per-request dispatch built by make_tool_dispatch must reach the
    tool actually executed inside the agent loop."""
    monkeypatch.setattr(wiki_vector, "is_available", lambda: True)
    monkeypatch.setattr(wiki_vector, "build_context",
                        lambda q, b: "### WIKI PAGE: fake\nVECTOR CONTEXT")

    class FakeLLM:
        def __init__(self):
            self.calls = []
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if not any(m.get("role") == "tool" for m in kwargs["messages"]):
                return tool_call_stream("search_knowledge",
                                        json.dumps({"query": "oil film"}))
            return content_stream("ok")

    import asyncio
    llm = FakeLLM()

    async def go():
        return [ev async for ev in agent.run_agent(
            llm, [{"role": "user", "content": "hi"}],
            tool_dispatch=tools.make_tool_dispatch("vector"))]
    asyncio.run(go())

    tool_msg = next(m for m in llm.calls[1]["messages"] if m.get("role") == "tool")
    assert "VECTOR CONTEXT" in tool_msg["content"]


def test_rag_status_reports_unavailability_with_reason(monkeypatch):
    monkeypatch.setattr(wiki_vector, "availability",
                        lambda: (False, "sentence-transformers is not installed"))
    with TestClient(main.app) as tc:
        resp = tc.get("/api/rag-status")
    assert resp.json() == {
        "vector": False,
        "embedding_model": None,
        "reason": "sentence-transformers is not installed",
    }


def test_rag_status_reports_model_when_available(monkeypatch):
    monkeypatch.setattr(wiki_vector, "availability", lambda: (True, None))
    with TestClient(main.app) as tc:
        resp = tc.get("/api/rag-status")
    body = resp.json()
    assert body["vector"] is True
    assert body["embedding_model"] == wiki_vector.EMBEDDING_MODEL_NAME


def test_chat_meta_event_reports_the_mode_actually_used(tmp_wiki, monkeypatch):
    """The stream's first event declares which retriever really built the
    context. It must NOT echo the request: vector-requested-but-unavailable
    reports "lexical" - that declaration is what makes the toggle provable."""
    class FakeLLM:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            return content_stream("answer")

    monkeypatch.setattr(main, "client", FakeLLM())

    def first_event(resp):
        return json.loads(resp.text.splitlines()[0])

    # vector requested, backend unavailable -> meta proves the degradation
    monkeypatch.setattr(wiki_vector, "is_available", lambda: False)
    with TestClient(main.app) as tc:
        resp = tc.post("/api/chat", json={"message": "hi", "history": [],
                                          "rag": "vector"})
    assert first_event(resp) == {"type": "meta", "rag": "lexical", "llm": "local"}

    # vector requested, backend available -> meta confirms vector really ran
    monkeypatch.setattr(wiki_vector, "is_available", lambda: True)
    monkeypatch.setattr(wiki_vector, "build_context",
                        lambda q, b: "### WIKI PAGE: fake\nctx")
    with TestClient(main.app) as tc:
        resp = tc.post("/api/chat", json={"message": "hi", "history": [],
                                          "rag": "vector"})
    assert first_event(resp) == {"type": "meta", "rag": "vector", "llm": "local"}


def test_chat_accepts_rag_field_and_degrades_gracefully(tmp_wiki, monkeypatch):
    """A vector request against a server without the optional dependency must
    behave exactly like a lexical one - no error event, normal answer."""
    monkeypatch.setattr(wiki_vector, "is_available", lambda: False)

    class FakeLLM:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            return content_stream("plain answer")

    monkeypatch.setattr(main, "client", FakeLLM())
    with TestClient(main.app) as tc:
        resp = tc.post("/api/chat", json={"message": "hi", "history": [],
                                          "rag": "vector"})
    assert resp.status_code == 200
    events = [json.loads(s) for s in resp.text.splitlines() if s.strip()]
    assert {"type": "done"} in events
    assert not any(e["type"] == "error" for e in events)
