"""
Vector-retrieval variant of the wiki retriever (the UI's "Vector" toggle).

Design constraints, in order:
  1. Same chunking and same output contract as the lexical retriever
     (wiki_logic.build_context): sections produced by flatten_tables_generic +
     split_into_sections, emitted as "### WIKI PAGE: <slug>" blocks. The
     system prompt, the citation format, and the grounding guard all parse
     that marker, so both retrievers are drop-in interchangeable — the rest
     of the pipeline never knows which one built the context.
  2. Optional dependency. sentence-transformers (and the torch it drags in)
     is only needed when the toggle is actually used: install with
     `pip install -e ".[vector]"`. availability() tells the frontend whether
     to enable the toggle; everything else falls back to lexical.
  3. Local-first, like the rest of the project: embeddings are computed on
     this machine by a small sentence-transformers model, never sent to an
     API. Section vectors are cached on disk keyed by content hash, so only
     new/changed sections (e.g. a freshly ingested run report) get embedded.

Search itself is exact brute-force cosine similarity over a numpy matrix.
That is deliberate: the corpus is a few hundred sections, where exact
search is both faster and simpler than an ANN index (FAISS/HNSW pay off
around ~10^5+ vectors). The index lives behind two functions
(_embed_chunks -> matrix, build_context -> ranking), so swapping in FAISS
later is a local change.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional

import numpy as np

from . import wiki_logic

# Small, standard, 384-dim model; ~80 MB download on first use. Override for
# multilingual corpora (e.g. paraphrase-multilingual-MiniLM-L12-v2).
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Sections scoring below this cosine similarity are considered noise and never
# enter the context, even with budget to spare — mirroring the lexical
# retriever's "score > 0" cutoff. 0.20 is intentionally permissive: the point
# is to drop clearly unrelated sections, not to second-guess the ranking.
MIN_SIMILARITY = 0.20

EmbedFn = Callable[[list[str]], np.ndarray]

_model = None  # lazy singleton; loading takes seconds, do it once


def availability() -> tuple[bool, Optional[str]]:
    """(is the vector backend usable, reason if not) — the /api/rag-status
    contract, so the frontend can disable the toggle instead of failing
    only after the user picks it (same pattern as the cloud LLM toggle)."""
    try:
        import sentence_transformers  # noqa: F401
        return True, None
    except Exception:
        return False, ("sentence-transformers is not installed; "
                       "run: pip install -e \".[vector]\"")


def is_available() -> bool:
    return availability()[0]


def _default_embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vecs = _model.encode(list(texts), normalize_embeddings=True,
                         show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    """Defensive: cosine ranking assumes unit vectors; don't trust an injected
    embed_fn (or a cached file from an older one) to have normalized."""
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(norms, 1e-12)


# ----------------------------------------------------------------------
# Chunking — identical to the lexical retriever's, on purpose
# ----------------------------------------------------------------------

def _collect_chunks() -> list[dict]:
    """One chunk per wiki section, using the exact same flatten+split pipeline
    as wiki_logic.build_context, so a lexical-vs-vector comparison is about
    the ranking and nothing else."""
    chunks: list[dict] = []
    for slug in wiki_logic.load_wiki_index():
        content = wiki_logic.load_page(slug)
        if not content:
            continue
        flattened = wiki_logic.flatten_tables_generic(content)
        for heading, body in wiki_logic.split_into_sections(flattened):
            # What the model reads (context block) vs what gets embedded: the
            # embed text keeps slug+heading (they carry meaning: "journal
            # bearing theory") but drops the "### WIKI PAGE:" markup, which
            # would only add identical noise to every vector.
            embed_text = f"{slug}\n{heading}\n{body}".strip()
            if not embed_text:
                continue
            chunks.append({
                "id": hashlib.sha1(embed_text.encode("utf-8")).hexdigest(),
                "text": f"### WIKI PAGE: {slug}\n## {heading}\n{body}".strip(),
                "embed_text": embed_text,
            })
    return chunks


# ----------------------------------------------------------------------
# Embedding cache — content-hash keyed, so re-indexing is incremental
# ----------------------------------------------------------------------

def _cache_file():
    # Derived from wiki_logic.WIKI_DIR at call time (not import time) so the
    # tests' tmp-wiki monkeypatch redirects the cache too.
    return wiki_logic.WIKI_DIR / ".vector_cache" / "embeddings.npz"


def _load_cache() -> dict[str, np.ndarray]:
    f = _cache_file()
    if not f.exists():
        return {}
    try:
        data = np.load(f, allow_pickle=False)
        return {cid: data["vectors"][i] for i, cid in enumerate(data["ids"])}
    except Exception:
        return {}  # corrupt/old cache -> rebuild from scratch


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    f = _cache_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, ids=np.array(list(cache.keys())),
                        vectors=np.stack(list(cache.values())))


def _embed_chunks(chunks: list[dict], embed_fn: EmbedFn) -> np.ndarray:
    """Section-vector matrix aligned with `chunks`, embedding only sections
    whose content hash isn't cached yet (a new run report = a handful of new
    sections, not a full re-index)."""
    cache = _load_cache()
    missing = [c for c in chunks if c["id"] not in cache]
    if missing:
        vecs = np.asarray(embed_fn([c["embed_text"] for c in missing]),
                          dtype=np.float32)
        for c, v in zip(missing, vecs):
            cache[c["id"]] = v
        live = {c["id"] for c in chunks}
        cache = {k: v for k, v in cache.items() if k in live}
        _save_cache(cache)
    return _normalize_rows(np.stack([cache[c["id"]] for c in chunks]))


# ----------------------------------------------------------------------
# Retrieval — same global ranking + token-budget fill as the lexical path
# ----------------------------------------------------------------------

def _index_fallback(budget: int) -> str:
    idx_file = wiki_logic.INDEX_FILE
    idx_text = idx_file.read_text(encoding="utf-8") if idx_file.exists() else ""
    return ("### WIKI INDEX (no specific section matched)\n\n"
            + wiki_logic.trim_to_token_budget(idx_text, budget))


def build_context(query: str, budget: int, embed_fn: EmbedFn | None = None) -> str:
    """Vector twin of wiki_logic.build_context(query, index, budget).

    embed_fn is injectable for the same reason the LLM client is: tests run a
    deterministic fake and never need torch or a model download.
    """
    embed_fn = embed_fn or _default_embed
    chunks = _collect_chunks()
    if not chunks:
        return _index_fallback(budget)

    matrix = _embed_chunks(chunks, embed_fn)
    query_vec = _normalize_rows(
        np.asarray(embed_fn([query]), dtype=np.float32))[0]
    sims = matrix @ query_vec  # unit vectors -> dot product == cosine

    context_parts: list[str] = []
    used = 0
    for i in np.argsort(-sims):
        if sims[i] < MIN_SIMILARITY:
            break
        text = chunks[i]["text"]
        tok = wiki_logic.count_tokens(text)
        if used + tok <= budget:
            context_parts.append(text)
            used += tok
        else:
            remaining = budget - used
            if remaining > wiki_logic.MIN_SECTION_TOKENS:
                context_parts.append(wiki_logic.trim_to_token_budget(text, remaining))
            break

    if not context_parts:
        return _index_fallback(budget)

    return "\n\n" + ("=" * 60) + "\n\n" + "\n\n---\n\n".join(context_parts)
