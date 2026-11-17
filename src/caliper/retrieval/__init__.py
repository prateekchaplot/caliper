"""Retrieval core (§7.5, §9.3): question → final bounded passage set.

`mode = "vector"` (T09): cosine KNN top-`cfg.retrieval.final_k` over the
per-fingerprint `chunks_<fp8>_vec` table, using the stable brute-force
sqlite-vec path (`WHERE embedding MATCH ? AND k = ?`) — no ANN indexes
(§7.5). `mode = "hybrid_rrf"` (T10): the same brute-force KNN at candidate
depth `vector_top_k` plus an FTS5 BM25 query at depth `fts_top_k`, fused by
Reciprocal Rank Fusion (`fuse.rrf`, k = `rrf_k`) and cut to `final_k`;
`score` is then the fused RRF score (higher = more relevant) and every
passage carries `stage = "rrf"`. Cross-encoder reranking (T18) is not wired
here; `retrieval.rerank` is passed through untouched.

Contract (§9.3): exactly `final_k` passages sorted by descending relevance —
in vector mode ascending cosine distance, so LOWER `score` means MORE
similar (the schema's `distance_metric=cosine` yields ≈0 for identical
vectors, per T03's tests); in hybrid_rrf mode descending RRF score, so
HIGHER `score` means MORE relevant — `rank` 1-based with 1 = best, and the
per-mode `stage` tag on every passage. If the index holds fewer rows than
`final_k`, fewer passages come back and the caller refuses (§9.3, §11.5);
an empty or never-ingested index returns []. No sentinel scores are ever
fabricated.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import sqlite_vec  # type: ignore[import-untyped]

from caliper.config import CaliperConfig, fp8
from caliper.db import get_chunks
from caliper.providers import EmbeddingBackend, get_embeddings
from caliper.providers.base import EMBEDDING_DIMENSIONS
from caliper.retrieval.fuse import rrf


class RetrievalError(Exception):
    """Retrieval failed hard on its inputs or the index (§16 typed exception).

    Raised for caller-contract violations only: an empty question, a
    retrieval mode this module does not implement, or a query embedding whose
    dimension is not 768. Mapped to the §9 error envelope at the CLI/web
    edges (later tasks). Graceful retrieval outcomes — fewer index rows than
    `final_k`, an empty index, a missing index — never raise; they return
    fewer passages or [] (§9.3, §11.5). Dependency failures keep their own
    typed exceptions (`DbError`, `ProviderError`) and propagate unchanged.
    """


@dataclass(frozen=True, slots=True)
class Passage:
    """One ranked passage with an observable score (§7.5) and stage tag (§9.3).

    §7.5 lists six fields; `stage` — required per passage by §9.3 and by the
    `request_retrievals` schema check (`vector`|`fts`|`rrf`|`rerank`, §8.5) —
    is the seventh, defaulted to "vector", the only mode this module
    implements. Smallest reconciliation; recorded in docs/TASKS.md (T09).
    """

    chunk_id: int
    page_url: str
    heading_path: str
    body: str
    score: float
    rank: int
    stage: str = "vector"


_FP8_RE = re.compile(r"[0-9a-f]{8}")


def _check_suffix(suffix: str) -> None:
    """Validate an fp8 table suffix: 8 lowercase hex chars, like db does.

    Mirrors db's private `_check_suffix` instead of importing it (db.py is a
    single-owner file, §14.4): table names are interpolated into SQL, so the
    suffix must never be arbitrary text. `retrieve` always derives the suffix
    from `config.fp8(cfg)`, making this defense in depth.
    """
    if _FP8_RE.fullmatch(suffix) is None:
        raise RetrievalError(f"invalid table suffix {suffix!r}: expected 8 lowercase hex chars")


def _vec_table_exists(conn: sqlite3.Connection, suffix: str) -> bool:
    """True when the per-fingerprint vec0 table exists (created at ingest, §8.3)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (f"chunks_{suffix}_vec",),
    ).fetchone()
    return row is not None


def _knn(
    conn: sqlite3.Connection, suffix: str, query_vector: list[float], k: int
) -> list[tuple[int, float]]:
    """Brute-force cosine KNN over the vec0 table (§7.5): (chunk_id, distance).

    `MATCH ... AND k = ?` is the stable sqlite-vec path (bound parameters,
    verified on the pinned sqlite-vec 0.1.9); no ANN indexes are used or
    created. Fewer rows than `k` simply return fewer hits, which is how the
    §9.3 fewer-rows contract is served. sqlite errors surface as
    RetrievalError with the sqlite message attached (DEBUG detail stays in
    the cause chain).
    """
    _check_suffix(suffix)
    try:
        rows = conn.execute(
            f"SELECT rowid, distance FROM chunks_{suffix}_vec WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(query_vector), k),
        ).fetchall()
    except sqlite3.Error as exc:
        raise RetrievalError(f"vector KNN query failed on chunks_{suffix}_vec: {exc}") from exc
    return [(int(row[0]), float(row[1])) for row in rows]


def _fts_table_exists(conn: sqlite3.Connection, suffix: str) -> bool:
    """True when the per-fingerprint FTS5 table exists (created at ingest, §8.3)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (f"chunks_{suffix}_fts",),
    ).fetchone()
    return row is not None


def _fts_escape(question: str) -> str | None:
    """Escape a raw question into a safe FTS5 MATCH query string (§7.5).

    Each whitespace-separated term is wrapped in double quotes (FTS5 quoted
    strings, so punctuation cannot form query syntax) and embedded double
    quotes are stripped, so user input can never break the MATCH or inject
    operators. A term kept only if it still contains a word character — a
    purely punctuation term would tokenize to an empty phrase. Returns None
    when no queryable term survives, which the caller treats as "no keyword
    candidates" (graceful), never as an error.
    """
    terms = [
        f'"{stripped}"'
        for term in question.split()
        if (stripped := term.replace('"', "")) and any(ch.isalnum() for ch in stripped)
    ]
    return " ".join(terms) if terms else None


def _bm25(conn: sqlite3.Connection, suffix: str, question: str, k: int) -> list[int]:
    """Best-first chunk ids from the FTS5 BM25 keyword query (§7.5).

    External-content FTS5 (content='chunks_<fp8>', content_rowid='chunk_id',
    trigger-synced, T03), so bm25() reads the live bodies and the FTS rowid
    IS the chunk id — selected as `rowid AS chunk_id` (the FTS table has no
    chunk_id column of its own). `bm25()` returns negative values where
    MORE negative = MORE relevant, so ascending score is best-first; the
    `chunk_id` tie-break mirrors the vector mode's deterministic rule — both
    ties (identical bm25, e.g. duplicate bodies) and FTS's internal order
    must not leak nondeterminism into the fused ranking. Fewer rows than
    `k` (or zero rows) simply return fewer hits — graceful (§9.3). sqlite
    errors surface as RetrievalError with the message attached.
    """
    _check_suffix(suffix)
    query = _fts_escape(question)
    if query is None:
        return []
    try:
        rows = conn.execute(
            f"SELECT rowid AS chunk_id, bm25(chunks_{suffix}_fts) AS score"
            f" FROM chunks_{suffix}_fts"
            f" WHERE chunks_{suffix}_fts MATCH ? ORDER BY score, chunk_id LIMIT ?",
            (query, k),
        ).fetchall()
    except sqlite3.Error as exc:
        raise RetrievalError(f"FTS query failed on chunks_{suffix}_fts: {exc}") from exc
    return [int(row[0]) for row in rows]


def _retrieve_hybrid_rrf(
    question: str,
    conn: sqlite3.Connection,
    cfg: CaliperConfig,
    embeddings: EmbeddingBackend | None,
) -> list[Passage]:
    """hybrid_rrf mode (§7.5 binding): vector + FTS5 candidates fused by RRF.

    Vector KNN at candidate depth `vector_top_k` (20 in §9.7) + FTS5 BM25 at
    depth `fts_top_k` (20), fused by `fuse.rrf` with k = `rrf_k` (60), then
    cut to `final_k` (5). Each returned passage's `score` is its fused RRF
    score — HIGHER = MORE relevant, the opposite polarity of vector mode's
    cosine distance — and `stage = "rrf"`; `rerank` is T18's stage on top of
    this list and is deliberately untouched here.

    A missing candidate source degrades gracefully (§9.3): a never-ingested
    fingerprint (neither table) returns [], a missing vec0 or FTS5 table
    contributes an empty list so the fused result is the other list alone
    re-scored by RRF, and a MATCH with zero keyword hits does the same. An
    empty fusion returns []. No new hard failures are introduced beyond the
    config guard below and the shared question/embedding checks.
    """
    # Config guard (§11.4): a v1/v2-shaped config turned to hybrid_rrf without
    # the fusion knobs is a config error, raised as a RetrievalError like the
    # other hard failures here (RetrievalConfig's model validator already
    # rejects this at load time; this is defense in depth for bypassed models).
    if cfg.retrieval.fts_top_k < 1 or cfg.retrieval.rrf_k < 1:
        raise RetrievalError(
            "hybrid_rrf requires retrieval.fts_top_k > 0 and retrieval.rrf_k >= 1"
            " (a v1/v2 preset lacks the fusion knobs; §11.4)"
        )
    backend = embeddings if embeddings is not None else get_embeddings(cfg)
    query_vector = backend.embed_query(question)
    if len(query_vector) != EMBEDDING_DIMENSIONS:
        raise RetrievalError(
            f"query embedding has dimension {len(query_vector)}, expected {EMBEDDING_DIMENSIONS}"
        )
    suffix = fp8(cfg)
    has_vec = _vec_table_exists(conn, suffix)
    has_fts = _fts_table_exists(conn, suffix)
    if not has_vec and not has_fts:
        # Never-ingested fingerprint ≡ empty index (§11.5): graceful [] — the
        # caller refuses; distinguishing §9's `index_missing` from a plain
        # refusal is the caller's edge decision. No sentinel passages.
        return []
    vector_ids: list[int] = []
    if has_vec:
        # Same deterministic order as vector mode: ascending distance (= more
        # similar first), ascending chunk_id on ties.
        hits = _knn(conn, suffix, query_vector, cfg.retrieval.vector_top_k)
        hits.sort(key=lambda hit: (hit[1], hit[0]))
        vector_ids = [chunk_id for chunk_id, _ in hits]
    fts_ids = _bm25(conn, suffix, question, cfg.retrieval.fts_top_k) if has_fts else []
    fused = rrf(vector_ids, fts_ids, k=cfg.retrieval.rrf_k)[: cfg.retrieval.final_k]
    if not fused:
        return []
    chunks = get_chunks(conn, suffix, [chunk_id for chunk_id, _ in fused])
    return [
        Passage(
            chunk_id=chunk.chunk_id,
            page_url=chunk.page_url,
            heading_path=chunk.heading_path,
            body=chunk.body,
            score=score,
            rank=rank,
            stage="rrf",
        )
        for rank, (chunk, (_, score)) in enumerate(zip(chunks, fused, strict=True), start=1)
    ]


def retrieve(
    question: str,
    conn: sqlite3.Connection,
    cfg: CaliperConfig,
    *,
    embeddings: EmbeddingBackend | None = None,
) -> list[Passage]:
    """Turn a question into the final bounded passage set (§7.5, §9.3).

    Vector mode: embed the question via the provider seam — `embed_query`,
    which carries the retrieval *query* task type inside each backend and
    emits 768-dim vectors (§7.2) — then cosine KNN top `cfg.retrieval.final_k`
    over the fingerprint's vec0 table. Returns exactly `final_k` passages
    ranked 1 (most similar) → `final_k`, `score` = cosine distance (lower is
    more similar), `stage = "vector"`; deterministic on fixed input (ties are
    broken by ascending chunk_id).

    hybrid_rrf mode (T10, §7.5): the same embedding seam, then vector KNN top
    `cfg.retrieval.vector_top_k` + FTS5 BM25 top `cfg.retrieval.fts_top_k`
    fused by `fuse.rrf` (k = `cfg.retrieval.rrf_k`) and cut to `final_k`.
    `score` = the fused RRF score (higher is more relevant), `stage = "rrf"`;
    keyword candidates are RRF-scored even when the FTS side is empty.

    The embedding backend is built with the provider factory
    (`get_embeddings`, which owns session setup per T04's `keepalive`
    pattern). Tests inject a backend through the keyword-only `embeddings`
    seam — the suite's fakes, §15 — so the positional §9.3 contract
    `retrieve(question, conn, cfg)` is unchanged.

    Failure modes (§16): empty question, unimplemented mode, a hybrid_rrf
    config missing the fusion knobs (§11.4 config guard), or a non-768-dim
    query embedding raise RetrievalError. An index with fewer rows than
    `final_k`, an empty index, or a missing (never-ingested) index return
    fewer passages or [] — the caller refuses per §9.3/§11.5; no sentinel
    `score` value is introduced.
    """
    if not question.strip():
        raise RetrievalError("question must be a non-empty string")
    mode = cfg.retrieval.mode
    if mode == "hybrid_rrf":
        return _retrieve_hybrid_rrf(question, conn, cfg, embeddings)
    if mode != "vector":
        raise RetrievalError(f"retrieval mode {mode!r} is not implemented here (rerank is T18)")
    backend = embeddings if embeddings is not None else get_embeddings(cfg)
    query_vector = backend.embed_query(question)
    if len(query_vector) != EMBEDDING_DIMENSIONS:
        raise RetrievalError(
            f"query embedding has dimension {len(query_vector)}, expected {EMBEDDING_DIMENSIONS}"
        )
    suffix = fp8(cfg)
    if not _vec_table_exists(conn, suffix):
        # Never-ingested fingerprint ≡ empty index (§11.5): graceful [] — the
        # caller refuses; distinguishing §9's `index_missing` from a plain
        # refusal is the caller's edge decision. No sentinel passages.
        return []
    hits = _knn(conn, suffix, query_vector, cfg.retrieval.final_k)
    # Deterministic order regardless of vec0's internal tie handling:
    # ascending distance (= descending relevance), chunk_id as tie-break.
    hits.sort(key=lambda hit: (hit[1], hit[0]))
    chunks = get_chunks(conn, suffix, [chunk_id for chunk_id, _ in hits])
    return [
        Passage(
            chunk_id=chunk.chunk_id,
            page_url=chunk.page_url,
            heading_path=chunk.heading_path,
            body=chunk.body,
            score=distance,
            rank=rank,
            stage="vector",
        )
        for rank, (chunk, (_, distance)) in enumerate(zip(chunks, hits, strict=True), start=1)
    ]
