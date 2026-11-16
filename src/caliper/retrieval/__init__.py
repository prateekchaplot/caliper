"""Retrieval core (§7.5, §9.3): question → final bounded passage set.

T09 implements `mode = "vector"`: cosine KNN top-`cfg.retrieval.final_k` over
the per-fingerprint `chunks_<fp8>_vec` table, using the stable brute-force
sqlite-vec path (`WHERE embedding MATCH ? AND k = ?`) — no ANN indexes
(§7.5). `mode = "hybrid_rrf"` (vector + FTS5 fusion in T10's `fuse.py`) and
cross-encoder reranking (T18) are not wired here and raise until they are.

Contract (§9.3): exactly `final_k` passages sorted by descending relevance —
ascending cosine distance, so LOWER `score` means MORE similar (the schema's
`distance_metric=cosine` yields ≈0 for identical vectors, per T03's tests) —
`rank` 1-based with 1 = best, and `stage = "vector"` on every passage. If the
index holds fewer rows than `final_k`, fewer passages come back and the
caller refuses (§9.3, §11.5); an empty or never-ingested index returns [].
No sentinel scores are ever fabricated.
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

    The embedding backend is built with the provider factory
    (`get_embeddings`, which owns session setup per T04's `keepalive`
    pattern). Tests inject a backend through the keyword-only `embeddings`
    seam — the suite's fakes, §15 — so the positional §9.3 contract
    `retrieve(question, conn, cfg)` is unchanged.

    Failure modes (§16): empty question, unimplemented mode, or a
    non-768-dim query embedding raise RetrievalError. An index with fewer
    rows than `final_k`, an empty index, or a missing (never-ingested) index
    return fewer passages or [] — the caller refuses per §9.3/§11.5; no
    sentinel `score` value is introduced.
    """
    if not question.strip():
        raise RetrievalError("question must be a non-empty string")
    mode = cfg.retrieval.mode
    if mode != "vector":
        raise RetrievalError(
            f"retrieval mode {mode!r} is not implemented here (hybrid_rrf is T10, rerank is T18)"
        )
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
