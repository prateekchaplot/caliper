"""T10 unit tests: hybrid_rrf mode wiring in `retrieve` (§7.5, §9.3).

Mirrors test_retrieval's fixture style (T09): a per-fingerprint index seeded
through T03's accessors, exercised only through the public `retrieve` with
FakeEmbeddings — fakes are the only mocks (§15): no network, no patch().

The bodies are chosen so both candidate rankings are hand-derivable with the
FakeEmbeddings hashing trick: the question "timer trigger retry" shares all
three tokens with bodies 0/1 (0 identical → vector rank 1), one token with
body 2, none with bodies 3+. FTS5's default tokenizer has no stemming and the
escaped MATCH query ANDs the terms, so only bodies containing *all three*
question tokens ("timer", "trigger", "retry") become keyword candidates —
body 0 (shortest full match) then body 1, by bm25's length normalization.
"""

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import FakeEmbeddings

from caliper.config import CaliperConfig, fp8, load_config
from caliper.db import (
    ChunkRow,
    connect,
    ensure_chunks_tables,
    ensure_schema,
    insert_chunk_embeddings,
    insert_chunks,
)
from caliper.retrieval import RetrievalError, retrieve

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"

QUESTION = "timer trigger retry"

BODIES = [
    "timer trigger retry",  # vector rank 1, FTS rank 1 (all 3 terms, shortest)
    "timer trigger retry soon",  # vector rank 2, FTS rank 2 (all 3 terms, longer)
    "retry",  # vector rank 3 (1 token), FTS: no (missing 2 terms)
    "blob storage account tiers",  # no overlap with the question at all
    "service bus queue lock",
    "deployment slots staging",
]


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    """A connected, schema-initialized in-memory database (as in test_db)."""
    connection = connect(":memory:")
    ensure_schema(connection)
    yield connection
    connection.close()


def _load_cfg(tmp_path: Path, user_toml: str = "") -> CaliperConfig:
    """The v3 preset (hybrid_rrf, 20/20/60, rerank=cross-encoder), overridable."""
    user = tmp_path / "user.toml"
    user.write_text(user_toml, encoding="utf-8")
    return load_config(user, CONFIGS / "v3.toml")


def _seed_index(conn: sqlite3.Connection, cfg: CaliperConfig, bodies: list[str]) -> list[int]:
    """Create the per-fingerprint trio and fill it the way ingest would (§7.4)."""
    suffix = fp8(cfg)
    ensure_chunks_tables(conn, suffix)
    chunk_ids = insert_chunks(
        conn,
        suffix,
        [
            ChunkRow(
                page_url=f"https://learn.microsoft.com/en-us/azure/azure-functions/page-{i}",
                heading_path=f"Docs > Section {i}",
                body=body,
                body_tokens=len(body.split()),
                chunk_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                ordinal=i - 1,
            )
            for i, body in enumerate(bodies, start=1)
        ],
    )
    vectors = FakeEmbeddings().embed_documents(bodies)
    insert_chunk_embeddings(conn, suffix, list(zip(chunk_ids, vectors, strict=True)))
    return chunk_ids


def _rrf_score(rank: int, k: int = 60) -> float:
    """One candidate list's RRF contribution at a 1-based rank."""
    return 1.0 / (k + rank)


# --- end-to-end fusion on the v3 preset (§9.7) ------------------------------------


def test_hybrid_rrf_end_to_end_on_v3_preset(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    assert cfg.retrieval.mode == "hybrid_rrf"
    assert cfg.retrieval.rerank == "cross-encoder"  # T18's stage: passed through untouched
    chunk_ids = _seed_index(conn, cfg, BODIES)

    passages = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    assert len(passages) == cfg.retrieval.final_k
    assert [p.rank for p in passages] == [1, 2, 3, 4, 5]
    assert all(p.stage == "rrf" for p in passages)
    # Top 3 are exactly the hand-derived fusion: body 0 wins both lists
    # (1/61 + 1/61), body 1 is second in both (1/62 + 1/62), body 2 is
    # vector-only rank 3 (1/63). k = 60 semantics: appearing in both lists
    # beats a single top rank, and both-list bodies outrank vector-only ones.
    assert passages[0].chunk_id == chunk_ids[0]
    assert passages[0].score == pytest.approx(2.0 / 61.0, abs=1e-9)
    assert passages[1].chunk_id == chunk_ids[1]
    assert passages[1].score == pytest.approx(2.0 / 62.0, abs=1e-9)
    assert passages[2].chunk_id == chunk_ids[2]
    assert passages[2].score == pytest.approx(1.0 / 63.0, abs=1e-9)
    # Slots 4-5 come from the no-overlap tie group (vector ranks 4-6 only);
    # only their identity domain and the strictly descending fused order
    # are asserted — the k-boundary membership is vec0's choice (cf. T09).
    scores = [p.score for p in passages]
    assert scores == sorted(scores, reverse=True)
    assert set(p.chunk_id for p in passages[3:]) <= set(chunk_ids[3:])
    assert all(p.score < passages[2].score for p in passages[3:])


def test_hybrid_rrf_is_deterministic_on_fixed_input(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)

    first = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())
    second = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    assert first == second  # identical ids, scores, ranks, stage


# --- candidate depths and fusion degradation (§9.3 graceful) -----------------------


def test_vector_top_k_is_the_fusion_candidate_depth(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # vector_top_k = 1: only body 0 is a vector candidate, so bodies 2+ drop
    # out of the fusion entirely (FTS AND-semantics excludes them anyway);
    # body 1 survives on its FTS rank alone — fused result has fewer than
    # final_k rows, returned gracefully (§9.3: the caller refuses).
    cfg = _load_cfg(tmp_path, "[retrieval]\nvector_top_k = 1\n")
    chunk_ids = _seed_index(conn, cfg, BODIES)

    passages = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    assert [p.chunk_id for p in passages] == chunk_ids[:2]
    assert [p.rank for p in passages] == [1, 2]
    assert all(p.stage == "rrf" for p in passages)
    assert passages[0].score == pytest.approx(2.0 / 61.0, abs=1e-9)  # rank 1 in both lists
    assert passages[1].score == pytest.approx(_rrf_score(2), abs=1e-9)  # FTS rank 2 only


def test_fts_zero_hits_falls_back_to_vector_only_with_rrf_scores(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # No body contains these tokens: the FTS MATCH returns zero rows — NOT a
    # hard failure; the fused result is the vector list alone, still
    # RRF-scored: consecutive ranks 1..6 → 1/61 .. 1/66, cut to final_k = 5.
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)

    passages = retrieve("zzz qqq", conn, cfg, embeddings=FakeEmbeddings())

    assert len(passages) == 5
    assert [p.rank for p in passages] == [1, 2, 3, 4, 5]
    assert all(p.stage == "rrf" for p in passages)
    for passage, rank in zip(passages, range(1, 6), strict=True):
        assert passage.score == pytest.approx(_rrf_score(rank), abs=1e-9)


def test_fts_match_gracefully_absent_for_punctuation_only_question(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # "???" leaves no queryable FTS term after escaping — the keyword side is
    # skipped (never a FTS5 syntax error) and vector-only fusion still runs.
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)

    passages = retrieve("???", conn, cfg, embeddings=FakeEmbeddings())

    assert len(passages) == 5
    assert all(p.stage == "rrf" for p in passages)


# --- FTS escaping (§7.5 safety) -----------------------------------------------------


def test_question_with_quotes_matches_its_clean_equivalent(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # Embedded double quotes are stripped before quoting each term, so the
    # escaped MATCH query (and the hash-tokenized query embedding) are
    # identical to the plain question's — same fused result, byte for byte.
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)

    nasty = retrieve('"timer trigger" retry?', conn, cfg, embeddings=FakeEmbeddings())
    clean = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    assert nasty == clean


def test_question_with_fts_syntax_characters_does_not_raise(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    # Operators and injection attempts must never reach MATCH as syntax: each
    # whitespace term is quoted, so this degrades to a keyword query matching
    # nothing ANDed with terms no body holds — zero FTS hits, graceful.
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)

    passages = retrieve('timer" OR 1=1 --', conn, cfg, embeddings=FakeEmbeddings())

    assert all(p.stage == "rrf" for p in passages)
    assert len(passages) <= cfg.retrieval.final_k


# --- graceful empty/missing index (§11.5) -------------------------------------------


def test_hybrid_rrf_returns_empty_list_for_missing_index(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)  # never ingested: no per-fingerprint trio at all

    assert retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings()) == []


def test_hybrid_rrf_returns_empty_list_for_empty_index(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    ensure_chunks_tables(conn, fp8(cfg))  # trio exists, no chunks/embeddings

    assert retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings()) == []


# --- config guard (§11.4) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("zeroed", "offending"),
    [
        ({"fts_top_k": 0}, "fts_top_k"),
        ({"rrf_k": 0}, "rrf_k"),
    ],
)
def test_hybrid_rrf_without_fusion_knobs_is_a_config_error(
    conn: sqlite3.Connection, tmp_path: Path, zeroed: dict[str, int], offending: str
) -> None:
    # RetrievalConfig's model validator rejects these at load time (§11.4);
    # inject them past validation to pin the defense-in-depth guard in
    # retrieve, raised as a RetrievalError like the other hard failures.
    cfg = _load_cfg(tmp_path)
    bypassed = cfg.retrieval.model_copy(update=zeroed)
    cfg = cfg.model_copy(update={"retrieval": bypassed})

    with pytest.raises(RetrievalError, match=offending):
        retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())


# --- provider seam usage (§7.5) -------------------------------------------------------


def test_hybrid_rrf_embeds_query_once_with_query_task_type(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    embeddings = FakeEmbeddings()

    retrieve(QUESTION, conn, cfg, embeddings=embeddings)

    assert len(embeddings.calls) == 1
    (call,) = embeddings.calls
    assert call.texts == (QUESTION,)
    assert call.task_type == "query"
