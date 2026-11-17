"""T09 unit tests: the vector-mode retrieval core (§7.5, §9.3).

Seeds a per-fingerprint index through T03's accessors and exercises only the
public `retrieve` interface with the suite's FakeEmbeddings — fakes are the
only mocks (§15): no network, no patch()/monkeypatch.
"""

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import FakeEmbeddings, cosine, fake_embedding

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

QUESTION = "How do I configure retries on a timer trigger?"

# Body 0 tokenizes identically to the question (FakeEmbeddings hashes
# [a-z0-9]+ tokens, punctuation ignored) → cosine distance ≈ 0. Body 1 shares
# "timer" → clearly second. Bodies 2+ share no vocabulary with the question →
# distance ≈ 1, forming the tie group the top-2 outrank.
BODIES = [
    "how do i configure retries on a timer trigger",
    "retry configuration for timer triggers",
    "azure functions hosting plans and scaling",
    "service bus queue message lock duration",
    "blob storage account properties and tiers",
    "deploy to staging slots with github actions",
]


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    """A connected, schema-initialized in-memory database (as in test_db)."""
    connection = connect(":memory:")
    ensure_schema(connection)
    yield connection
    connection.close()


def _load_cfg(tmp_path: Path, user_toml: str = "") -> CaliperConfig:
    """The v1 preset (vector mode, final_k=5), optionally user-overridden."""
    user = tmp_path / "user.toml"
    user.write_text(user_toml, encoding="utf-8")
    return load_config(user, CONFIGS / "v1.toml")


def _seed_index(conn: sqlite3.Connection, cfg: CaliperConfig, bodies: list[str]) -> list[int]:
    """Create the per-fingerprint trio and fill it the way ingest would (§7.4).

    Stored embeddings come from the same deterministic hash the question is
    embedded with, so similarity between bodies and the question is exact.
    """
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


def _expected_distances(question: str, bodies: list[str]) -> list[float]:
    """Oracle distances (1 - cosine) matching the vec0 cosine metric."""
    query = fake_embedding(question)
    return [1.0 - cosine(query, fake_embedding(body)) for body in bodies]


# --- determinism + shape (§9.3) -------------------------------------------------


def test_retrieve_is_deterministic_on_fixed_input(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)

    first = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())
    second = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    assert first == second  # identical ids, scores, ranks, stage
    assert len(first) == 5


def test_retrieve_returns_exactly_final_k_passages(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    chunk_ids = _seed_index(conn, cfg, BODIES)
    assert len(chunk_ids) == 6  # one more than final_k

    passages = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    assert len(passages) == cfg.retrieval.final_k
    assert [p.rank for p in passages] == [1, 2, 3, 4, 5]
    scores = [p.score for p in passages]
    assert scores == sorted(scores)  # distance ascending = relevance descending (§9.3)
    assert all(p.stage == "vector" for p in passages)
    # The two chunks sharing question vocabulary are guaranteed top-2; the
    # remaining slots come from the distance-tied unrelated group, whose
    # exact membership at the k-boundary is vec0's choice — only the top-2
    # identities and the strict separation are asserted.
    assert passages[0].chunk_id == chunk_ids[0]
    assert passages[1].chunk_id == chunk_ids[1]
    assert passages[0].score == pytest.approx(0.0, abs=1e-6)
    assert passages[1].score < passages[2].score


def test_retrieve_ranks_closest_chunk_first_with_cosine_scores(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    chunk_ids = _seed_index(conn, cfg, BODIES[: cfg.retrieval.final_k])
    expected = sorted(
        zip(chunk_ids, _expected_distances(QUESTION, BODIES[: cfg.retrieval.final_k]), strict=True),
        key=lambda hit: (hit[1], hit[0]),
    )

    passages = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    # The closest chunk (identical vocabulary) ranks first.
    assert passages[0].chunk_id == expected[0][0]
    assert passages[0].score == pytest.approx(expected[0][1], abs=1e-6)
    # Every score matches the vec0 cosine distance within float32 precision.
    for passage, (_, distance) in zip(
        sorted(passages, key=lambda p: p.score),
        sorted(expected, key=lambda hit: hit[1]),
        strict=True,
    ):
        assert passage.score == pytest.approx(distance, abs=1e-6)


def test_retrieve_returns_fewer_passages_when_index_has_fewer_rows(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    chunk_ids = _seed_index(conn, cfg, BODIES[:3])

    passages = retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())

    # §9.3: fewer rows than final_k → fewer passages, caller refuses; no
    # sentinel scores are fabricated (§11.5).
    assert [p.rank for p in passages] == [1, 2, 3]
    assert [p.chunk_id for p in passages] == chunk_ids  # closest first
    assert all(p.stage == "vector" for p in passages)


def test_retrieve_returns_empty_list_for_empty_index(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    ensure_chunks_tables(conn, fp8(cfg))  # trio exists, no chunks/embeddings

    assert retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings()) == []


def test_retrieve_returns_empty_list_for_missing_index(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)  # never ingested: no per-fingerprint trio at all

    assert retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings()) == []


# --- typed hard failures (§16) --------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_retrieve_rejects_empty_question(
    conn: sqlite3.Connection, tmp_path: Path, question: str
) -> None:
    cfg = _load_cfg(tmp_path)

    with pytest.raises(RetrievalError, match="non-empty"):
        retrieve(question, conn, cfg, embeddings=FakeEmbeddings())


def test_retrieve_rejects_unimplemented_mode(conn: sqlite3.Connection, tmp_path: Path) -> None:
    # hybrid_rrf is wired since T10; a mode outside the §7.5 pair still raises.
    # RetrievalConfig's Literal rejects bogus modes at load time (§11.4), so
    # this injects one past validation with pydantic's bypassing model_copy.
    cfg = _load_cfg(tmp_path)
    bogus = cfg.retrieval.model_copy(update={"mode": "semantic"})
    cfg = cfg.model_copy(update={"retrieval": bogus})

    with pytest.raises(RetrievalError, match="not implemented"):
        retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings())


def test_retrieve_rejects_wrong_embedding_dimension(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)

    with pytest.raises(RetrievalError, match="dimension"):
        retrieve(QUESTION, conn, cfg, embeddings=FakeEmbeddings(dimensions=64))


# --- provider seam usage (§7.5) -------------------------------------------------


def test_retrieve_embeds_query_once_with_query_task_type(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    embeddings = FakeEmbeddings()

    retrieve(QUESTION, conn, cfg, embeddings=embeddings)

    assert len(embeddings.calls) == 1
    (call,) = embeddings.calls
    assert call.texts == (QUESTION,)
    assert call.task_type == "query"  # embed_query, the retrieval query task type
