"""T08 resume tests: per-batch commits + checkpoint resume (§10.3, §14.2).

Reuses the fixture corpus and seams from tests/integration/test_ingest.py:

- a crash mid-ingest leaves finished batches durable and a checkpoint row
  with `finished_at IS NULL`;
- resume skips embedded chunks, repairs chunks missing their vector row,
  and never re-embeds finished batches;
- a corrupt or complete checkpoint is handled without re-embedding.

No network, no patch(); monkeypatch only for env/cwd isolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeEmbeddings, fake_embedding
from test_ingest import (  # noqa: E402  (shared fixtures)
    FIXED_SHA_A,
    MINI_CORPUS,
    FakeFetcher,
    _tarball,
)

from caliper.config import CaliperConfig, ChunkingConfig, fingerprint, fp8
from caliper.db import connect
from caliper.ingest import DB_ENV_VAR, RESUME_KEY, run_ingest
from caliper.ingest.source import CACHE_DIR_ENV_VAR


@pytest.fixture()
def corpus_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """§11.2 paths pointed at tmp_path, cwd pinned there for the CLI tests."""
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "caliper.db"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


#: A crash-embedding backend: raises on the Nth embed_documents call.
class CrashEmbeddings:
    def __init__(self, fail_on_call: int) -> None:
        self._fail_on_call = fail_on_call
        self.calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        if len(self.calls) == self._fail_on_call:
            raise RuntimeError("simulated crash mid-ingest")
        return [fake_embedding(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return fake_embedding(text)


# --- 121-chunk corpus (two ≤100 batches) ----------------------------------------

_SECTIONS = "".join(f"## Section {index:03d}\n\nBody {index} text.\n\n" for index in range(121))
MANY_CORPUS: dict[str, str] = {
    "articles/azure-functions/many-sections.md": f"# Many sections\n\n{_SECTIONS}",
    "articles/service-bus-messaging/queues.md": "# Queues\n\nQueue text.\n",
}


def _resume_row(db_path: Path, full_fingerprint: str) -> dict | None:
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT summary_json, finished_at FROM ingest_runs WHERE fingerprint = ?",
            (full_fingerprint,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"summary_json": row[0], "finished_at": row[1]}


def _chunks(db_path: Path, suffix: str) -> int:
    conn = connect(db_path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM chunks_{suffix}").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


def _vectors(db_path: Path, suffix: str) -> int:
    conn = connect(db_path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM chunks_{suffix}_vec").fetchone()
    finally:
        conn.close()
    assert row is not None
    return int(row[0])


# --- crash mid-ingest: batch 1 durable, batch 2 lost -----------------------------


def test_crash_leaves_first_batch_and_checkpoint(corpus_env: Path) -> None:
    cfg = CaliperConfig(chunking=ChunkingConfig(size_tokens=1))
    db_path = corpus_env / "caliper.db"
    suffix = fp8(cfg)
    embeddings = CrashEmbeddings(fail_on_call=2)  # die on batch 2

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_ingest(
            cfg,
            fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
            embeddings=embeddings,
        )

    # Batch 1 (100 chunks) committed; batch 2 (22 chunks) never written.
    assert _chunks(db_path, suffix) == 100
    assert _vectors(db_path, suffix) == 100
    row = _resume_row(db_path, fingerprint(cfg))
    assert row is not None
    assert row["finished_at"] is None
    checkpoint = json.loads(row["summary_json"])[RESUME_KEY]
    assert checkpoint["embedded_chunks"] == 100
    assert checkpoint["total_chunks"] == 122
    assert checkpoint["version"] == 1
    assert len(embeddings.calls) == 2  # call 1 succeeded; call 2 recorded then raised


# --- resume completes the remaining batch without re-embedding ------------------


def test_resume_skips_finished_batches(corpus_env: Path) -> None:
    cfg = CaliperConfig(chunking=ChunkingConfig(size_tokens=1))
    db_path = corpus_env / "caliper.db"
    suffix = fp8(cfg)

    # First run dies after batch 1.
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_ingest(
            cfg,
            fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
            embeddings=CrashEmbeddings(fail_on_call=2),
        )
    assert _chunks(db_path, suffix) == 100

    # Resume: only the remaining 22 chunks are embedded (1 call).
    resume_embeddings = FakeEmbeddings()
    summary = run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
        embeddings=resume_embeddings,
    )

    assert summary.chunks_written == 22
    assert summary.chunks_removed == 0
    assert summary.embed_requests == 1
    assert len(resume_embeddings.calls) == 1
    assert sum(len(call.texts) for call in resume_embeddings.calls) == 22

    # Final index complete; checkpoint replaced by the §9.4 summary.
    assert _chunks(db_path, suffix) == 122
    assert _vectors(db_path, suffix) == 122
    row = _resume_row(db_path, fingerprint(cfg))
    assert row is not None
    assert row["finished_at"] is not None
    payload = json.loads(row["summary_json"])
    assert RESUME_KEY not in payload
    assert set(payload) == {
        "fingerprint",
        "source_ref",
        "pages_ok",
        "pages_skipped",
        "skipped",
        "chunks_written",
        "chunks_removed",
        "embed_requests",
        "elapsed_s",
    }


# --- repair: chunk without its vector is re-embedded alone -----------------------


def test_repair_restores_missing_vector(corpus_env: Path) -> None:
    cfg = CaliperConfig(chunking=ChunkingConfig(size_tokens=1))
    db_path = corpus_env / "caliper.db"
    suffix = fp8(cfg)

    run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
        embeddings=FakeEmbeddings(),
    )
    assert _chunks(db_path, suffix) == 122
    assert _vectors(db_path, suffix) == 122

    # Simulate a crash between chunk insert and vector insert for one row.
    conn = connect(db_path)
    try:
        conn.execute(f"DELETE FROM chunks_{suffix}_vec WHERE rowid = 1")
        conn.commit()
    finally:
        conn.close()
    assert _vectors(db_path, suffix) == 121

    # Repair embeds only the one missing vector.
    repair_embeddings = FakeEmbeddings()
    run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
        embeddings=repair_embeddings,
    )
    assert _chunks(db_path, suffix) == 122
    assert _vectors(db_path, suffix) == 122
    assert len(repair_embeddings.calls) == 1
    assert len(repair_embeddings.calls[0].texts) == 1


# --- corrupt checkpoint is ignored ----------------------------------------------


def test_corrupt_checkpoint_ignored(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    db_path = corpus_env / "caliper.db"
    suffix = fp8(cfg)

    run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MINI_CORPUS)),
        embeddings=FakeEmbeddings(),
    )

    # Corrupt the ingest_runs row: bad checkpoint + unfinished marker.
    conn = connect(db_path)
    try:
        conn.execute(
            "UPDATE ingest_runs SET finished_at = NULL, summary_json = ? WHERE fingerprint = ?",
            (json.dumps({"resume": {"version": 999, "not": "a checkpoint"}}), suffix),
        )
        conn.commit()
    finally:
        conn.close()

    # Ingest still completes and overwrites the row with a real summary.
    summary = run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MINI_CORPUS)),
        embeddings=FakeEmbeddings(),
    )
    assert summary.chunks_written == 0  # nothing new to insert
    row = _resume_row(db_path, fingerprint(cfg))
    assert row is not None and row["finished_at"] is not None


# --- checkpoint claims everything embedded: zero embed calls on resume -----------


def test_complete_checkpoint_means_no_reembed(corpus_env: Path) -> None:
    cfg = CaliperConfig(chunking=ChunkingConfig(size_tokens=1))
    db_path = corpus_env / "caliper.db"
    suffix = fp8(cfg)

    run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
        embeddings=FakeEmbeddings(),
    )
    assert _chunks(db_path, suffix) == 122
    assert _vectors(db_path, suffix) == 122

    # Pretend a previous run crashed right after its last batch: all chunks
    # committed, checkpoint says embedded == total, finished_at still NULL.
    conn = connect(db_path)
    try:
        conn.execute(
            "UPDATE ingest_runs SET finished_at = NULL, summary_json = ? WHERE fingerprint = ?",
            (
                json.dumps(
                    {
                        "resume": {
                            "version": 1,
                            "fingerprint": suffix,
                            "source_ref": f"main@{FIXED_SHA_A[:7]}",
                            "embedded_chunks": 122,
                            "total_chunks": 122,
                            "last_embedded_chunk_id": 122,
                            "updated_at": "2026-11-27T00:00:00+00:00",
                        }
                    }
                ),
                suffix,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Resume must not call the embedding backend at all.
    noop_embeddings = CrashEmbeddings(fail_on_call=1)  # must not be called
    summary = run_ingest(
        cfg,
        fetcher=FakeFetcher(_tarball(MANY_CORPUS)),
        embeddings=noop_embeddings,
    )
    assert len(noop_embeddings.calls) == 0
    assert summary.chunks_written == 0
    assert summary.embed_requests == 0
    row = _resume_row(db_path, fingerprint(cfg))
    assert row is not None and row["finished_at"] is not None
