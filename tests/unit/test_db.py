"""T03 unit tests: connection setup, schema management, and accessors (§7.3, §8)."""

import json

import pytest
import sqlite_vec

from caliper.db import (
    EMBEDDING_DIMENSIONS,
    SCHEMA_VERSION,
    ChunkRow,
    CitationRow,
    DbError,
    FeedbackRow,
    IngestRunRow,
    RequestRow,
    RetrievalRow,
    RunResultRow,
    RunRow,
    SpotCheckRow,
    connect,
    create_run,
    delete_chunks,
    ensure_chunks_tables,
    ensure_schema,
    finish_ingest_run,
    finish_run,
    get_chunks,
    get_feedback,
    get_ingest_run,
    get_request_trace,
    get_run,
    insert_chunk_embeddings,
    insert_chunks,
    insert_citations,
    insert_feedback,
    insert_request,
    insert_retrievals,
    insert_run_results,
    insert_spot_check,
    list_run_results,
    list_runs,
    list_spot_checks,
    record_fingerprint,
    upsert_ingest_run,
)

NO_TS = "2026-11-16T00:00:00+00:00"


@pytest.fixture()
def db():
    """A connected, schema-initialized in-memory database."""
    conn = connect(":memory:")
    ensure_schema(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# connect + ensure_schema (§7.3, §8.9)
# ---------------------------------------------------------------------------


def test_connect_loads_sqlite_vec():
    conn = connect(":memory:")
    try:
        version = conn.execute("SELECT vec_version()").fetchone()
        assert version is not None
    finally:
        conn.close()


def test_connect_sets_wal_and_foreign_keys(tmp_path):
    conn = connect(tmp_path / "caliper.db")
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_ensure_schema_is_idempotent(db):
    ensure_schema(db)
    row = db.execute("SELECT schema_version FROM schema_meta").fetchone()
    assert int(row[0]) == SCHEMA_VERSION


def test_ensure_schema_rejects_newer_version(tmp_path):
    conn = connect(tmp_path / "caliper.db")
    try:
        conn.execute("CREATE TABLE schema_meta (schema_version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_meta (schema_version) VALUES (999)")
        with pytest.raises(DbError, match="schema_version"):
            ensure_schema(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# config_fingerprints (§8.2)
# ---------------------------------------------------------------------------


def test_record_fingerprint_is_idempotent(db):
    assert record_fingerprint(db, fingerprint="a" * 64, config_json="{}") is True
    assert record_fingerprint(db, fingerprint="a" * 64, config_json="{}") is False
    count = db.execute("SELECT COUNT(*) FROM config_fingerprints").fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# ingest_runs (§8.3)
# ---------------------------------------------------------------------------


def test_upsert_ingest_run_replaces_per_fingerprint(db):
    upsert_ingest_run(db, _ingest_row())
    finish_ingest_run(db, 1, finished_at=NO_TS, pages_ok=10, pages_skipped=2)
    upsert_ingest_run(db, _ingest_row(summary_json='{"pages": 3}'))
    record = get_ingest_run(db, "f" * 64)
    assert record is not None
    assert record.ingest_run_id == 1  # one row per fingerprint; upsert keeps the rowid
    assert record.summary_json == '{"pages": 3}'
    assert record.finished_at is None  # reset until the new run finishes


def _ingest_row(summary_json: str = "{}"):
    return IngestRunRow(
        fingerprint="f" * 64,
        source_ref="MicrosoftDocs/azure-docs@abc1234",
        summary_json=summary_json,
    )


# ---------------------------------------------------------------------------
# runs + run_results (§8.7)
# ---------------------------------------------------------------------------


def test_create_run_and_finish_run_roundtrip(db):
    run_id = create_run(
        db, RunRow(version_label="v1", fingerprint="a" * 64, golden_hash="b" * 64, seed=7)
    )
    assert get_run(db, run_id).status == "running"
    finish_run(db, run_id, status="complete", finished_at=NO_TS, summary_json='{"ok": true}')
    run = get_run(db, run_id)
    assert run.status == "complete"
    assert run.finished_at == NO_TS
    assert json.loads(run.summary_json) == {"ok": True}


def test_finish_run_rejects_unknown_run(db):
    with pytest.raises(DbError, match="not in status 'running'"):
        finish_run(db, 404, status="failed", finished_at=NO_TS, summary_json="{}")


def test_list_runs_orders_oldest_first(db):
    first = create_run(
        db, RunRow(version_label="v1", fingerprint="a" * 64, golden_hash="b" * 64, seed=7)
    )
    second = create_run(
        db, RunRow(version_label="v2", fingerprint="a" * 64, golden_hash="b" * 64, seed=7)
    )
    assert [r.run_id for r in list_runs(db)] == [first, second]


def test_insert_run_results_roundtrip(db):
    run_id = create_run(
        db, RunRow(version_label="v1", fingerprint="a" * 64, golden_hash="b" * 64, seed=7)
    )
    insert_run_results(
        db,
        run_id,
        [
            RunResultRow(
                question_id="q001",
                answer_type="answer",
                answer_text="Use the retryPolicy setting.",
                confidence="high",
                hit=1,
                refusal_expected=0,
                latency_ms=1200,
                cost_usd=0.0,
                retrieval_json="[]",
            )
        ],
    )
    results = list_run_results(db, run_id)
    assert len(results) == 1
    assert results[0].question_id == "q001"
    assert results[0].hit == 1


# ---------------------------------------------------------------------------
# spot_checks (§8.8)
# ---------------------------------------------------------------------------


def test_spot_check_judgments_append_with_latest_wins(db):
    run_id = create_run(
        db, RunRow(version_label="v1", fingerprint="a" * 64, golden_hash="b" * 64, seed=7)
    )
    first = insert_spot_check(
        db,
        SpotCheckRow(
            run_id=run_id, question_id="q001", verdict="correct", judged_at=NO_TS, judge="owner"
        ),
    )
    second = insert_spot_check(
        db,
        SpotCheckRow(
            run_id=run_id,
            question_id="q001",
            verdict="incorrect",
            judged_at="2026-11-16T01:00:00+00:00",
            judge="owner",
            note="quote was paraphrased",
        ),
    )
    records = list_spot_checks(db, run_id)
    assert [r.spot_check_id for r in records] == [first, second]
    # Metrics pick the latest row per (run_id, question_id); both rows persist.
    assert records[-1].verdict == "incorrect"


# ---------------------------------------------------------------------------
# requests + children (§8.4, §8.5)
# ---------------------------------------------------------------------------


def test_request_trace_roundtrip(db):
    request_id = insert_request(db, _request_row())
    insert_retrievals(
        db,
        request_id,
        [
            RetrievalRow(ordinal=1, chunk_id=11, score=0.91, stage="vector"),
            RetrievalRow(ordinal=2, chunk_id=12, score=0.72, stage="vector"),
        ],
    )
    insert_citations(
        db,
        request_id,
        [
            CitationRow(
                ordinal=1,
                chunk_id=11,
                page_url="https://x/y",
                heading_path="A > B",
                quote="q",
                verified=1,
            )
        ],
    )
    trace = get_request_trace(db, request_id)
    assert trace is not None
    assert trace.request.question == "How do I set retryPolicy?"
    assert [r.chunk_id for r in trace.retrievals] == [11, 12]  # ordinal order
    assert trace.citations[0].verified == 1
    assert get_request_trace(db, 999) is None


def test_request_refusal_has_no_citations(db):
    request_id = insert_request(
        db,
        _request_row(
            answer_type="refusal",
            refusal_stage="threshold",
            answer_text="I don't know.",
        ),
    )
    trace = get_request_trace(db, request_id)
    assert trace.request.answer_type == "refusal"
    assert trace.request.refusal_stage == "threshold"
    assert trace.citations == []


def _request_row(**overrides):
    fields = dict(
        question="How do I set retryPolicy?",
        fingerprint="a" * 64,
        run_id=None,
        answer_type="answer",
        answer_text="Set functions retryPolicy.",
        confidence="high",
        latency_ms=1500,
        prompt_tokens=900,
        completion_tokens=120,
        embed_tokens=40,
        cost_usd=0.0,
        refusal_stage=None,
        payload_json='{"system": "s", "user": "u"}',
        ts=NO_TS,
    )
    return RequestRow(**{**fields, **overrides})


# ---------------------------------------------------------------------------
# feedback (§8.8, §9.2)
# ---------------------------------------------------------------------------


def test_feedback_insert_is_idempotent_per_request(db):
    request_id = insert_request(db, _request_row())
    feedback_id, created = insert_feedback(
        db, FeedbackRow(request_id=request_id, helpful=1, created_at=NO_TS)
    )
    assert created is True
    same_id, created_again = insert_feedback(
        db, FeedbackRow(request_id=request_id, helpful=0, created_at=NO_TS)
    )
    assert created_again is False
    assert same_id == feedback_id
    stored = get_feedback(db, request_id)
    assert stored.helpful == 1  # the first write wins; repeats are no-ops


# ---------------------------------------------------------------------------
# chunks_<fp8> trio (§8.3) — created on demand, FTS synced by triggers
# ---------------------------------------------------------------------------


def test_ensure_chunks_tables_rejects_bad_suffix(db):
    with pytest.raises(DbError, match="suffix"):
        ensure_chunks_tables(db, "NOT-HEX")
    with pytest.raises(DbError, match="suffix"):
        ensure_chunks_tables(db, "a" * 64)


def test_chunk_tables_roundtrip_with_fts_and_vec(db):
    ensure_chunks_tables(db, "0123abcd")
    ensure_chunks_tables(db, "0123abcd")  # idempotent
    ids = insert_chunks(
        db,
        "0123abcd",
        [
            ChunkRow(
                page_url="https://learn.microsoft.com/en-us/azure/azure-functions/timer",
                heading_path="Functions > Timer trigger",
                body="Set the schedule with a CRON expression.",
                body_tokens=9,
                chunk_hash="c" * 64,
                ordinal=0,
            ),
            ChunkRow(
                page_url="https://learn.microsoft.com/en-us/azure/azure-functions/timer",
                heading_path="Functions > Timer trigger > Retry",
                body="retryPolicy controls failed invocation retries.",
                body_tokens=8,
                chunk_hash="d" * 64,
                ordinal=1,
            ),
        ],
    )
    assert ids == [1, 2]
    fetched = get_chunks(db, "0123abcd", [2, 1])
    assert [c.chunk_id for c in fetched] == [2, 1]  # caller order preserved
    assert fetched[0].heading_path.endswith("Retry")

    # FTS rowid-mapped to chunk_id, kept in sync by triggers (§8.3).
    fts_hits = db.execute(
        "SELECT rowid FROM chunks_0123abcd_fts WHERE chunks_0123abcd_fts MATCH 'retryPolicy'"
    ).fetchall()
    assert [int(row[0]) for row in fts_hits] == [2]

    # vec0: float[768] cosine, rowid = chunk_id (§8.3).
    vector = [1.0 / (EMBEDDING_DIMENSIONS**0.5)] * EMBEDDING_DIMENSIONS  # L2 norm 1.0
    insert_chunk_embeddings(db, "0123abcd", [(1, vector)])
    stored = db.execute(
        "SELECT rowid, vec_distance_cosine(embedding, ?) FROM chunks_0123abcd_vec",
        (sqlite_vec.serialize_float32(vector),),
    ).fetchone()
    assert int(stored[0]) == 1
    assert stored[1] == pytest.approx(0.0, abs=1e-6)


def test_delete_chunks_removes_fts_rows(db):
    ensure_chunks_tables(db, "0123abcd")
    (chunk_id,) = insert_chunks(
        db,
        "0123abcd",
        [
            ChunkRow(
                page_url="https://x",
                heading_path="H",
                body="retryPolicy controls retries.",
                body_tokens=5,
                chunk_hash="c" * 64,
                ordinal=0,
            )
        ],
    )
    delete_chunks(db, "0123abcd", [chunk_id])
    with pytest.raises(DbError, match="missing"):
        get_chunks(db, "0123abcd", [chunk_id])
    fts_hits = db.execute(
        "SELECT rowid FROM chunks_0123abcd_fts WHERE chunks_0123abcd_fts MATCH 'retryPolicy'"
    ).fetchall()
    assert fts_hits == []


def test_insert_chunk_embeddings_rejects_wrong_dimension(db):
    ensure_chunks_tables(db, "0123abcd")
    with pytest.raises(DbError, match="dimension"):
        insert_chunk_embeddings(db, "0123abcd", [(1, [0.1] * (EMBEDDING_DIMENSIONS - 1))])
