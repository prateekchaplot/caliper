"""T03 integration test: append-only trigger enforcement (§8.9, §15).

The four protected tables — runs, run_results, spot_checks, feedback — must
reject UPDATE and DELETE at the schema level. The one legal UPDATE is the
runs status transition running → complete | failed (§8.7). `requests` and
its children are deliberately NOT protected (corrective edits permitted,
§8.9); the chunks trio is not protected either (FR-02).
"""

import sqlite3

import pytest

from caliper.db import (
    CitationRow,
    FeedbackRow,
    RequestRow,
    RetrievalRow,
    RunResultRow,
    RunRow,
    SpotCheckRow,
    connect,
    create_run,
    ensure_schema,
    finish_run,
    insert_citations,
    insert_feedback,
    insert_request,
    insert_retrievals,
    insert_run_results,
    insert_spot_check,
)

TS = "2026-11-16T00:00:00+00:00"


@pytest.fixture()
def db():
    conn = connect(":memory:")
    ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture()
def seeded(db):
    """A run in 'running' status plus one row in each protected table."""
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
                answer_text="ok",
                confidence="high",
                hit=1,
                refusal_expected=0,
                latency_ms=100,
                cost_usd=0.0,
                retrieval_json="[]",
            )
        ],
    )
    insert_spot_check(
        db,
        SpotCheckRow(
            run_id=run_id, question_id="q001", verdict="correct", judged_at=TS, judge="owner"
        ),
    )
    request_id = insert_request(
        db,
        RequestRow(
            question="q?",
            fingerprint="a" * 64,
            run_id=run_id,
            answer_type="answer",
            answer_text="ok",
            confidence="high",
            latency_ms=100,
            prompt_tokens=10,
            completion_tokens=5,
            embed_tokens=3,
            cost_usd=0.0,
            refusal_stage=None,
            payload_json="{}",
        ),
    )
    insert_retrievals(
        db, request_id, [RetrievalRow(ordinal=1, chunk_id=1, score=0.5, stage="vector")]
    )
    insert_citations(
        db,
        request_id,
        [CitationRow(ordinal=1, chunk_id=1, page_url="u", heading_path="h", quote="t", verified=1)],
    )
    insert_feedback(db, FeedbackRow(request_id=request_id, helpful=1, created_at=TS))
    return run_id


def _assert_update_blocked(conn, table: str, column: str) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"UPDATE {table} SET {column} = 'tampered'")


def _assert_delete_blocked(conn, table: str, where: str, params: tuple = ()) -> None:
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute(f"DELETE FROM {table} WHERE {where}", params)


def test_run_results_reject_update_and_delete(db, seeded):
    _assert_update_blocked(db, "run_results", "answer_text")
    _assert_delete_blocked(db, "run_results", "run_id = ?", (seeded,))


def test_spot_checks_reject_update_and_delete(db, seeded):
    _assert_update_blocked(db, "spot_checks", "verdict")
    _assert_delete_blocked(db, "spot_checks", "run_id = ?", (seeded,))


def test_feedback_reject_update_and_delete(db, seeded):
    _assert_update_blocked(db, "feedback", "helpful")
    _assert_delete_blocked(db, "feedback", "request_id IS NOT NULL")


def test_runs_reject_delete_and_terminal_update(db, seeded):
    _assert_delete_blocked(db, "runs", "run_id = ?", (seeded,))
    finish_run(db, seeded, status="complete", finished_at=TS, summary_json="{}")
    with pytest.raises(sqlite3.IntegrityError, match="append-only once terminal"):
        db.execute("UPDATE runs SET summary_json = '{}' WHERE run_id = ?", (seeded,))


def test_runs_only_permitted_update_is_running_to_terminal(db, seeded):
    # The legal transition: status + finished_at + summary_json in one statement.
    db.execute(
        "UPDATE runs SET status = 'failed', finished_at = ?, summary_json = ? WHERE run_id = ?",
        (TS, "{}", seeded),
    )
    status = db.execute("SELECT status FROM runs WHERE run_id = ?", (seeded,)).fetchone()[0]
    assert status == "failed"


def test_runs_insert_must_start_running(db):
    with pytest.raises(sqlite3.IntegrityError, match="status running"):
        db.execute(
            "INSERT INTO runs (version_label, fingerprint, golden_hash, seed,"
            " started_at, status, summary_json)"
            " VALUES ('v1', ?, ?, 7, ?, 'complete', '{}')",
            ("a" * 64, "b" * 64, TS),
        )


def test_requests_and_children_stay_editable(db, seeded):
    # §8.9: corrective edits to a mis-logged interactive request are permitted.
    db.execute("UPDATE requests SET answer_text = 'corrected'")
    db.execute("DELETE FROM request_citations")
    db.execute("DELETE FROM request_retrievals")
    count = db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    assert count == 1
