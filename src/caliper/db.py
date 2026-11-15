"""SQLite storage: connection setup, schema management, typed accessors (§7.3, §8).

Owns the single SQLite file. Responsibilities (§7.3): connection setup with
`sqlite-vec` loaded, schema creation/versioning (including the per-fingerprint
index tables, created on demand at ingest time, §8.9), typed accessors for
every table, and append-only enforcement helpers.

This module is a leaf: it depends on nothing (§7.3) — callers hand in
fingerprints, hashes, and timestamps as plain strings — and it contains no
business logic: no ranking, no prompt building, no metrics.

Transaction discipline: accessors execute statements but never commit; the
caller owns transaction boundaries (commit/rollback).

Append-only tables (§8.9): `runs`, `run_results`, `spot_checks`, `feedback`
are protected by triggers in schema.sql. `finish_run()` performs the single
permitted UPDATE (running → complete | failed). `requests` and its children
are deliberately updatable (corrective edits, §8.9); the chunks trio is
deliberately upsert/delete-able (FR-02).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# sqlite_vec ships no py.typed marker; its API is tiny and used only here.
import sqlite_vec  # type: ignore[import-untyped]

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# §8.3: the embedding table is `embedding float[768] distance_metric=cosine`.
EMBEDDING_DIMENSIONS = 768

_SUFFIX_RE = re.compile(r"[0-9a-f]{8}")


class DbError(Exception):
    """A database operation failed (§16 typed component exception).

    Wraps sqlite3 errors and schema/lifecycle violations; mapped to the
    §9 error envelope at the CLI/web edges.
    """


def _now_iso() -> str:
    """ISO-8601 UTC timestamp (§8 column convention)."""
    return datetime.now(UTC).isoformat()


def _execute(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> sqlite3.Cursor:
    """Execute one statement, wrapping sqlite3 errors as DbError (§16)."""
    try:
        return conn.execute(sql, params)
    except sqlite3.Error as exc:
        raise DbError(f"sqlite error: {exc}") from exc


def _executemany(conn: sqlite3.Connection, sql: str, rows: Sequence[Sequence[object]]) -> None:
    """Execute one statement over many parameter tuples, wrapped as DbError (§16)."""
    try:
        conn.executemany(sql, rows)
    except sqlite3.Error as exc:
        raise DbError(f"sqlite error: {exc}") from exc


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the Caliper database with `sqlite-vec` loaded (§7.3).

    Sets WAL journal mode and turns foreign keys on. Raises DbError if the
    sqlite-vec extension cannot be loaded (e.g. a Python build without
    extension support).
    """
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error as exc:
        raise DbError(f"could not open database at {path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception as exc:  # noqa: BLE001 - any load failure is a DbError
        conn.close()
        raise DbError(f"could not load sqlite-vec extension: {exc}") from exc
    finally:
        # Whatever happened above, extension loading must end up disabled.
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass
    _execute(conn, "PRAGMA journal_mode=WAL")
    _execute(conn, "PRAGMA foreign_keys=ON")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Apply the global schema idempotently and check the version (§8.9).

    Safe to call on every connection. If the file already carries a schema
    version other than the one this code ships, raise DbError — there are no
    downgrades and no forward migrations in v1.
    """
    try:
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except sqlite3.Error as exc:
        raise DbError(f"could not apply schema: {exc}") from exc
    _execute(
        conn,
        "INSERT OR IGNORE INTO schema_meta (schema_version) VALUES (?)",
        (SCHEMA_VERSION,),
    )
    row = _execute(conn, "SELECT MAX(schema_version) FROM schema_meta").fetchone()
    if row is None or int(row[0]) != SCHEMA_VERSION:
        found = int(row[0]) if row is not None else None
        raise DbError(f"unexpected schema_version {found!r} (this code ships v{SCHEMA_VERSION})")


# ---------------------------------------------------------------------------
# Per-fingerprint index tables (§8.3), created on demand at ingest time
# ---------------------------------------------------------------------------


def _check_suffix(suffix: str) -> None:
    if _SUFFIX_RE.fullmatch(suffix) is None:
        raise DbError(f"invalid table suffix {suffix!r}: expected 8 lowercase hex chars")


def ensure_chunks_tables(conn: sqlite3.Connection, suffix: str) -> None:
    """Create the per-fingerprint trio + FTS sync triggers, idempotently (§8.3).

    `suffix` is `config.fp8(cfg)` — the first 8 hex chars of the fingerprint.
    FTS flavor: §8.3's "contentless external-content" is implemented as an
    external-content FTS5 table (content=chunks_<fp8>, content_rowid=chunk_id),
    which stores no text itself (contentless in that sense) but can serve
    snippet() from the content table; AFTER INSERT/DELETE triggers keep the
    index in sync with orphan removal (FR-02).
    """
    _check_suffix(suffix)
    try:
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS chunks_{suffix} (
                chunk_id    INTEGER PRIMARY KEY,
                page_url    TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                body        TEXT NOT NULL,
                body_tokens INTEGER NOT NULL,
                chunk_hash  TEXT NOT NULL UNIQUE,
                ordinal     INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_{suffix}_fts USING fts5(
                body,
                content='chunks_{suffix}',
                content_rowid='chunk_id'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_{suffix}_vec USING vec0(
                embedding float[{EMBEDDING_DIMENSIONS}] distance_metric=cosine
            );
            CREATE TRIGGER IF NOT EXISTS chunks_{suffix}_fts_ai
            AFTER INSERT ON chunks_{suffix}
            BEGIN
                INSERT INTO chunks_{suffix}_fts (rowid, body) VALUES (new.chunk_id, new.body);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_{suffix}_fts_ad
            AFTER DELETE ON chunks_{suffix}
            BEGIN
                INSERT INTO chunks_{suffix}_fts (chunks_{suffix}_fts, rowid, body)
                VALUES ('delete', old.chunk_id, old.body);
            END;
            """
        )
    except sqlite3.Error as exc:
        raise DbError(f"could not create chunk tables for {suffix}: {exc}") from exc


# ---------------------------------------------------------------------------
# Row types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestRow:
    """Insert payload for `requests` (§8.4). `ts=None` means "now"."""

    question: str
    fingerprint: str
    run_id: int | None
    answer_type: str
    answer_text: str
    confidence: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    embed_tokens: int
    cost_usd: float
    refusal_stage: str | None
    payload_json: str
    ts: str | None = None


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """A stored `requests` row (§8.4)."""

    request_id: int
    ts: str
    question: str
    fingerprint: str
    run_id: int | None
    answer_type: str
    answer_text: str
    confidence: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    embed_tokens: int
    cost_usd: float
    refusal_stage: str | None
    payload_json: str


@dataclass(frozen=True, slots=True)
class RetrievalRow:
    """Insert payload for one `request_retrievals` row (§8.5)."""

    ordinal: int
    chunk_id: int
    score: float
    stage: str


@dataclass(frozen=True, slots=True)
class RetrievalRecord:
    """A stored `request_retrievals` row (§8.5)."""

    request_id: int
    ordinal: int
    chunk_id: int
    score: float
    stage: str


@dataclass(frozen=True, slots=True)
class CitationRow:
    """Insert payload for one `request_citations` row (§8.5)."""

    ordinal: int
    chunk_id: int
    page_url: str
    heading_path: str
    quote: str
    verified: int


@dataclass(frozen=True, slots=True)
class CitationRecord:
    """A stored `request_citations` row (§8.5)."""

    request_id: int
    ordinal: int
    chunk_id: int
    page_url: str
    heading_path: str
    quote: str
    verified: int


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """Everything logged for one request, reconstructable (NFR-01, A7)."""

    request: RequestRecord
    retrievals: list[RetrievalRecord]
    citations: list[CitationRecord]


@dataclass(frozen=True, slots=True)
class RunRow:
    """Insert payload for `runs` (§8.7). Rows always start in status 'running';
    `summary_json` holds a '{}' placeholder until finish_run() writes the
    §9.6 summary in the same statement as the status transition."""

    version_label: str
    fingerprint: str
    golden_hash: str
    seed: int
    started_at: str | None = None
    summary_json: str = "{}"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A stored `runs` row (§8.7)."""

    run_id: int
    version_label: str
    fingerprint: str
    golden_hash: str
    seed: int
    started_at: str | None
    finished_at: str | None
    status: str
    summary_json: str


@dataclass(frozen=True, slots=True)
class RunResultRow:
    """Insert payload for one `run_results` row (§8.7)."""

    question_id: str
    answer_type: str
    answer_text: str
    confidence: str
    hit: int
    refusal_expected: int
    latency_ms: int
    cost_usd: float
    retrieval_json: str


@dataclass(frozen=True, slots=True)
class RunResultRecord:
    """A stored `run_results` row (§8.7)."""

    run_id: int
    question_id: str
    answer_type: str
    answer_text: str
    confidence: str
    hit: int
    refusal_expected: int
    latency_ms: int
    cost_usd: float
    retrieval_json: str


@dataclass(frozen=True, slots=True)
class SpotCheckRow:
    """Insert payload for `spot_checks` (§8.8)."""

    run_id: int
    question_id: str
    verdict: str
    judged_at: str
    judge: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class SpotCheckRecord:
    """A stored `spot_checks` row (§8.8)."""

    spot_check_id: int
    run_id: int
    question_id: str
    verdict: str
    note: str | None
    judged_at: str
    judge: str


@dataclass(frozen=True, slots=True)
class FeedbackRow:
    """Insert payload for `feedback` (§8.8)."""

    request_id: int
    helpful: int
    created_at: str
    note: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackRecord:
    """A stored `feedback` row (§8.8)."""

    feedback_id: int
    request_id: int
    helpful: int
    note: str | None
    created_at: str


@dataclass(frozen=True, slots=True)
class IngestRunRow:
    """Insert/update payload for `ingest_runs` (§8.3), upserted per fingerprint."""

    fingerprint: str
    source_ref: str
    summary_json: str
    started_at: str | None = None


@dataclass(frozen=True, slots=True)
class IngestRunRecord:
    """A stored `ingest_runs` row (§8.3)."""

    ingest_run_id: int
    fingerprint: str
    source_ref: str
    started_at: str | None
    finished_at: str | None
    pages_ok: int | None
    pages_skipped: int | None
    summary_json: str


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """Insert payload for one `chunks_<fp8>` row (§8.3)."""

    page_url: str
    heading_path: str
    body: str
    body_tokens: int
    chunk_hash: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A stored `chunks_<fp8>` row (§8.3)."""

    chunk_id: int
    page_url: str
    heading_path: str
    body: str
    body_tokens: int
    chunk_hash: str
    ordinal: int


# ---------------------------------------------------------------------------
# Accessors: config_fingerprints (§8.2)
# ---------------------------------------------------------------------------


def record_fingerprint(
    conn: sqlite3.Connection, *, fingerprint: str, config_json: str, created_at: str | None = None
) -> bool:
    """Insert the fingerprint row on first sight; idempotent no-op after (§8.2).

    Returns True only when a new row was created. Rows are never updated or
    deleted (§8.2 lifecycle).
    """
    cursor = _execute(
        conn,
        """
        INSERT INTO config_fingerprints (fingerprint, config_json, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT (fingerprint) DO NOTHING
        """,
        (fingerprint, config_json, created_at or _now_iso()),
    )
    return cursor.rowcount == 1


# ---------------------------------------------------------------------------
# Accessors: ingest_runs (§8.3) — NOT append-only (idempotent upsert, §8.3)
# ---------------------------------------------------------------------------


def upsert_ingest_run(conn: sqlite3.Connection, row: IngestRunRow) -> int:
    """Insert or replace the ingest run for the row's fingerprint (§8.3).

    Replaces the prior row for the same fingerprint ("latest per fingerprint",
    §8.1), keeping FR-02's double-ingest idempotent. Returns the id of the
    upserted row — the existing row's id when it was replaced, since an
    UPDATE keeps the rowid.
    """
    cursor = _execute(
        conn,
        """
        INSERT INTO ingest_runs (fingerprint, source_ref, summary_json, started_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (fingerprint) DO UPDATE SET
            source_ref = excluded.source_ref,
            summary_json = excluded.summary_json,
            started_at = excluded.started_at,
            finished_at = NULL,
            pages_ok = NULL,
            pages_skipped = NULL
        """,
        (row.fingerprint, row.source_ref, row.summary_json, row.started_at or _now_iso()),
    )
    if cursor.lastrowid is None:
        raise DbError("ingest_runs upsert did not yield a rowid")
    return int(cursor.lastrowid)


def finish_ingest_run(
    conn: sqlite3.Connection,
    ingest_run_id: int,
    *,
    finished_at: str,
    pages_ok: int,
    pages_skipped: int,
) -> None:
    """Fill in the summary fields when an ingest completes (§8.3)."""
    _execute(
        conn,
        """
        UPDATE ingest_runs
        SET finished_at = ?, pages_ok = ?, pages_skipped = ?
        WHERE ingest_run_id = ?
        """,
        (finished_at, pages_ok, pages_skipped, ingest_run_id),
    )


def get_ingest_run(conn: sqlite3.Connection, fingerprint: str) -> IngestRunRecord | None:
    """The latest ingest run for a fingerprint, or None (§8.1)."""
    row = _execute(
        conn,
        """
        SELECT ingest_run_id, fingerprint, source_ref, started_at, finished_at,
               pages_ok, pages_skipped, summary_json
        FROM ingest_runs WHERE fingerprint = ?
        """,
        (fingerprint,),
    ).fetchone()
    if row is None:
        return None
    return IngestRunRecord(
        ingest_run_id=int(row[0]),
        fingerprint=str(row[1]),
        source_ref=str(row[2]),
        started_at=row[3],
        finished_at=row[4],
        pages_ok=row[5],
        pages_skipped=row[6],
        summary_json=str(row[7]),
    )


# ---------------------------------------------------------------------------
# Accessors: runs + run_results (§8.7) — append-only
# ---------------------------------------------------------------------------


def create_run(conn: sqlite3.Connection, row: RunRow) -> int:
    """Create a run in status 'running' and return its id (§8.7).

    The only permitted follow-up UPDATE is finish_run()'s transition to a
    terminal status; the schema triggers reject everything else (§8.9).
    """
    cursor = _execute(
        conn,
        """
        INSERT INTO runs (version_label, fingerprint, golden_hash, seed,
                          started_at, status, summary_json)
        VALUES (?, ?, ?, ?, ?, 'running', ?)
        """,
        (
            row.version_label,
            row.fingerprint,
            row.golden_hash,
            row.seed,
            row.started_at or _now_iso(),
            row.summary_json,
        ),
    )
    if cursor.lastrowid is None:
        raise DbError("runs insert did not yield a rowid")
    return int(cursor.lastrowid)


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    status: Literal["complete", "failed"],
    finished_at: str,
    summary_json: str,
) -> None:
    """The single permitted UPDATE: running → complete | failed (§8.7, §8.9).

    Writes the terminal status, finished_at, and the §9.6 summary_json in one
    statement. Raises DbError if the run does not exist or is already
    terminal (the schema trigger would abort the UPDATE otherwise).
    """
    cursor = _execute(
        conn,
        """
        UPDATE runs
        SET status = ?, finished_at = ?, summary_json = ?
        WHERE run_id = ? AND status = 'running'
        """,
        (status, finished_at, summary_json, run_id),
    )
    if cursor.rowcount != 1:
        raise DbError(f"run {run_id} is not in status 'running' (append-only once terminal, §8.9)")


def get_run(conn: sqlite3.Connection, run_id: int) -> RunRecord | None:
    """One run by id, or None (§8.7)."""
    row = _execute(
        conn,
        """
        SELECT run_id, version_label, fingerprint, golden_hash, seed,
               started_at, finished_at, status, summary_json
        FROM runs WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    return _run_record(row) if row is not None else None


def list_runs(conn: sqlite3.Connection) -> list[RunRecord]:
    """All runs, oldest first (§7.3)."""
    rows = _execute(
        conn,
        """
        SELECT run_id, version_label, fingerprint, golden_hash, seed,
               started_at, finished_at, status, summary_json
        FROM runs ORDER BY run_id
        """,
    ).fetchall()
    return [_run_record(row) for row in rows]


def _run_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=int(row[0]),
        version_label=str(row[1]),
        fingerprint=str(row[2]),
        golden_hash=str(row[3]),
        seed=int(row[4]),
        started_at=row[5],
        finished_at=row[6],
        status=str(row[7]),
        summary_json=str(row[8]),
    )


def insert_run_results(conn: sqlite3.Connection, run_id: int, rows: Sequence[RunResultRow]) -> None:
    """Append eval results for a run, batched (§8.7)."""
    _executemany(
        conn,
        """
        INSERT INTO run_results (run_id, question_id, answer_type, answer_text,
                                 confidence, hit, refusal_expected, latency_ms,
                                 cost_usd, retrieval_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                row.question_id,
                row.answer_type,
                row.answer_text,
                row.confidence,
                row.hit,
                row.refusal_expected,
                row.latency_ms,
                row.cost_usd,
                row.retrieval_json,
            )
            for row in rows
        ],
    )


def list_run_results(conn: sqlite3.Connection, run_id: int) -> list[RunResultRecord]:
    """All results for one run, ordered by question_id (§8.7)."""
    rows = _execute(
        conn,
        """
        SELECT run_id, question_id, answer_type, answer_text, confidence,
               hit, refusal_expected, latency_ms, cost_usd, retrieval_json
        FROM run_results WHERE run_id = ? ORDER BY question_id
        """,
        (run_id,),
    ).fetchall()
    return [
        RunResultRecord(
            run_id=int(row[0]),
            question_id=str(row[1]),
            answer_type=str(row[2]),
            answer_text=str(row[3]),
            confidence=str(row[4]),
            hit=int(row[5]),
            refusal_expected=int(row[6]),
            latency_ms=int(row[7]),
            cost_usd=float(row[8]),
            retrieval_json=str(row[9]),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Accessors: spot_checks (§8.8) — append-only
# ---------------------------------------------------------------------------


def insert_spot_check(conn: sqlite3.Connection, row: SpotCheckRow) -> int:
    """Append a spot-check judgment; never updated or deleted (§8.8).

    An erroneous judgment is corrected by a new row with a later judged_at;
    metrics use the latest row per (run_id, question_id). Returns its id.
    """
    cursor = _execute(
        conn,
        """
        INSERT INTO spot_checks (run_id, question_id, verdict, note, judged_at, judge)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (row.run_id, row.question_id, row.verdict, row.note, row.judged_at, row.judge),
    )
    if cursor.lastrowid is None:
        raise DbError("spot_checks insert did not yield a rowid")
    return int(cursor.lastrowid)


def list_spot_checks(conn: sqlite3.Connection, run_id: int) -> list[SpotCheckRecord]:
    """All judgments for one run, oldest first (§8.8; callers pick the latest)."""
    rows = _execute(
        conn,
        """
        SELECT spot_check_id, run_id, question_id, verdict, note, judged_at, judge
        FROM spot_checks WHERE run_id = ? ORDER BY spot_check_id
        """,
        (run_id,),
    ).fetchall()
    return [
        SpotCheckRecord(
            spot_check_id=int(row[0]),
            run_id=int(row[1]),
            question_id=str(row[2]),
            verdict=str(row[3]),
            note=row[4],
            judged_at=str(row[5]),
            judge=str(row[6]),
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Accessors: requests + children (§8.4, §8.5) — deliberately not append-only
# ---------------------------------------------------------------------------


def insert_request(conn: sqlite3.Connection, row: RequestRow) -> int:
    """Log one question (interactive or eval-driven) and return its id (§8.4)."""
    cursor = _execute(
        conn,
        """
        INSERT INTO requests (ts, question, fingerprint, run_id, answer_type,
                              answer_text, confidence, latency_ms, prompt_tokens,
                              completion_tokens, embed_tokens, cost_usd,
                              refusal_stage, payload_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row.ts or _now_iso(),
            row.question,
            row.fingerprint,
            row.run_id,
            row.answer_type,
            row.answer_text,
            row.confidence,
            row.latency_ms,
            row.prompt_tokens,
            row.completion_tokens,
            row.embed_tokens,
            row.cost_usd,
            row.refusal_stage,
            row.payload_json,
        ),
    )
    if cursor.lastrowid is None:
        raise DbError("requests insert did not yield a rowid")
    return int(cursor.lastrowid)


def insert_retrievals(
    conn: sqlite3.Connection, request_id: int, rows: Sequence[RetrievalRow]
) -> None:
    """Append the final retrieval set for one request (§8.5)."""
    _executemany(
        conn,
        """
        INSERT INTO request_retrievals (request_id, ordinal, chunk_id, score, stage)
        VALUES (?, ?, ?, ?, ?)
        """,
        [(request_id, row.ordinal, row.chunk_id, row.score, row.stage) for row in rows],
    )


def insert_citations(
    conn: sqlite3.Connection, request_id: int, rows: Sequence[CitationRow]
) -> None:
    """Append the citations for one request; refusals write none (§8.5)."""
    _executemany(
        conn,
        """
        INSERT INTO request_citations (request_id, ordinal, chunk_id, page_url,
                                       heading_path, quote, verified)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                request_id,
                row.ordinal,
                row.chunk_id,
                row.page_url,
                row.heading_path,
                row.quote,
                row.verified,
            )
            for row in rows
        ],
    )


def get_request_trace(conn: sqlite3.Connection, request_id: int) -> TraceRecord | None:
    """Reconstruct one request's full trace, or None (§7.3, A7)."""
    request_row = _execute(
        conn,
        """
        SELECT request_id, ts, question, fingerprint, run_id, answer_type,
               answer_text, confidence, latency_ms, prompt_tokens,
               completion_tokens, embed_tokens, cost_usd, refusal_stage, payload_json
        FROM requests WHERE request_id = ?
        """,
        (request_id,),
    ).fetchone()
    if request_row is None:
        return None
    request = RequestRecord(
        request_id=int(request_row[0]),
        ts=str(request_row[1]),
        question=str(request_row[2]),
        fingerprint=str(request_row[3]),
        run_id=request_row[4],
        answer_type=str(request_row[5]),
        answer_text=str(request_row[6]),
        confidence=str(request_row[7]),
        latency_ms=int(request_row[8]),
        prompt_tokens=int(request_row[9]),
        completion_tokens=int(request_row[10]),
        embed_tokens=int(request_row[11]),
        cost_usd=float(request_row[12]),
        refusal_stage=request_row[13],
        payload_json=str(request_row[14]),
    )
    retrievals = [
        RetrievalRecord(
            request_id=int(row[0]),
            ordinal=int(row[1]),
            chunk_id=int(row[2]),
            score=float(row[3]),
            stage=str(row[4]),
        )
        for row in _execute(
            conn,
            """
            SELECT request_id, ordinal, chunk_id, score, stage
            FROM request_retrievals WHERE request_id = ? ORDER BY ordinal
            """,
            (request_id,),
        ).fetchall()
    ]
    citations = [
        CitationRecord(
            request_id=int(row[0]),
            ordinal=int(row[1]),
            chunk_id=int(row[2]),
            page_url=str(row[3]),
            heading_path=str(row[4]),
            quote=str(row[5]),
            verified=int(row[6]),
        )
        for row in _execute(
            conn,
            """
            SELECT request_id, ordinal, chunk_id, page_url, heading_path, quote, verified
            FROM request_citations WHERE request_id = ? ORDER BY ordinal
            """,
            (request_id,),
        ).fetchall()
    ]
    return TraceRecord(request=request, retrievals=retrievals, citations=citations)


# ---------------------------------------------------------------------------
# Accessors: feedback (§8.8) — append-only, idempotent insert (§9.2)
# ---------------------------------------------------------------------------


def insert_feedback(conn: sqlite3.Connection, row: FeedbackRow) -> tuple[int, bool]:
    """Store feedback; repeats for the same request return the existing row.

    Returns (feedback_id, created). Idempotency is per §9.2: max one row per
    request (UNIQUE in §8.8), so repeat POSTs are no-ops, not errors.
    """
    cursor = _execute(
        conn,
        """
        INSERT INTO feedback (request_id, helpful, note, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (request_id) DO NOTHING
        """,
        (row.request_id, row.helpful, row.note, row.created_at),
    )
    if cursor.rowcount == 1:
        if cursor.lastrowid is None:
            raise DbError("feedback insert did not yield a rowid")
        return int(cursor.lastrowid), True
    existing = _execute(
        conn, "SELECT feedback_id FROM feedback WHERE request_id = ?", (row.request_id,)
    ).fetchone()
    if existing is None:
        raise DbError(
            f"feedback insert for request {row.request_id} reported no row and no conflict"
        )
    return int(existing[0]), False


def get_feedback(conn: sqlite3.Connection, request_id: int) -> FeedbackRecord | None:
    """The feedback row for one request, or None (§8.8)."""
    row = _execute(
        conn,
        """
        SELECT feedback_id, request_id, helpful, note, created_at
        FROM feedback WHERE request_id = ?
        """,
        (request_id,),
    ).fetchone()
    if row is None:
        return None
    return FeedbackRecord(
        feedback_id=int(row[0]),
        request_id=int(row[1]),
        helpful=int(row[2]),
        note=row[3],
        created_at=str(row[4]),
    )


# ---------------------------------------------------------------------------
# Accessors: chunks_<fp8> (§8.3) — NOT append-only (FR-02 repeatability)
# ---------------------------------------------------------------------------


def insert_chunks(conn: sqlite3.Connection, suffix: str, rows: Sequence[ChunkRow]) -> list[int]:
    """Insert chunk rows in order and return their ids (§8.3).

    `chunk_hash` is UNIQUE in §8.3, so re-ingesting identical content into a
    fresh (empty) table reproduces identical ids; a genuine duplicate within
    one batch raises DbError via the UNIQUE constraint. Rows are inserted
    one by one because the ids must be reported back to the caller.
    """
    _check_suffix(suffix)
    ids: list[int] = []
    for row in rows:
        cursor = _execute(
            conn,
            f"INSERT INTO chunks_{suffix} (page_url, heading_path, body, body_tokens,"
            " chunk_hash, ordinal) VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.page_url,
                row.heading_path,
                row.body,
                row.body_tokens,
                row.chunk_hash,
                row.ordinal,
            ),
        )
        if cursor.lastrowid is None:
            raise DbError("chunks insert did not yield a rowid")
        ids.append(int(cursor.lastrowid))
    return ids


def get_chunks(
    conn: sqlite3.Connection, suffix: str, chunk_ids: Sequence[int]
) -> list[ChunkRecord]:
    """Fetch chunk rows by id, preserving the caller's order (§8.3).

    Raises DbError if any id is missing — callers never reference chunks
    that are not in the index.
    """
    _check_suffix(suffix)
    if not chunk_ids:
        return []
    placeholders = ", ".join("?" for _ in chunk_ids)
    rows = _execute(
        conn,
        f"""
        SELECT chunk_id, page_url, heading_path, body, body_tokens, chunk_hash, ordinal
        FROM chunks_{suffix} WHERE chunk_id IN ({placeholders})
        """,
        list(chunk_ids),
    ).fetchall()
    by_id = {
        int(row[0]): ChunkRecord(
            chunk_id=int(row[0]),
            page_url=str(row[1]),
            heading_path=str(row[2]),
            body=str(row[3]),
            body_tokens=int(row[4]),
            chunk_hash=str(row[5]),
            ordinal=int(row[6]),
        )
        for row in rows
    }
    missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
    if missing:
        raise DbError(f"chunks_{suffix} is missing chunk ids {missing}")
    return [by_id[chunk_id] for chunk_id in chunk_ids]


def delete_chunks(conn: sqlite3.Connection, suffix: str, chunk_ids: Sequence[int]) -> None:
    """Delete chunk rows; the FTS sync trigger keeps the index aligned (§8.3).

    Used by ingest's orphan removal (FR-02/T07). Embedding rows for the same
    ids are removed first: the vec0 table is keyed by rowid = chunk_id.
    """
    _check_suffix(suffix)
    if not chunk_ids:
        return
    placeholders = ", ".join("?" for _ in chunk_ids)
    _execute(
        conn,
        f"DELETE FROM chunks_{suffix}_vec WHERE rowid IN ({placeholders})",
        list(chunk_ids),
    )
    _execute(
        conn,
        f"DELETE FROM chunks_{suffix} WHERE chunk_id IN ({placeholders})",
        list(chunk_ids),
    )


def insert_chunk_embeddings(
    conn: sqlite3.Connection, suffix: str, embeddings: Sequence[tuple[int, list[float]]]
) -> None:
    """Store L2-normalized embeddings keyed by rowid = chunk_id (§8.3).

    `embeddings` pairs each chunk_id with its float[768] vector (already
    normalized client-side by providers, §7.2 — db stores, never transforms).
    """
    _check_suffix(suffix)
    for chunk_id, vector in embeddings:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise DbError(
                f"embedding for chunk {chunk_id} has dimension {len(vector)}, "
                f"expected {EMBEDDING_DIMENSIONS}"
            )
    _executemany(
        conn,
        f"INSERT INTO chunks_{suffix}_vec (rowid, embedding) VALUES (?, ?)",
        [(chunk_id, sqlite_vec.serialize_float32(vector)) for chunk_id, vector in embeddings],
    )
