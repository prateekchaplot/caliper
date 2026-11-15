-- Caliper database schema (T03; owner: src/caliper/db.py, §6.3).
--
-- Global tables per docs/ARCHITECTURE.md §8. The per-fingerprint index tables
-- (chunks_<fp8>, chunks_<fp8>_fts, chunks_<fp8>_vec, §8.3) are NOT defined
-- here: they are created on demand at ingest time by
-- db.ensure_chunks_tables() (§8.9), because their names depend on the
-- config fingerprint.
--
-- Append-only enforcement (§8.9): runs / run_results / spot_checks / feedback
-- are protected by BEFORE UPDATE / BEFORE DELETE triggers below. `requests`
-- and its children are deliberately NOT protected (corrective edits to a
-- mis-logged interactive request are permitted; §8.9). The chunks_* trio is
-- deliberately NOT protected (FR-02 repeatability requires upsert/delete).
--
-- DDL is idempotent (IF NOT EXISTS everywhere) so ensure_schema() can be
-- applied repeatedly.

CREATE TABLE IF NOT EXISTS schema_meta (
    -- Single-row table versioning the schema (§8.9). Managed by db.py.
    schema_version INTEGER NOT NULL
);

-- §8.2 — created when an ingest or eval first sees a new fingerprint;
-- never updated or deleted.
CREATE TABLE IF NOT EXISTS config_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

-- §8.3 — NOT append-only: repeatability (FR-02) requires upsert/delete.
-- One row per fingerprint ("latest per fingerprint", §8.1).
CREATE TABLE IF NOT EXISTS ingest_runs (
    ingest_run_id INTEGER PRIMARY KEY,
    fingerprint   TEXT NOT NULL UNIQUE,
    source_ref    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    pages_ok      INTEGER,
    pages_skipped INTEGER,
    summary_json  TEXT NOT NULL
);

-- §8.7 — status lifecycle: running → complete | failed. The only permitted
-- UPDATE is that transition (enforced by the trigger below); rows are
-- created in status 'running' only, and never deleted.
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY,
    version_label TEXT NOT NULL CHECK (version_label IN ('v1', 'v2', 'v3')),
    fingerprint   TEXT NOT NULL,
    golden_hash   TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    status        TEXT NOT NULL CHECK (status IN ('running', 'complete', 'failed')),
    summary_json  TEXT NOT NULL
);

-- §8.7 — append-only.
CREATE TABLE IF NOT EXISTS run_results (
    run_id           INTEGER NOT NULL REFERENCES runs(run_id),
    question_id      TEXT NOT NULL,
    answer_type      TEXT NOT NULL CHECK (answer_type IN ('answer', 'refusal')),
    answer_text      TEXT NOT NULL,
    confidence       TEXT NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),
    hit              INTEGER NOT NULL CHECK (hit IN (0, 1)),
    refusal_expected INTEGER NOT NULL CHECK (refusal_expected IN (0, 1)),
    latency_ms       INTEGER NOT NULL,
    cost_usd         REAL NOT NULL,
    retrieval_json   TEXT NOT NULL
);

-- §8.8 — append-only; an erroneous judgment is corrected by a later row,
-- and metrics use the latest row per (run_id, question_id).
CREATE TABLE IF NOT EXISTS spot_checks (
    spot_check_id INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES runs(run_id),
    question_id   TEXT NOT NULL,
    verdict       TEXT NOT NULL CHECK (verdict IN ('correct', 'incorrect')),
    note          TEXT,
    judged_at     TEXT NOT NULL,
    judge         TEXT NOT NULL
);

-- §8.4 — every question, interactive or eval-driven. Deliberately NOT
-- append-only (corrective edits permitted, §8.9). payload_json holds the
-- exact prompts; it never contains secrets (§8.9 retention/privacy).
CREATE TABLE IF NOT EXISTS requests (
    request_id        INTEGER PRIMARY KEY,
    ts                TEXT NOT NULL,
    question          TEXT NOT NULL,
    fingerprint       TEXT NOT NULL,
    run_id            INTEGER REFERENCES runs(run_id),
    answer_type       TEXT NOT NULL CHECK (answer_type IN ('answer', 'refusal')),
    answer_text       TEXT NOT NULL,
    confidence        TEXT NOT NULL CHECK (confidence IN ('high', 'medium', 'low')),
    latency_ms        INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    embed_tokens      INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    refusal_stage     TEXT CHECK (refusal_stage IN ('threshold', 'model')),
    payload_json      TEXT NOT NULL
);

-- §8.5 — final-set observability (FR-06/FR-09). ordinal is 1..final_k.
-- chunk_id has no FOREIGN KEY: the chunks table it points into is created
-- on demand per fingerprint (§8.3), so integrity is enforced by callers.
CREATE TABLE IF NOT EXISTS request_retrievals (
    request_id INTEGER NOT NULL REFERENCES requests(request_id),
    ordinal    INTEGER NOT NULL CHECK (ordinal >= 1),
    chunk_id   INTEGER NOT NULL,
    score      REAL NOT NULL,
    stage      TEXT NOT NULL CHECK (stage IN ('vector', 'fts', 'rrf', 'rerank'))
);

-- §8.5 — refusal rows write no citations.
CREATE TABLE IF NOT EXISTS request_citations (
    request_id   INTEGER NOT NULL REFERENCES requests(request_id),
    ordinal      INTEGER NOT NULL CHECK (ordinal >= 1),
    chunk_id     INTEGER NOT NULL,
    page_url     TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    quote        TEXT NOT NULL,
    verified     INTEGER NOT NULL CHECK (verified IN (0, 1))
);

-- §8.8 — max one row per request; repeat writes return the existing row
-- (§9.2 idempotency, enforced in db.insert_feedback).
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY,
    request_id  INTEGER NOT NULL UNIQUE REFERENCES requests(request_id),
    helpful     INTEGER NOT NULL CHECK (helpful IN (0, 1)),
    note        TEXT,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_fingerprint ON requests(fingerprint);
CREATE INDEX IF NOT EXISTS idx_requests_run ON requests(run_id);
CREATE INDEX IF NOT EXISTS idx_request_retrievals_request ON request_retrievals(request_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_request_citations_request ON request_citations(request_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_run_results_run ON run_results(run_id);
CREATE INDEX IF NOT EXISTS idx_spot_checks_run_question ON spot_checks(run_id, question_id);

-- §8.9 — append-only enforcement. A runs row may only be updated while its
-- stored status is non-terminal, i.e. the single permitted UPDATE is the
-- running → complete | failed transition (which also sets finished_at /
-- summary_json in the same statement). Everything else is rejected.
CREATE TRIGGER IF NOT EXISTS runs_immutable_after_terminal
BEFORE UPDATE ON runs
WHEN OLD.status IN ('complete', 'failed')
BEGIN
    SELECT RAISE(ABORT, 'run is append-only once terminal');
END;

CREATE TRIGGER IF NOT EXISTS runs_insert_must_start_running
BEFORE INSERT ON runs
WHEN NEW.status != 'running'
BEGIN
    SELECT RAISE(ABORT, 'runs rows are created in status running');
END;

CREATE TRIGGER IF NOT EXISTS runs_append_only_delete
BEFORE DELETE ON runs
BEGIN
    SELECT RAISE(ABORT, 'runs is append-only');
END;

CREATE TRIGGER IF NOT EXISTS run_results_append_only_update
BEFORE UPDATE ON run_results
BEGIN
    SELECT RAISE(ABORT, 'run_results is append-only');
END;

CREATE TRIGGER IF NOT EXISTS run_results_append_only_delete
BEFORE DELETE ON run_results
BEGIN
    SELECT RAISE(ABORT, 'run_results is append-only');
END;

CREATE TRIGGER IF NOT EXISTS spot_checks_append_only_update
BEFORE UPDATE ON spot_checks
BEGIN
    SELECT RAISE(ABORT, 'spot_checks is append-only');
END;

CREATE TRIGGER IF NOT EXISTS spot_checks_append_only_delete
BEFORE DELETE ON spot_checks
BEGIN
    SELECT RAISE(ABORT, 'spot_checks is append-only');
END;

CREATE TRIGGER IF NOT EXISTS feedback_append_only_update
BEFORE UPDATE ON feedback
BEGIN
    SELECT RAISE(ABORT, 'feedback is append-only');
END;

CREATE TRIGGER IF NOT EXISTS feedback_append_only_delete
BEFORE DELETE ON feedback
BEGIN
    SELECT RAISE(ABORT, 'feedback is append-only');
END;
