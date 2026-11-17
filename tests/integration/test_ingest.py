"""T07 integration tests: run_ingest on a mini-corpus + `caliper ingest` (§14.2, §15).

Covers the task's acceptance criteria end-to-end with the suite's fakes and a
test-local Fetcher fake (the established test_source.py pattern — no network,
no patch() on internals; monkeypatch only for env/cwd isolation):

- double-ingest on the mini-corpus yields identical index state (A8/FR-02);
- orphan chunks are removed with the FTS and vec tables kept in sync;
- the §9.4 summary contract holds: key set, fingerprint, source_ref, skip
  reasons (`excluded-pattern`, `parse-error: no H1`), batch-bound embed_requests;
- `caliper ingest` prints the summary JSON on stdout (exit 0) and maps a bad
  config to the §9 envelope with exit 2 (§9 error-code table).

The CLI success path runs offline by pre-seeding the corpus cache through the
public fetcher seam (cache hit ⇒ zero network) and using a corpus whose pages
are all excluded or unparseable — zero chunks, zero embed calls, so the
constructed (ollama) backend is never contacted.
"""

import io
import json
import math
import tarfile
from pathlib import Path

import pytest
from fakes import FakeEmbeddings
from typer.testing import CliRunner

from caliper.cli import app
from caliper.config import CaliperConfig, ChunkingConfig, fingerprint, fp8
from caliper.db import connect, get_ingest_run
from caliper.ingest import DB_ENV_VAR, IngestSummary, SkippedPage, run_ingest
from caliper.ingest.source import (
    CACHE_DIR_ENV_VAR,
    SourceError,
    fetch_corpus,
    format_source_ref,
    page_to_url,
)
from caliper.providers.base import EMBEDDING_DIMENSIONS

# --- fixture corpus + seams ------------------------------------------------------------

FIXED_SHA_A = "9f3ab12e4d5c6b7a8f9e0d1c2b3a495867766554"
FIXED_SHA_B = "0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e"
ROOT_NAME = f"azure-docs-{FIXED_SHA_A[:7]}"
COMMITS_URL = "https://api.github.com/repos/MicrosoftDocs/azure-docs/commits/main"


def _codeload(sha: str) -> str:
    return f"https://codeload.github.com/MicrosoftDocs/azure-docs/tar.gz/{sha}"


#: 6 markdown files (§15's scaled fixture): 4 discovered (1 unparseable) and
#: 2 excluded (`_` prefix, `/includes/` segment). The three parseable pages
#: each merge into exactly one chunk under the default 800-token heading unit.
MINI_CORPUS: dict[str, str] = {
    "articles/azure-functions/durable-functions.md": (
        "# Durable Functions\n\n"
        "Durable intro text.\n\n"
        "## Orchestrations\n\n"
        "Orchestrations coordinate agent activity.\n\n"
        "## Timers\n\n"
        "Timers schedule durable work.\n"
    ),
    "articles/azure-functions/timer-trigger.md": (
        "# Timer Trigger\n\nTimer intro text.\n\n## Usage\n\nUsage notes for the timer trigger.\n"
    ),
    "articles/service-bus-messaging/queues.md": (
        "# Service Bus queues\n\n"
        "Queue intro text.\n\n"
        "## Peek\n\n"
        "Peek at messages without locking.\n"
    ),
    "articles/azure-functions/broken.md": "no heading anywhere in here\n",
    "articles/azure-functions/_shared.md": "# Shared\n\nunderscore-excluded.\n",
    "articles/azure-functions/includes/snip.md": "# Snip\n\nincludes-excluded.\n",
}

#: v2 of the mini-corpus: queues.md disappeared upstream, timer-trigger.md
#: gained a section (its single chunk changes), durable-functions.md is
#: untouched, plus a brand-new page.
MINI_CORPUS_V2: dict[str, str] = {
    "articles/azure-functions/durable-functions.md": MINI_CORPUS[
        "articles/azure-functions/durable-functions.md"
    ],
    "articles/azure-functions/timer-trigger.md": (
        "# Timer Trigger\n\n"
        "Timer intro text.\n\n"
        "## Usage\n\n"
        "Usage notes for the timer trigger.\n\n"
        "## Retry\n\n"
        "Retry policy notes.\n"
    ),
    "articles/azure-functions/bindings.md": "# Bindings\n\nBinding intro text.\n",
    "articles/azure-functions/broken.md": MINI_CORPUS["articles/azure-functions/broken.md"],
    "articles/azure-functions/_shared.md": MINI_CORPUS["articles/azure-functions/_shared.md"],
    "articles/azure-functions/includes/snip.md": MINI_CORPUS[
        "articles/azure-functions/includes/snip.md"
    ],
    # queues.md disappeared upstream, but the tree itself must still exist
    # (discover_pages raises SourceError when either article tree is missing).
    "articles/service-bus-messaging/includes/snip.md": "# Snip\n\nx\n",
}

#: A corpus with no parseable page at all: the CLI success path can run it
#: offline (zero chunks ⇒ zero embed calls ⇒ no backend contact).
SKIP_ONLY_CORPUS: dict[str, str] = {
    "articles/azure-functions/broken.md": MINI_CORPUS["articles/azure-functions/broken.md"],
    "articles/azure-functions/_shared.md": MINI_CORPUS["articles/azure-functions/_shared.md"],
    "articles/service-bus-messaging/includes/snip.md": "# Snip\n\nx\n",
}

_SUMMARY_CONTRACT_KEYS = {
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


def _expected_skips() -> list[SkippedPage]:
    """The expected §9.4 skip list for MINI_CORPUS (sorted by path)."""
    return [
        SkippedPage(path="articles/azure-functions/_shared.md", reason="excluded-pattern"),
        SkippedPage(path="articles/azure-functions/broken.md", reason="parse-error: no H1"),
        SkippedPage(path="articles/azure-functions/includes/snip.md", reason="excluded-pattern"),
    ]


def _tarball(pages: dict[str, str]) -> bytes:
    """A real .tar.gz shaped like the upstream one (one top-level
    `azure-docs-<sha>/` directory, which run_ingest strips)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for rel_path, content in pages.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"{ROOT_NAME}/{rel_path}")
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeFetcher:
    """Serves the commits-API JSON for FIXED_SHA_A and its tarball; records
    every URL so tests can assert cache hits cause zero fetches."""

    def __init__(self, tarball_bytes: bytes) -> None:
        self._tarball = tarball_bytes
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url == COMMITS_URL:
            return json.dumps({"sha": FIXED_SHA_A}).encode("utf-8")
        if url == _codeload(FIXED_SHA_A):
            return self._tarball
        raise AssertionError(f"unexpected url {url}")


class TwoShaFetcher:
    """Serves sha A then sha B from the same instance (commits API), each with
    its own tarball — lets a `refresh=True` re-ingest present a changed corpus
    into the same database, exactly the FR-02 re-index scenario."""

    def __init__(self, tarball_a: bytes, tarball_b: bytes) -> None:
        self._tarballs = {FIXED_SHA_A: tarball_a, FIXED_SHA_B: tarball_b}
        self._pending = [FIXED_SHA_A, FIXED_SHA_B]
        self._last: str | None = None
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url == COMMITS_URL:
            assert self._pending, "TwoShaFetcher: shas exhausted"
            self._last = self._pending.pop(0)
            return json.dumps({"sha": self._last}).encode("utf-8")
        if self._last is not None and url == _codeload(self._last):
            return self._tarballs[self._last]
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture()
def corpus_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """§11.2 paths pointed at tmp_path, cwd pinned there for the CLI tests
    (monkeypatch for env/cwd isolation only, mirroring test_source.py)."""
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    monkeypatch.setenv(DB_ENV_VAR, str(tmp_path / "caliper.db"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _index_state(db_path: Path, suffix: str) -> list[tuple[object, ...]]:
    """Full chunk-table content, ordered — the FR-02 idempotency snapshot."""
    conn = connect(db_path)
    try:
        return [
            tuple(row)
            for row in conn.execute(
                "SELECT chunk_id, page_url, heading_path, body, body_tokens, chunk_hash, ordinal "
                f"FROM chunks_{suffix} ORDER BY chunk_id"
            )
        ]
    finally:
        conn.close()


def _table_count(db_path: Path, table: str) -> int:
    """Row count of a fully named table (`chunks_<fp8>`, `chunks_<fp8>_vec`, …)."""
    conn = connect(db_path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()


# --- first ingest: summary contract + index state --------------------------------------


def test_first_ingest_matches_summary_contract(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    summary = run_ingest(
        cfg, fetcher=FakeFetcher(_tarball(MINI_CORPUS)), embeddings=FakeEmbeddings()
    )

    assert summary.fingerprint == fp8(cfg)
    assert summary.source_ref == format_source_ref(FIXED_SHA_A)
    assert summary.pages_ok == 3
    assert summary.pages_skipped == 3
    assert summary.skipped == _expected_skips()
    assert summary.chunks_written == 3
    assert summary.chunks_removed == 0
    assert summary.embed_requests == 1  # 3 chunks ⇒ one ≤100-text batch
    assert summary.elapsed_s >= 0.0
    assert set(summary.as_dict()) == _SUMMARY_CONTRACT_KEYS


def test_first_ingest_writes_index_and_run_record(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    suffix = fp8(cfg)
    db_path = corpus_env / "caliper.db"
    run_ingest(cfg, fetcher=FakeFetcher(_tarball(MINI_CORPUS)), embeddings=FakeEmbeddings())

    conn = connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT page_url, heading_path, ordinal FROM chunks_{suffix} "
            "ORDER BY page_url, ordinal"
        ).fetchall()
        record = get_ingest_run(conn, fingerprint(cfg))
    finally:
        conn.close()

    assert [row[0] for row in rows] == [
        page_to_url("articles/azure-functions/durable-functions.md"),
        page_to_url("articles/azure-functions/timer-trigger.md"),
        page_to_url("articles/service-bus-messaging/queues.md"),
    ]
    # The trio stays in sync on a fresh ingest: chunks == vec rows == FTS rows.
    assert _table_count(db_path, f"chunks_{suffix}") == 3
    assert _table_count(db_path, f"chunks_{suffix}_vec") == 3
    assert _table_count(db_path, f"chunks_{suffix}_fts") == 3
    assert record is not None
    assert record.finished_at is not None
    assert record.pages_ok == 3
    assert record.pages_skipped == 3
    assert json.loads(record.summary_json)["skipped"] == [
        {"path": skip.path, "reason": skip.reason} for skip in _expected_skips()
    ]


def test_embeddings_are_stored_per_chunk(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    suffix = fp8(cfg)
    db_path = corpus_env / "caliper.db"
    run_ingest(cfg, fetcher=FakeFetcher(_tarball(MINI_CORPUS)), embeddings=FakeEmbeddings())

    conn = connect(db_path)
    try:
        rows = conn.execute(f"SELECT embedding FROM chunks_{suffix}_vec ORDER BY rowid").fetchall()
    finally:
        conn.close()

    assert len(rows) == 3
    # Stored as serialized float32: 768 dims × 4 bytes.
    assert len(rows[0][0]) == EMBEDDING_DIMENSIONS * 4


# --- FR-02/A8: idempotent double-ingest -------------------------------------------------


def test_double_ingest_yields_identical_index_and_stable_counts(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    suffix = fp8(cfg)
    db_path = corpus_env / "caliper.db"
    embeddings = FakeEmbeddings()
    fetcher = FakeFetcher(_tarball(MINI_CORPUS))

    summary1 = run_ingest(cfg, fetcher=fetcher, embeddings=embeddings)
    state1 = _index_state(db_path, suffix)
    embed_calls_after_run1 = len(embeddings.calls)
    fetches_after_run1 = len(fetcher.calls)

    summary2: IngestSummary = run_ingest(cfg, fetcher=fetcher, embeddings=embeddings)
    state2 = _index_state(db_path, suffix)

    assert summary1.chunks_written == 3
    assert summary2.chunks_written == 0
    assert summary2.chunks_removed == 0
    assert summary2.embed_requests == 0
    assert summary2.pages_ok == summary1.pages_ok
    assert summary2.pages_skipped == summary1.pages_skipped
    assert state2 == state1  # identical index state (FR-02)
    assert len(embeddings.calls) == embed_calls_after_run1  # nothing re-embedded
    assert len(fetcher.calls) == fetches_after_run1  # cache hit, zero refetch


# --- orphan removal (pages disappeared / changed upstream) -------------------------------


def test_orphans_removed_and_tables_stay_in_sync(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    suffix = fp8(cfg)
    db_path = corpus_env / "caliper.db"
    timer_url = page_to_url("articles/azure-functions/timer-trigger.md")
    fetcher = TwoShaFetcher(_tarball(MINI_CORPUS), _tarball(MINI_CORPUS_V2))

    run_ingest(cfg, fetcher=fetcher, embeddings=FakeEmbeddings())
    state1 = _index_state(db_path, suffix)
    old_timer_hashes = {row[5] for row in state1 if row[1] == timer_url}

    # refresh=True presents the changed corpus (sha B) to the same database.
    summary2 = run_ingest(cfg, refresh=True, fetcher=fetcher, embeddings=FakeEmbeddings())
    state2 = _index_state(db_path, suffix)

    # queues.md disappeared (1 orphan) and timer-trigger.md changed (old chunk
    # orphaned, new inserted); durable-functions.md is untouched (zero churn).
    assert summary2.chunks_removed == 2
    assert summary2.chunks_written == 2  # new timer chunk + bindings.md
    page_urls = {row[1] for row in state2}
    assert page_to_url("articles/service-bus-messaging/queues.md") not in page_urls
    new_timer_hashes = {row[5] for row in state2 if row[1] == timer_url}
    assert old_timer_hashes.isdisjoint(new_timer_hashes)
    assert len(new_timer_hashes) == 1
    assert _table_count(db_path, f"chunks_{suffix}") == 3
    assert _table_count(db_path, f"chunks_{suffix}_vec") == 3
    assert _table_count(db_path, f"chunks_{suffix}_fts") == 3
    # Only the new/changed chunks were embedded (2 texts ⇒ one batch).
    assert summary2.embed_requests == 1


def test_untouched_page_keeps_its_chunks_and_ids(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    suffix = fp8(cfg)
    db_path = corpus_env / "caliper.db"
    durable_url = page_to_url("articles/azure-functions/durable-functions.md")
    fetcher = TwoShaFetcher(_tarball(MINI_CORPUS), _tarball(MINI_CORPUS_V2))

    run_ingest(cfg, fetcher=fetcher, embeddings=FakeEmbeddings())
    state1 = _index_state(db_path, suffix)
    run_ingest(cfg, refresh=True, fetcher=fetcher, embeddings=FakeEmbeddings())
    state2 = _index_state(db_path, suffix)

    durable1 = [row for row in state1 if row[1] == durable_url]
    durable2 = [row for row in state2 if row[1] == durable_url]
    assert durable1 == durable2  # same chunk_id: the row was never touched


# --- embedding batch bound (§7.4: ≤100 texts/request) ------------------------------------


def test_embed_requests_respect_the_batch_cap(corpus_env: Path) -> None:
    cfg = CaliperConfig(chunking=ChunkingConfig(size_tokens=1))
    # size_tokens=1 forbids merging, so every H2 section becomes its own chunk:
    # 120 sections ⇒ >100 chunks ⇒ two batches.
    sections = "".join(f"## Section {index:03d}\n\nBody {index} text.\n\n" for index in range(120))
    pages = {
        "articles/azure-functions/many-sections.md": f"# Many sections\n\n{sections}",
        # discover_pages requires both article trees to exist (§7.4/OQ3).
        "articles/service-bus-messaging/queues.md": "# Queues\n\nQueue text.\n",
    }
    embeddings = FakeEmbeddings()

    summary = run_ingest(cfg, fetcher=FakeFetcher(_tarball(pages)), embeddings=embeddings)

    assert summary.chunks_written >= 101
    assert summary.embed_requests == math.ceil(summary.chunks_written / 100)
    assert all(len(call.texts) <= 100 for call in embeddings.calls)
    assert sum(len(call.texts) for call in embeddings.calls) == summary.chunks_written
    assert all(call.task_type == "document" for call in embeddings.calls)


# --- loud failure on a broken corpus layout ----------------------------------------------


def test_unexpected_tarball_layout_raises_source_error(corpus_env: Path) -> None:
    cfg = CaliperConfig()
    bad_layout = {"wrong-tree/foo.md": "# Foo\n\nbody\n"}
    with pytest.raises(SourceError, match="articles"):
        run_ingest(cfg, fetcher=FakeFetcher(_tarball(bad_layout)), embeddings=FakeEmbeddings())


# --- `caliper ingest` (§9.1): JSON on stdout, §9 envelope on error ------------------------


def _write_cli_configs(root: Path) -> None:
    """Minimal (all-defaults) user config + v1 preset, per §11.2's paths."""
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    (configs / "v1.toml").write_text("", encoding="utf-8")
    (configs / "caliper.sample.toml").write_text("", encoding="utf-8")


def test_cli_ingest_prints_summary_json(corpus_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIPER_PROVIDER", "ollama")  # offline transport build
    _write_cli_configs(corpus_env)
    cfg = CaliperConfig()
    fetch_corpus(cfg, fetcher=FakeFetcher(_tarball(SKIP_ONLY_CORPUS)))  # seed cache

    result = CliRunner().invoke(app, ["ingest", "--version", "v1"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert set(payload) == _SUMMARY_CONTRACT_KEYS
    assert payload["fingerprint"] == fp8(cfg)
    assert payload["source_ref"] == format_source_ref(FIXED_SHA_A)
    assert payload["pages_ok"] == 0  # nothing parseable — offline by construction
    assert payload["pages_skipped"] == 3
    assert payload["chunks_written"] == 0
    assert payload["chunks_removed"] == 0
    assert payload["embed_requests"] == 0


def test_cli_ingest_config_error_maps_to_envelope(
    corpus_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cli_configs(corpus_env)
    bad = corpus_env / "bad.toml"
    bad.write_text("chunking = { size_tokens = 999999 }\n", encoding="utf-8")
    monkeypatch.setenv("CALIPER_CONFIG", str(bad))

    result = CliRunner().invoke(app, ["ingest"])

    assert result.exit_code == 2
    envelope = json.loads(result.stderr)
    assert envelope["error"]["code"] == "config_invalid"
    assert "chunking.size_tokens" in envelope["error"]["message"]
