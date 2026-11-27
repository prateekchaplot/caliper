"""Ingest orchestration (§7.4, §10.3): corpus → parsed pages → chunks → index.

`run_ingest(cfg)` is the §7.4 public entry point. It fetches the frozen corpus
once through source.py's pinned-ref cache, extracts the tarball (stripping the
upstream top-level `azure-docs-<sha>/` directory — T05's recorded handoff),
parses and chunks every discovered page with the configured chunker, hashes
chunks per §7.4 (`SHA-256(normalized body + url + heading path)`, `chunk_hash`
below), diffs the planned rows against the stored index (re-ingest inserts
new/changed chunks and deletes orphans — FR-02 idempotency), embeds the
inserted chunks in batches of ≤100 texts with the configured backend (HTTP 429
backoff is the backends' own bounded retry, §7.2/T04), stores the
L2-normalized vectors in the per-fingerprint `vec0` table, and records the
completed run in `ingest_runs` with the §9.4 summary JSON.

The §9.4 summary is both the return value and the JSON body `caliper ingest`
prints. Keyword-only seams (`fetcher`, `embeddings`) follow the established
injection pattern (source.py's Fetcher, T09's embeddings seam): tests inject
fakes, production uses the defaults. Transaction discipline: all writes commit
once at the end — a crashed ingest leaves the previous complete state intact
and a re-run starts from scratch; resume checkpoints are T08's job (§14.2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tarfile
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from caliper.config import CaliperConfig, fingerprint, fingerprint_payload
from caliper.db import (
    ChunkRow,
    DbError,
    IngestRunRow,
    connect,
    delete_chunks,
    ensure_chunks_tables,
    ensure_schema,
    finish_ingest_run,
    get_chunks,
    get_ingest_run,
    insert_chunk_embeddings,
    insert_chunks,
    record_fingerprint,
    upsert_ingest_run,
)
from caliper.ingest.chunkers import make_chunker
from caliper.ingest.parse import ParseError, parse_page
from caliper.ingest.source import (
    ARTICLE_TREES,
    Fetcher,
    SourceError,
    discover_pages,
    fetch_corpus,
    read_cache_meta,
)
from caliper.providers import EmbeddingBackend, get_embeddings
from caliper.providers.base import EMBED_BATCH_SIZE

_LOGGER = logging.getLogger(__name__)

#: §11.2: the DB path lives in the environment, never in the fingerprinted
#: config model — read directly here, mirroring source.py's CALIPER_CACHE_DIR
#: precedent (recorded in docs/TASKS.md Notes).
DB_ENV_VAR = "CALIPER_DB"
DEFAULT_DB_PATH = "data/caliper.db"


# --- Resume checkpoint (§10.3; stored inside ingest_runs.summary_json) ----------

#: Top-level key under which a checkpoint lives inside a summary JSON dict.
RESUME_KEY = "resume"
#: Bump if the checkpoint shape changes.
RESUME_VERSION = 1


@dataclass(frozen=True, slots=True)
class IngestCheckpoint:
    """A durable, per-fingerprint resume checkpoint stored in ingest_runs.

    A row with ``finished_at IS NULL`` marks an earlier run that died mid-batch;
    the checkpoint records exactly how many chunks were committed before the
    crash so resume never re-embeds finished batches.
    """

    version: int
    fingerprint: str
    source_ref: str
    embedded_chunks: int
    total_chunks: int
    last_embedded_chunk_id: int | None
    updated_at: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "source_ref": self.source_ref,
            "embedded_chunks": self.embedded_chunks,
            "total_chunks": self.total_chunks,
            "last_embedded_chunk_id": self.last_embedded_chunk_id,
            "updated_at": self.updated_at,
        }


def _read_checkpoint(
    conn: sqlite3.Connection, fingerprint: str, source_ref: str
) -> IngestCheckpoint | None:
    """Return a usable checkpoint row, or None if there is none worth resuming.

    Honored only when ``finished_at IS NULL`` (in-progress), the version,
    fingerprint and source_ref all match, and the counts are sane — so a
    stale or corrupt checkpoint can never claim work we did not do.
    """
    row = get_ingest_run(conn, fingerprint)
    if row is None or row.finished_at is not None:
        return None
    try:
        blob = json.loads(row.summary_json or "{}")
        resume = blob.get(RESUME_KEY)
        if not isinstance(resume, dict):
            return None
        if resume.get("version") != RESUME_VERSION:
            return None
        if resume.get("fingerprint") != fingerprint:
            return None
        if resume.get("source_ref") != source_ref:
            return None
        embedded = int(resume.get("embedded_chunks", 0))
        total = int(resume.get("total_chunks", 0))
        if total <= 0 or embedded < 0 or embedded > total:
            return None
        return IngestCheckpoint(
            version=int(resume["version"]),
            fingerprint=str(resume["fingerprint"]),
            source_ref=str(resume["source_ref"]),
            embedded_chunks=embedded,
            total_chunks=total,
            last_embedded_chunk_id=(
                int(resume["last_embedded_chunk_id"])
                if resume.get("last_embedded_chunk_id") is not None
                else None
            ),
            updated_at=str(resume.get("updated_at", "")),
        )
    except (ValueError, KeyError, TypeError):
        return None


def _write_checkpoint(
    conn: sqlite3.Connection,
    *,
    fingerprint: str,
    source_ref: str,
    full_fingerprint: str,
    embedded: int,
    total: int,
    last_chunk_id: int | None,
    started_at: str,
) -> None:
    """Persist progress inside this batch's transaction (caller commits after)."""
    row = IngestRunRow(
        fingerprint=full_fingerprint,
        source_ref=source_ref,
        summary_json=json.dumps(
            {
                RESUME_KEY: IngestCheckpoint(
                    version=RESUME_VERSION,
                    fingerprint=fingerprint,
                    source_ref=source_ref,
                    embedded_chunks=embedded,
                    total_chunks=total,
                    last_embedded_chunk_id=last_chunk_id,
                    updated_at=datetime.now(UTC).isoformat(),
                ).as_dict()
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        started_at=started_at,
    )
    upsert_ingest_run(conn, row)


# --- §9.4 summary contract -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SkippedPage:
    """One §9.4 skipped entry: a corpus-relative path and a stable reason."""

    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """The §9.4 ingestion summary: `run_ingest`'s return value and the JSON
    `caliper ingest` prints. `fingerprint` is the 8-hex `config.fp8(cfg)`
    suffix (§9.4's example is 8 chars); `pages_ok` counts discovered pages
    that parsed (H1-only pages included, zero chunks); `pages_skipped` counts
    every other `.md` under the trees (excluded patterns + parse errors)."""

    fingerprint: str
    source_ref: str
    pages_ok: int
    pages_skipped: int
    skipped: list[SkippedPage]
    chunks_written: int
    chunks_removed: int
    embed_requests: int
    elapsed_s: float

    def as_dict(self) -> dict[str, object]:
        """The §9.4 JSON object in its contract key order (json.dumps preserves
        insertion order)."""
        return {
            "fingerprint": self.fingerprint,
            "source_ref": self.source_ref,
            "pages_ok": self.pages_ok,
            "pages_skipped": self.pages_skipped,
            "skipped": [{"path": skip.path, "reason": skip.reason} for skip in self.skipped],
            "chunks_written": self.chunks_written,
            "chunks_removed": self.chunks_removed,
            "embed_requests": self.embed_requests,
            "elapsed_s": round(self.elapsed_s, 3),
        }


# --- Chunk identity (§7.4 idempotency; re-derived by T08's resume logic) ---------


def _normalize_body(body: str) -> str:
    """The "normalized body" of §7.4's chunk_hash composition: LF lines with
    per-line trailing whitespace and outer whitespace stripped. Bodies
    themselves stay verbatim (parse.py keeps the canonical quote source, §17);
    this normalization only makes hashes stable against insignificant
    whitespace churn and platform line endings."""
    return "\n".join(line.rstrip() for line in body.strip().splitlines())


def chunk_hash(page_url: str, heading_path: str, body: str) -> str:
    """SHA-256 of the §7.4 composition `normalized body + url + heading path`.

    The three fields are joined with NUL separators so no field's content can
    bleed into a neighbor's (a "url" ending where the body "begins" must not
    collide with a different split). Deterministic across runs and platforms;
    `chunk_hash` is UNIQUE in `chunks_<fp8>` (§8.3) and is the diff key that
    makes re-ingest idempotent (FR-02).
    """
    payload = "\x00".join((_normalize_body(body), page_url, heading_path))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- Orchestration helpers -------------------------------------------------------


def _db_path() -> Path:
    """The database path: CALIPER_DB env var, default `data/caliper.db` (§11.2)."""
    return Path(os.environ.get(DB_ENV_VAR, DEFAULT_DB_PATH))


def _extract(tarball: Path, workdir: Path) -> Path:
    """Extract the corpus tarball under `workdir`; return the directory that
    contains `articles/`. The upstream tarball carries exactly one top-level
    `azure-docs-<sha>/` directory, which is stripped here (T05's recorded
    handoff in docs/TASKS.md Notes); a tarball that is not itself a corpus
    root and holds no single such top-level directory is a SourceError — the
    loud A1-breakage signal (§7.4)."""
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(workdir, filter="data")  # stdlib safe-extract filter (Python 3.12)
    if (workdir / "articles").is_dir():
        return workdir
    top_level = sorted(entry for entry in workdir.iterdir() if entry.is_dir())
    if len(top_level) == 1 and (top_level[0] / "articles").is_dir():
        return top_level[0]
    raise SourceError(
        "unexpected corpus tarball layout: expected the extracted root — or "
        f"exactly one top-level directory under {workdir} — to contain 'articles/'"
    )


def _excluded_pages(extracted_root: Path, pages: Sequence[str]) -> list[str]:
    """`.md` files under the article trees that `discover_pages` dropped.

    discover_pages applies the exclusion rules and is the single source of
    truth (source.py is T05-owned/frozen, §14.4), so exclusion is computed as
    the complement against a raw scan. Each excluded file becomes a §9.4 skip
    with reason `excluded-pattern`.
    """
    all_md: set[str] = set()
    for tree in ARTICLE_TREES:
        for path in (extracted_root / tree).rglob("*.md"):
            all_md.add(path.relative_to(extracted_root).as_posix())
    return sorted(all_md.difference(pages))


def _plan_chunks(
    cfg: CaliperConfig, extracted_root: Path, pages: Sequence[str]
) -> tuple[list[ChunkRow], list[SkippedPage]]:
    """Parse and chunk every page; return hash-tagged insert rows plus skips.

    Rows are built positionally from T06's frozen `Chunk` field order
    (docs/TASKS.md Notes: "Chunk field order mirrors db.ChunkRow minus
    chunk_hash"). A ParseError skips the page with the §9.4 reason
    `parse-error: <slug>`; H1-only pages parse to zero chunks and stay
    pages_ok (T06 freeze)."""
    chunker = make_chunker(cfg)
    rows: list[ChunkRow] = []
    skips: list[SkippedPage] = []
    for rel_path in pages:
        try:
            text = (extracted_root / rel_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise SourceError(f"could not read corpus page {rel_path}: {exc}") from exc
        try:
            page = parse_page(rel_path, text)
        except ParseError as exc:
            _LOGGER.warning("skipped %s: parse-error: %s", rel_path, exc.reason)
            skips.append(SkippedPage(path=rel_path, reason=f"parse-error: {exc.reason}"))
            continue
        for chunk in chunker.chunk_page(page):
            rows.append(
                ChunkRow(
                    page_url=chunk.page_url,
                    heading_path=chunk.heading_path,
                    body=chunk.body,
                    body_tokens=chunk.body_tokens,
                    chunk_hash=chunk_hash(chunk.page_url, chunk.heading_path, chunk.body),
                    ordinal=chunk.ordinal,
                )
            )
    return rows, skips


def _diff_chunks(
    conn: sqlite3.Connection, suffix: str, rows: Sequence[ChunkRow]
) -> tuple[list[ChunkRow], list[int]]:
    """Diff the planned rows against the stored index (§7.4 idempotency).

    Returns the rows to insert (new or changed content — a changed page
    orphans its old chunks and inserts its new ones, the §7.4 "upsert") and
    the orphan chunk ids to delete (stored chunks whose (page_url, chunk_hash)
    the plan no longer contains: pages disappeared upstream or changed). Reads
    the table directly on the passed connection — db.py is T03-owned (§14.4),
    mirroring T09's KNN precedent; `suffix` always derives from
    `config.fp8(cfg)` here."""
    existing: dict[tuple[str, str], int] = {}
    for chunk_id, page_url, stored_hash in conn.execute(
        f"SELECT chunk_id, page_url, chunk_hash FROM chunks_{suffix}"
    ):
        existing[(str(page_url), str(stored_hash))] = int(chunk_id)
    planned = {(row.page_url, row.chunk_hash) for row in rows}
    orphans = sorted(orphan_id for key, orphan_id in existing.items() if key not in planned)
    inserts = [row for row in rows if (row.page_url, row.chunk_hash) not in existing]
    return inserts, orphans


def run_ingest(
    cfg: CaliperConfig,
    *,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
    embeddings: EmbeddingBackend | None = None,
) -> IngestSummary:
    """Fetch, parse, chunk, diff, embed, and index the frozen corpus (§7.4).

    `refresh` re-resolves and refetches the corpus even when cached (§7.4);
    `fetcher` and `embeddings` are the §15 injection seams (defaults: the httpx
    fetcher and `providers.get_embeddings(cfg)`). The §9.4 summary is returned
    and stored in `ingest_runs` (`upsert_ingest_run` + `finish_ingest_run`
    happen at completion per §10.3, so a crashed ingest leaves the previous
    completed row for the fingerprint untouched). All index writes commit once
    at the end; resume checkpoints are T08's job (§14.2)."""
    backend = embeddings if embeddings is not None else get_embeddings(cfg)
    started_wall = datetime.now(UTC).isoformat()
    started = time.monotonic()

    tarball = fetch_corpus(cfg, refresh=refresh, fetcher=fetcher)
    meta = read_cache_meta(cfg)
    full_fingerprint = fingerprint(cfg)
    suffix = full_fingerprint[:8]

    with tempfile.TemporaryDirectory(prefix="caliper-ingest-") as work:
        extracted_root = _extract(tarball, Path(work))
        pages = discover_pages(extracted_root)
        skipped = [
            SkippedPage(path=path, reason="excluded-pattern")
            for path in _excluded_pages(extracted_root, pages)
        ]
        rows, parse_skips = _plan_chunks(cfg, extracted_root, pages)
    skipped.extend(parse_skips)
    skipped.sort(key=lambda skip: skip.path)
    pages_ok = len(pages) - len(parse_skips)

    conn = connect(_db_path())
    try:
        ensure_schema(conn)
        ensure_chunks_tables(conn, suffix)
        record_fingerprint(
            conn,
            fingerprint=full_fingerprint,
            config_json=json.dumps(fingerprint_payload(cfg), sort_keys=True, separators=(",", ":")),
        )
        inserts, orphans = _diff_chunks(conn, suffix, rows)
        delete_chunks(conn, suffix, orphans)
        conn.commit()  # orphan removal durable before the batch loop

        # Resume: an earlier run may have committed chunks and a checkpoint and
        # then died mid-batch. We never re-embed what is already in the index:
        # `_diff_chunks` excludes committed chunks from `inserts`, and anything
        # committed but still missing its vector row gets repaired below.
        checkpoint = _read_checkpoint(conn, suffix, meta.source_ref)
        if checkpoint is not None:
            _LOGGER.info(
                "resuming ingest %s: %d/%d chunks already embedded",
                suffix,
                checkpoint.embedded_chunks,
                checkpoint.total_chunks,
            )
        existing_count_row = conn.execute(f"SELECT COUNT(*) FROM chunks_{suffix}").fetchone()
        existing_count = int(existing_count_row[0])
        total_chunks = existing_count + len(inserts)

        # Repair: chunks committed without their vector row (a crash under the
        # old all-at-once commit). Embed only those, never the rest.
        missing_ids = [
            int(row[0])
            for row in conn.execute(
                f"""
                SELECT chunk_id FROM chunks_{suffix}
                WHERE chunk_id NOT IN (SELECT rowid FROM chunks_{suffix}_vec)
                """
            )
        ]
        embed_requests = 0
        embedded = existing_count - len(missing_ids)
        for start in range(0, len(missing_ids), EMBED_BATCH_SIZE):
            batch_ids = missing_ids[start : start + EMBED_BATCH_SIZE]
            repair_rows = get_chunks(conn, suffix, batch_ids)
            vectors = backend.embed_documents([row.body for row in repair_rows])
            embed_requests += 1
            # Chunks already exist in the table; only their vectors are missing.
            insert_chunk_embeddings(
                conn,
                suffix,
                list(zip(batch_ids, vectors, strict=True)),
            )
            embedded += len(repair_rows)
            _write_checkpoint(
                conn,
                fingerprint=suffix,
                source_ref=meta.source_ref,
                full_fingerprint=full_fingerprint,
                embedded=embedded,
                total=total_chunks,
                last_chunk_id=batch_ids[-1],
                started_at=started_wall,
            )
            conn.commit()

        # Insert + embed each ≤100 chunk batch, one transaction per batch, so a
        # crash loses at most the in-flight batch and resume skips finished ones.
        # Embed before opening any write so network failure never holds a
        # transaction open; the bounded backoff lives in the backend (§7.2/T04).
        for start in range(0, len(inserts), EMBED_BATCH_SIZE):
            insert_batch = inserts[start : start + EMBED_BATCH_SIZE]
            vectors = backend.embed_documents([row.body for row in insert_batch])
            chunk_ids = insert_chunks(conn, suffix, insert_batch)
            insert_chunk_embeddings(
                conn,
                suffix,
                list(zip(chunk_ids, vectors, strict=True)),
            )
            embed_requests += 1
            embedded += len(insert_batch)
            _write_checkpoint(
                conn,
                fingerprint=suffix,
                source_ref=meta.source_ref,
                full_fingerprint=full_fingerprint,
                embedded=embedded,
                total=total_chunks,
                last_chunk_id=chunk_ids[-1],
                started_at=started_wall,
            )
            conn.commit()

        summary = IngestSummary(
            fingerprint=suffix,
            source_ref=meta.source_ref,
            pages_ok=pages_ok,
            pages_skipped=len(skipped),
            skipped=skipped,
            chunks_written=len(inserts),
            chunks_removed=len(orphans),
            embed_requests=embed_requests,
            elapsed_s=time.monotonic() - started,
        )
        upsert_ingest_run(
            conn,
            IngestRunRow(
                fingerprint=full_fingerprint,
                source_ref=meta.source_ref,
                summary_json=json.dumps(summary.as_dict()),
                started_at=started_wall,
            ),
        )
        # `upsert_ingest_run`'s lastrowid is unreliable on the ON CONFLICT
        # update path (returns 0), so re-read the row to get its id before
        # finishing it. (`db.upsert_ingest_run` is T03-owned; worked around
        # here rather than crossing the §14.4 ownership boundary in T08.)
        record = get_ingest_run(conn, full_fingerprint)
        if record is None:
            raise DbError(
                f"ingest_runs row for fingerprint {full_fingerprint} disappeared after upsert"
            )
        finish_ingest_run(
            conn,
            record.ingest_run_id,
            finished_at=datetime.now(UTC).isoformat(),
            pages_ok=summary.pages_ok,
            pages_skipped=summary.pages_skipped,
        )
        conn.commit()
    finally:
        conn.close()

    _LOGGER.info(
        "ingest %s complete: %d pages ok, %d skipped, %d chunks written, "
        "%d removed, %d embed requests in %.1fs",
        suffix,
        summary.pages_ok,
        summary.pages_skipped,
        summary.chunks_written,
        summary.chunks_removed,
        summary.embed_requests,
        summary.elapsed_s,
    )
    return summary
