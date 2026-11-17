"""Unit tests: ingest/__init__.py — chunk_hash and the §9.4 summary shape (§7.4, §15).

The hash composition is a frozen contract: T08's resume logic re-derives the
same hashes and the UNIQUE `chunk_hash` column makes re-ingest idempotent
(FR-02). run_ingest itself is covered through its public interface in
tests/integration/test_ingest.py (§15: components tested through their public
interfaces).
"""

import json

import pytest

from caliper.ingest import IngestSummary, SkippedPage, chunk_hash

BODY = "Orchestrators coordinate activity.\n\nTimers schedule work.\n"
URL = "https://learn.microsoft.com/en-us/azure/azure-functions/durable-functions"
HEADING = "Durable Functions > Timers"


def test_chunk_hash_is_deterministic() -> None:
    assert chunk_hash(URL, HEADING, BODY) == chunk_hash(URL, HEADING, BODY)


@pytest.mark.parametrize(
    ("url", "heading", "body"),
    [
        (URL + "-x", HEADING, BODY),  # different url
        (URL, HEADING + " > Deep", BODY),  # different heading path
        (URL, HEADING, BODY + "extra"),  # different body
    ],
)
def test_chunk_hash_changes_on_any_field_change(url: str, heading: str, body: str) -> None:
    assert chunk_hash(url, heading, body) != chunk_hash(URL, HEADING, BODY)


def test_chunk_hash_normalizes_line_endings_and_trailing_whitespace() -> None:
    """The normalization keeps hashes stable against line-ending and trailing
    whitespace churn (bodies themselves stay verbatim in the index)."""
    churned = "Orchestrators coordinate activity.  \r\n\r\nTimers schedule work.\r\n   \n"
    assert chunk_hash(URL, HEADING, churned) == chunk_hash(URL, HEADING, BODY)


def test_chunk_hash_separates_fields() -> None:
    """NUL separators: no re-split of (body, url, heading) may collide with a
    different split of the same characters."""
    a = chunk_hash("https://x", "b", "c")
    b = chunk_hash("https://x", "bc", "")
    c = chunk_hash("https://xb", "c", "")
    assert len({a, b, c}) == 3


def test_summary_as_dict_matches_the_9_4_contract() -> None:
    summary = IngestSummary(
        fingerprint="a1b2c3d4",
        source_ref="main@9f3ab12",
        pages_ok=297,
        pages_skipped=2,
        skipped=[
            SkippedPage(path="articles/azure-functions/_footer.md", reason="excluded-pattern"),
            SkippedPage(path="articles/azure-functions/broken.md", reason="parse-error: no H1"),
        ],
        chunks_written=3108,
        chunks_removed=12,
        embed_requests=34,
        elapsed_s=2140.0004,
    )
    data = summary.as_dict()
    assert list(data) == [  # §9.4 key order
        "fingerprint",
        "source_ref",
        "pages_ok",
        "pages_skipped",
        "skipped",
        "chunks_written",
        "chunks_removed",
        "embed_requests",
        "elapsed_s",
    ]
    assert data["skipped"] == [
        {"path": "articles/azure-functions/_footer.md", "reason": "excluded-pattern"},
        {"path": "articles/azure-functions/broken.md", "reason": "parse-error: no H1"},
    ]
    assert data["elapsed_s"] == 2140.0  # rounded for the printed JSON
    json.dumps(data)  # the contract body must serialize cleanly
