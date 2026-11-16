"""Unit tests: ingest/source.py — pinned-ref tarball fetch/cache, A1 URL mapping,
and corpus page discovery with exclusions (§7.4, §15).

The single Fetcher seam is injected through fetch_corpus's public `fetcher`
parameter: a test-local fake serving a locally built fixture .tar.gz — no
network, no patch() on internals (§15 mocking policy; same injection style as
the provider tests' FakeGeminiModels). Cache/env tests run under tmp_path via
monkeypatch, like the provider env tests.
"""

import io
import json
import tarfile
from pathlib import Path

import pytest

from caliper.config import CaliperConfig
from caliper.ingest.source import (
    CACHE_DIR_ENV_VAR,
    CacheMeta,
    SourceError,
    _fetch_via_toc_api,
    discover_pages,
    fetch_corpus,
    format_source_ref,
    page_to_url,
    read_cache_meta,
)

# --- fixture constants -------------------------------------------------------------

# 40-hex scripted SHAs (never a real commit; only the format is contract).
FIXED_SHA_A = "9f3ab12e4d5c6b7a8f9e0d1c2b3a495867766554"
FIXED_SHA_B = "0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e"

# The wire URLs are pinned as literals so the test is honest about the format.
COMMITS_URL = "https://api.github.com/repos/MicrosoftDocs/azure-docs/commits/main"
CODELOAD_URL = "https://codeload.github.com/MicrosoftDocs/azure-docs/tar.gz/{sha}"
ROOT_NAME_A = f"azure-docs-{FIXED_SHA_A[:7]}"

LEARN = "https://learn.microsoft.com/en-us/azure"


def _fixture_tarball(root_name: str, pages: dict[str, str]) -> bytes:
    """A real .tar.gz with in-memory members, shaped like the upstream tarball
    (one top-level `azure-docs-<sha>/` directory — T07 strips it on extract)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for rel_path, content in pages.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=f"{root_name}/{rel_path}")
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _write_tree(root: Path, pages: dict[str, str]) -> None:
    """Materialize an extracted-corpus directory tree for discovery tests."""
    for rel_path, content in pages.items():
        path = root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


class FakeFetcher:
    """Fake of source.py's single Fetcher seam (§15).

    Serves the commits-API JSON for the scripted SHAs (popped in order; the
    last one repeats) and the fixture tarball bytes for the matching codeload
    URL. Every fetched URL is recorded in `calls` so tests can assert call
    counts (cache hit ⇒ 0, refresh ⇒ refetch). No network, no patch().
    """

    def __init__(self, tarball: bytes, *shas: str) -> None:
        self._tarball = tarball
        self._pending = list(shas) or [FIXED_SHA_A]
        self._last_sha: str | None = None
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url == COMMITS_URL:
            if self._pending:
                self._last_sha = self._pending.pop(0)
            assert self._last_sha is not None, "FakeFetcher: no sha scripted"
            return json.dumps({"sha": self._last_sha}).encode("utf-8")
        if self._last_sha is not None and url == CODELOAD_URL.format(sha=self._last_sha):
            return self._tarball
        raise AssertionError(f"FakeFetcher: unexpected url {url}")


@pytest.fixture(autouse=True)
def _isolate_cache_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ambient CALIPER_CACHE_DIR, and cwd pinned to tmp_path so the
    relative `data/cache` default can never write into the repo."""
    monkeypatch.delenv(CACHE_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)


# --- fetch_corpus: fetch once, cache, pin -------------------------------------------


def test_fetch_corpus_downloads_and_pins_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tarball = _fixture_tarball(
        ROOT_NAME_A,
        {"articles/azure-functions/functions-timer.md": "# Timer triggers\n"},
    )
    fake = FakeFetcher(tarball, FIXED_SHA_A)
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))

    path = fetch_corpus(CaliperConfig(), fetcher=fake)

    assert fake.calls == [COMMITS_URL, CODELOAD_URL.format(sha=FIXED_SHA_A)]
    assert path.is_file()
    assert path.read_bytes() == tarball
    assert path == tmp_path / "cache" / "azure-docs" / f"{FIXED_SHA_A}.tar.gz"
    pointer = json.loads(
        (tmp_path / "cache" / "azure-docs" / "main.json").read_text(encoding="utf-8")
    )
    assert pointer["branch"] == "main"
    assert pointer["sha"] == FIXED_SHA_A
    assert pointer["source_ref"] == f"main@{FIXED_SHA_A[:7]}"
    assert pointer["fetched_at"]


def test_fetch_corpus_cache_hit_makes_no_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetcher(_fixture_tarball(ROOT_NAME_A, {}), FIXED_SHA_A)
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))

    first = fetch_corpus(CaliperConfig(), fetcher=fake)
    assert len(fake.calls) == 2  # ref resolution + download

    second = fetch_corpus(CaliperConfig(), fetcher=fake)

    assert second == first
    assert fake.calls == [COMMITS_URL, CODELOAD_URL.format(sha=FIXED_SHA_A)]  # 0 new calls


def test_fetch_corpus_refresh_reresolves_refetches_and_repins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetcher(_fixture_tarball(ROOT_NAME_A, {}), FIXED_SHA_A, FIXED_SHA_B)
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    fetch_corpus(CaliperConfig(), fetcher=fake)

    refreshed = fetch_corpus(CaliperConfig(), refresh=True, fetcher=fake)

    assert len(fake.calls) == 4  # resolve+download again
    assert fake.calls[-1] == CODELOAD_URL.format(sha=FIXED_SHA_B)
    cache_dir = tmp_path / "cache" / "azure-docs"
    assert (cache_dir / f"{FIXED_SHA_A}.tar.gz").is_file()  # sha-keyed refs coexist
    assert (cache_dir / f"{FIXED_SHA_B}.tar.gz").is_file()
    assert refreshed == cache_dir / f"{FIXED_SHA_B}.tar.gz"
    pointer = json.loads((cache_dir / "main.json").read_text(encoding="utf-8"))
    assert pointer["sha"] == FIXED_SHA_B
    assert pointer["source_ref"] == f"main@{FIXED_SHA_B[:7]}"

    # the refreshed pin is now served from cache with zero network calls
    assert fetch_corpus(CaliperConfig(), fetcher=fake) == refreshed
    assert len(fake.calls) == 4


def test_fetch_corpus_reuses_cached_sha_tarball_without_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetcher(_fixture_tarball(ROOT_NAME_A, {}), FIXED_SHA_A)
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    first = fetch_corpus(CaliperConfig(), fetcher=fake)
    (tmp_path / "cache" / "azure-docs" / "main.json").unlink()  # pin lost, tarball kept

    second = fetch_corpus(CaliperConfig(), fetcher=fake)

    assert second == first
    # only the ref resolution ran again; the sha-keyed tarball was not redownloaded
    assert fake.calls == [COMMITS_URL, CODELOAD_URL.format(sha=FIXED_SHA_A), COMMITS_URL]


def test_fetch_corpus_defaults_to_data_cache_relative_to_cwd() -> None:
    fake = FakeFetcher(_fixture_tarball(ROOT_NAME_A, {}), FIXED_SHA_A)

    path = fetch_corpus(CaliperConfig(), fetcher=fake)

    assert path == Path("data/cache/azure-docs") / f"{FIXED_SHA_A}.tar.gz"
    assert path.is_file()  # resolved under tmp_path (cwd pinned by the fixture)


def test_fetch_corpus_wraps_fetcher_failures_as_source_error() -> None:
    def offline(url: str) -> bytes:
        raise RuntimeError("connection refused")

    with pytest.raises(SourceError, match="RuntimeError: connection refused"):
        fetch_corpus(CaliperConfig(), fetcher=offline)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (b"<html>rate limited</html>", "invalid JSON"),
        (json.dumps({"sha": "9f3ab12"}).encode("utf-8"), "40-hex"),
        (json.dumps({"commit": {"sha": FIXED_SHA_A}}).encode("utf-8"), "40-hex"),
    ],
)
def test_fetch_corpus_rejects_bad_ref_payload(body: bytes, match: str) -> None:
    def bad_api(url: str) -> bytes:
        assert url == COMMITS_URL
        return body

    with pytest.raises(SourceError, match=match):
        fetch_corpus(CaliperConfig(), fetcher=bad_api)


def test_fetch_corpus_raises_on_corrupt_pin_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetcher(_fixture_tarball(ROOT_NAME_A, {}), FIXED_SHA_A)
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    fetch_corpus(CaliperConfig(), fetcher=fake)
    pin = tmp_path / "cache" / "azure-docs" / "main.json"
    pin.write_text("{not json", encoding="utf-8")

    with pytest.raises(SourceError, match="unreadable"):
        fetch_corpus(CaliperConfig(), fetcher=fake)
    assert len(fake.calls) == 2  # corrupt pin is loud, never silently refetched


# --- read_cache_meta (T07 touchpoint) ------------------------------------------------


def test_read_cache_meta_returns_pinned_ref_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetcher(_fixture_tarball(ROOT_NAME_A, {}), FIXED_SHA_A)
    monkeypatch.setenv(CACHE_DIR_ENV_VAR, str(tmp_path / "cache"))
    fetch_corpus(CaliperConfig(), fetcher=fake)

    meta = read_cache_meta(CaliperConfig())

    assert isinstance(meta, CacheMeta)
    assert meta.source_ref == f"main@{FIXED_SHA_A[:7]}"
    assert meta.sha == FIXED_SHA_A
    assert meta.branch == "main"
    assert meta.fetched_at
    assert len(fake.calls) == 2  # no additional network for the metadata read


def test_read_cache_meta_raises_when_nothing_is_cached() -> None:
    with pytest.raises(SourceError, match="no corpus cached"):
        read_cache_meta(CaliperConfig())


def test_format_source_ref_uses_seven_char_sha() -> None:
    assert format_source_ref(FIXED_SHA_A) == f"main@{FIXED_SHA_A[:7]}"
    with pytest.raises(SourceError, match="40-hex"):
        format_source_ref("nothex")


# --- page_to_url (A1 mapping, 5 known pages) ------------------------------------------

# Mapping tests per the task spec: real article slugs (the first two appear in
# ARCHITECTURE §8 examples) pinning the path→URL rule — NOT existence tests.
KNOWN_PAGES = [
    (
        "articles/azure-functions/functions-timer.md",
        f"{LEARN}/azure-functions/functions-timer",
    ),
    (
        "articles/azure-functions/functions-bindings-timer.md",
        f"{LEARN}/azure-functions/functions-bindings-timer",
    ),
    (
        "articles/azure-functions/functions-create-first-function-vscode.md",
        f"{LEARN}/azure-functions/functions-create-first-function-vscode",
    ),
    (
        "articles/service-bus-messaging/service-bus-messaging-overview.md",
        f"{LEARN}/service-bus-messaging/service-bus-messaging-overview",
    ),
    (
        "articles/service-bus-messaging/service-bus-queues-topics-subscriptions.md",
        f"{LEARN}/service-bus-messaging/service-bus-queues-topics-subscriptions",
    ),
]


@pytest.mark.parametrize(("rel_path", "expected_url"), KNOWN_PAGES)
def test_page_to_url_maps_known_pages(rel_path: str, expected_url: str) -> None:
    assert page_to_url(rel_path) == expected_url


@pytest.mark.parametrize(
    "rel_path",
    [
        "articles/app-service/functions-timer.md",  # outside the two trees
        "articles/azure-functions/functions-timer.txt",  # wrong extension
        "articles/azure-functions/functions-timer.markdown",  # wrong extension
        "articles/azure-functions/functions-timer",  # no extension
        "articles/azure-functions/.md",  # empty slug
        "articles/azure-functions",  # the tree root itself
        "README.md",
        "",
    ],
)
def test_page_to_url_rejects_out_of_tree_or_non_md_paths(rel_path: str) -> None:
    with pytest.raises(SourceError, match="outside the article trees"):
        page_to_url(rel_path)


def test_page_to_url_normalizes_windows_separators() -> None:
    expected = KNOWN_PAGES[0][1]
    assert page_to_url("articles\\azure-functions\\functions-timer.md") == expected


# --- discover_pages (both trees + exclusions) -----------------------------------------


def test_discover_pages_lists_both_trees_sorted(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "articles/service-bus-messaging/service-bus-messaging-overview.md": "kept",
            "articles/azure-functions/functions-timer.md": "kept",
            "articles/azure-functions/prebuilt/functions-container-image.md": "kept (nested)",
            "articles/app-service/not-in-corpus.md": "other tree, out of scope",
            "articles/azure-functions/notes.txt": "not markdown",
        },
    )

    pages = discover_pages(tmp_path)

    assert pages == [
        "articles/azure-functions/functions-timer.md",
        "articles/azure-functions/prebuilt/functions-container-image.md",
        "articles/service-bus-messaging/service-bus-messaging-overview.md",
    ]


def test_discover_pages_applies_exclusion_rules(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        {
            "articles/azure-functions/functions-timer.md": "kept",
            "articles/azure-functions/_footer.md": "excluded: underscore name",
            "articles/azure-functions/includes/required-permissions.md": "excluded: segment",
            "articles/azure-functions/includes-guide.md": "kept: not an includes segment",
            "articles/service-bus-messaging/service-bus-queues-topics-subscriptions.md": "kept",
            "articles/service-bus-messaging/_toc.md": "excluded: underscore name",
            "articles/service-bus-messaging/service-bus-topics/includes/endpoint.md": "nested",
        },
    )

    pages = discover_pages(tmp_path)

    assert pages == [
        "articles/azure-functions/functions-timer.md",
        "articles/azure-functions/includes-guide.md",
        "articles/service-bus-messaging/service-bus-queues-topics-subscriptions.md",
    ]


def test_discover_pages_raises_on_missing_root(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not a directory"):
        discover_pages(tmp_path / "missing")


def test_discover_pages_raises_when_a_tree_is_missing(tmp_path: Path) -> None:
    _write_tree(
        tmp_path, {"articles/service-bus-messaging/service-bus-messaging-overview.md": "kept"}
    )

    with pytest.raises(SourceError, match="assumption A1"):
        discover_pages(tmp_path)


# --- TOC-API fallback seam -------------------------------------------------------------


def test_toc_api_fallback_seam_is_deliberately_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="A1"):
        _fetch_via_toc_api("https://learn.microsoft.com/api/catalog/")
