"""Corpus source: pinned-ref tarball fetch + cache, A1 URL mapping, page discovery (§7.4).

Downloads the `MicrosoftDocs/azure-docs` tarball exactly once at a pinned ref:
the branch head SHA is resolved via the GitHub API at first fetch and recorded
in the cache as `main@<7-char sha>` (§9.4 `source_ref`). The tarball is cached
under the cache dir keyed by SHA; every later call re-serves the cache with no
network call unless `refresh=True` re-resolves and refetches. This component
performs no database writes (`requests`/`runs` are never touched) and never
modifies or redistributes corpus files — the cache holds local copies for
private indexing only (NFR-06).

URL mapping per A1: `articles/azure-functions/<slug>.md` maps to
`https://learn.microsoft.com/en-us/azure/azure-functions/<slug>`, analogously
for `articles/service-bus-messaging/<slug>.md`. While A1 holds,
learn.microsoft.com is never scraped; the only fallback — the Learn TOC-API
crawler — stays behind the single unimplemented seam `_fetch_via_toc_api`
(§7.4).

The HTTP layer is one injectable seam (§15): `fetcher: Callable[[str], bytes]`,
defaulting to the httpx implementation. Tests inject a fake serving local
fixture bytes — no network, no `patch()` on internals.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from caliper.config import CaliperConfig

_LOGGER = logging.getLogger(__name__)

# --- Constants (§7.4, §11.2) ---------------------------------------------------

#: §11.2: paths live in the environment, never in the fingerprinted config
#: model. `CaliperConfig` has no cache-dir field (recorded in docs/TASKS.md
#: Notes), so the cache dir is read from this env var with the §11.2 default.
CACHE_DIR_ENV_VAR = "CALIPER_CACHE_DIR"
DEFAULT_CACHE_DIR = "data/cache"

#: The corpus repository and the branch whose head pins the frozen corpus.
AZURE_DOCS_REPO = "MicrosoftDocs/azure-docs"
AZURE_DOCS_BRANCH = "main"

#: §7.4/OQ3: the two article trees read from the corpus (every other
#: `articles/*` tree is out of scope).
ARTICLE_TREES: tuple[str, ...] = (
    "articles/azure-functions",
    "articles/service-bus-messaging",
)

#: A1: Learn's public base URL for the `articles/<tree>` content.
LEARN_BASE_URL = "https://learn.microsoft.com/en-us/azure"

#: Generous but bounded: the tarball is tens of MB (§16, no unbounded waits).
HTTP_TIMEOUT_S = 300.0

#: Headers acceptable to both the GitHub API (Accept) and codeload (ignored).
_HTTP_HEADERS = {
    "user-agent": "caliper/0.1.0 (local documentation corpus fetcher)",
    "accept": "application/vnd.github+json, application/octet-stream",
}

_CACHE_SUBDIR = "azure-docs"

_SHA_RE = re.compile(r"[0-9a-f]{40}")


# --- Typed component exception (§11.3) ------------------------------------------


class SourceError(Exception):
    """A corpus-source failure (§11.3): ref resolution, tarball fetch, cache
    metadata, or an out-of-contract corpus path. Mapped to the §9 error
    envelope at the CLI/web edges (T07/T12); no bare exceptions escape."""


# --- Cache metadata (read by T07 for the §9.4 summary) ---------------------------


@dataclass(frozen=True)
class CacheMeta:
    """The cached pin: which commit is frozen and when it was fetched.

    `source_ref` is the §9.4 ingestion-summary value; T07 reads it via
    `read_cache_meta` — no network, no re-resolution of the ref.
    """

    branch: str
    sha: str
    source_ref: str
    fetched_at: str


# --- HTTP seam (§15) --------------------------------------------------------------


#: The single HTTP touchpoint of this component: fetch `url`, return the raw
#: response body. The default implementation is `_http_fetch`; tests inject a
#: fake through the public `fetcher` parameter — no network, no patch().
Fetcher = Callable[[str], bytes]


def _http_fetch(url: str) -> bytes:
    """The default Fetcher: one httpx GET, strict status handling, body out."""
    response = httpx.get(url, timeout=HTTP_TIMEOUT_S, headers=_HTTP_HEADERS, follow_redirects=True)
    response.raise_for_status()
    return response.content


def _fetch_bytes(fetcher: Fetcher, url: str, describe: str) -> bytes:
    """Run one fetcher call, wrapping every failure as SourceError (§11.3).

    The wrap is the component boundary: callers (T07/CLI) only ever see the
    typed error. Raw failures stay at DEBUG, never user-facing.
    """
    try:
        return fetcher(url)
    except SourceError:
        raise
    except Exception as exc:  # noqa: BLE001 - single typed wrap, never swallowed
        _LOGGER.debug("%s failed", describe, exc_info=True)
        raise SourceError(f"{describe} failed for {url}: {type(exc).__name__}: {exc}") from exc


# --- Wire URLs (public: T07 logs them alongside the summary) ----------------------


def commits_api_url(branch: str = AZURE_DOCS_BRANCH) -> str:
    """GitHub API URL for the branch head commit (the §7.4 pin resolution)."""
    return f"https://api.github.com/repos/{AZURE_DOCS_REPO}/commits/{branch}"


def tarball_url(sha: str) -> str:
    """Codeload tarball URL for an exact commit SHA (the §7.4 pin download)."""
    return f"https://codeload.github.com/{AZURE_DOCS_REPO}/tar.gz/{sha}"


def format_source_ref(sha: str) -> str:
    """§9.4 `source_ref` format: `main@<7-char sha>` (e.g. `main@9f3ab12`)."""
    if _SHA_RE.fullmatch(sha) is None:
        raise SourceError(f"expected a 40-hex commit SHA, got {sha!r}")
    return f"{AZURE_DOCS_BRANCH}@{sha[:7]}"


# --- Cache layout ------------------------------------------------------------------


def _cache_dir(cfg: CaliperConfig) -> Path:
    """The cache dir: CALIPER_CACHE_DIR (§11.2), default `data/cache`.

    `cfg` is part of the frozen §7.4 signature; the fingerprint model
    deliberately holds no paths (§11.2), and `CaliperConfig` has no cache-dir
    field (choice recorded in docs/TASKS.md Notes). If a field is ever added,
    prefer it over the env var here.
    """
    return Path(os.environ.get(CACHE_DIR_ENV_VAR, DEFAULT_CACHE_DIR))


def _component_dir(cache_root: Path) -> Path:
    """The per-source cache directory: `<cache>/azure-docs/`."""
    return cache_root / _CACHE_SUBDIR


def _pointer_path(cache_root: Path) -> Path:
    """The per-branch pin pointer: `<cache>/azure-docs/main.json`."""
    return _component_dir(cache_root) / f"{AZURE_DOCS_BRANCH}.json"


def _tarball_path(cache_root: Path, sha: str) -> Path:
    """The SHA-keyed tarball path (distinct pinned refs coexist)."""
    return _component_dir(cache_root) / f"{sha}.tar.gz"


def _now_iso() -> str:
    """ISO-8601 UTC timestamp (§8 column convention)."""
    return datetime.now(UTC).isoformat()


def _read_pointer(pointer: Path) -> CacheMeta:
    """Parse + validate the pin pointer; SourceError on any problem.

    A corrupt pin raises rather than masquerading as a cache miss: the pin is
    what keeps the corpus frozen, and silently refetching over it would move
    the pin without `refresh`. (Writes are atomic, so corruption means manual
    tampering — fix or delete the cache dir.)
    """
    try:
        raw: object = json.loads(pointer.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceError(f"cache pin file {pointer} is unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise SourceError(f"cache pin file {pointer} must be a JSON object")
    branch = raw.get("branch")
    sha = raw.get("sha")
    source_ref = raw.get("source_ref")
    fetched_at = raw.get("fetched_at")
    if not isinstance(branch, str) or branch != AZURE_DOCS_BRANCH:
        raise SourceError(
            f"cache pin file {pointer} is for branch {branch!r}, not {AZURE_DOCS_BRANCH!r}"
        )
    if not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        raise SourceError(f"cache pin file {pointer} has no valid 40-hex 'sha'")
    if not isinstance(source_ref, str) or source_ref != format_source_ref(sha):
        raise SourceError(f"cache pin file {pointer} has an inconsistent 'source_ref'")
    if not isinstance(fetched_at, str) or not fetched_at:
        raise SourceError(f"cache pin file {pointer} has no 'fetched_at' timestamp")
    return CacheMeta(branch=branch, sha=sha, source_ref=source_ref, fetched_at=fetched_at)


def _write_pointer(pointer: Path, meta: CacheMeta) -> None:
    """Write the pin pointer atomically (tmp + os.replace, crash-safe)."""
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_name(f"{pointer.name}.tmp")
    tmp.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, pointer)


def _cache_hit(cache_root: Path) -> Path | None:
    """The cached tarball path when pin + tarball are both present, else None.

    A valid pin whose tarball vanished is a repairable miss (refetch); a
    corrupt pin is a loud SourceError (see _read_pointer).
    """
    pointer = _pointer_path(cache_root)
    if not pointer.is_file():
        return None
    meta = _read_pointer(pointer)
    tarball = _tarball_path(cache_root, meta.sha)
    if not tarball.is_file():
        _LOGGER.warning("cache pin exists but tarball %s is missing; refetching", tarball)
        return None
    return tarball


# --- Ref resolution + download (both via the Fetcher seam) -------------------------


def _resolve_sha(fetcher: Fetcher) -> str:
    """Resolve the branch head SHA once via the GitHub API (§7.4 pin)."""
    url = commits_api_url()
    raw = _fetch_bytes(fetcher, url, f"resolving {AZURE_DOCS_REPO} {AZURE_DOCS_BRANCH} head")
    try:
        payload: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceError(f"{url} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise SourceError(f"{url} response must be a JSON object")
    sha = payload.get("sha")
    if not isinstance(sha, str) or _SHA_RE.fullmatch(sha) is None:
        raise SourceError(f"{url} response has no 40-hex 'sha' commit")
    return sha


def _download_tarball(fetcher: Fetcher, sha: str) -> bytes:
    """Download the codeload tarball for the pinned SHA."""
    describe = f"fetching {AZURE_DOCS_REPO} tarball at {sha[:7]}"
    return _fetch_bytes(fetcher, tarball_url(sha), describe)


# --- Public interface (frozen for T06/T07) ------------------------------------------


def fetch_corpus(
    cfg: CaliperConfig,
    *,
    refresh: bool = False,
    fetcher: Fetcher | None = None,
) -> Path:
    """Fetch the frozen corpus tarball once and cache it; return its path (§7.4).

    First call resolves the branch head SHA (GitHub API), downloads the
    codeload tarball for that exact SHA, and caches it keyed by SHA plus a
    pin pointer recording `source_ref` (§9.4) and the fetch timestamp. Any
    later call with a complete cache returns the cached path with **no
    network call at all** — the pin stays whatever was resolved at first
    fetch. `refresh=True` re-resolves the ref and refetches unconditionally;
    SHA-keyed storage keeps distinct refs coexisting. A tarball already on
    disk for the resolved SHA is reused without redownloading (the bytes are
    fetched exactly once per SHA).

    Both network steps go through the injected `fetcher` seam (§15). `cfg`
    shapes the frozen §7.4 interface for T07's `run_ingest(cfg)`; the cache
    directory comes from CALIPER_CACHE_DIR (§11.2 — see `_cache_dir`).

    Raises SourceError on any fetch, resolution, or cache-metadata failure.
    """
    fetch: Fetcher = fetcher if fetcher is not None else _http_fetch
    cache_root = _cache_dir(cfg)
    if not refresh:
        cached = _cache_hit(cache_root)
        if cached is not None:
            _LOGGER.info("corpus cache hit: %s", cached)
            return cached

    sha = _resolve_sha(fetch)
    tarball = _tarball_path(cache_root, sha)
    if refresh or not tarball.is_file():
        blob = _download_tarball(fetch, sha)
        tarball.parent.mkdir(parents=True, exist_ok=True)
        tarball.write_bytes(blob)
    meta = CacheMeta(
        branch=AZURE_DOCS_BRANCH,
        sha=sha,
        source_ref=format_source_ref(sha),
        fetched_at=_now_iso(),
    )
    _write_pointer(_pointer_path(cache_root), meta)
    _LOGGER.info("corpus cached: %s (source_ref %s)", tarball, meta.source_ref)
    return tarball


def read_cache_meta(cfg: CaliperConfig) -> CacheMeta:
    """The cached pin (§9.4 `source_ref`) without any network call.

    T07 touchpoint: supplies `source_ref` for the ingestion summary from the
    cache instead of re-resolving the ref. Raises SourceError when nothing is
    cached yet or the pin file is corrupt.
    """
    pointer = _pointer_path(_cache_dir(cfg))
    if not pointer.is_file():
        raise SourceError(
            f"no corpus cached yet (missing pin file {pointer}); run fetch_corpus first"
        )
    return _read_pointer(pointer)


def page_to_url(rel_path: str) -> str:
    """Map a corpus-relative page path to its public Learn URL (A1).

    `articles/azure-functions/<slug>.md` maps to
    `https://learn.microsoft.com/en-us/azure/azure-functions/<slug>` and
    analogously for `articles/service-bus-messaging/<slug>.md`; nested slugs
    keep their subpath (Learn mirrors the docs tree). Backslashes are
    normalized so Windows-built paths map identically. Raises SourceError for
    paths outside the two §7.4/OQ3 trees, wrong extensions, or an empty slug;
    `_`/`includes` filtering is `discover_pages`' job, not this mapping's.
    """
    normalized = rel_path.replace("\\", "/")
    for tree in ARTICLE_TREES:
        prefix = f"{tree}/"
        if normalized.startswith(prefix) and normalized.endswith(".md"):
            slug = normalized.removeprefix(prefix).removesuffix(".md")
            if slug:
                return f"{LEARN_BASE_URL}/{tree.removeprefix('articles/')}/{slug}"
    raise SourceError(
        f"corpus path {rel_path!r} is outside the article trees "
        f"({', '.join(ARTICLE_TREES)}) or not a .md page"
    )


def discover_pages(extracted_root: str | Path) -> list[str]:
    """Corpus-relative `.md` paths of the two article trees (§7.4, OQ3).

    `extracted_root` is the root of the *extracted* tarball — the directory
    containing `articles/` (T07 strips the tarball's top-level
    `azure-docs-<sha>/` directory before calling this). Returns sorted
    forward-slash paths relative to `extracted_root` — feed them straight to
    `page_to_url`. Both exclusion rules apply: any path segment named
    `includes`, and any filename starting with `_`.

    Raises SourceError when the root or either tree is missing — that is the
    A1-failure signal that would trigger the TOC-API fallback seam.
    """
    root = Path(extracted_root)
    if not root.is_dir():
        raise SourceError(f"extracted corpus root is not a directory: {root}")
    pages: list[str] = []
    for tree in ARTICLE_TREES:
        tree_root = root / tree
        if not tree_root.is_dir():
            raise SourceError(
                f"corpus article tree {tree!r} is missing under {root} "
                "(assumption A1 broken; TOC-API fallback would apply)"
            )
        for path in tree_root.rglob("*.md"):
            rel = path.relative_to(root)
            if "includes" in rel.parts or rel.name.startswith("_"):
                continue
            pages.append(rel.as_posix())
    return sorted(pages)


def _fetch_via_toc_api(toc_url: str) -> bytes:
    """A1 fallback seam: the Learn TOC-API crawler (§7.4), unimplemented.

    Used ONLY if assumption A1 breaks — the upstream repository no longer
    keeps Azure Functions articles at `articles/azure-functions/*.md` and
    Service Bus articles at `articles/service-bus-messaging/*.md`, or the
    repository disappears. In that case, `discover_pages` raises the A1
    error and fetching must switch to the documented Learn TOC API plus a
    polite per-page fetcher (§7.4: "fallback TOC-API crawler only if A1
    fails, behind source.py's single seam"). While A1 holds,
    learn.microsoft.com is never scraped (NFR-06, §7.4).
    """
    raise NotImplementedError(
        "TOC-API fallback is unimplemented: assumption A1 holds, so the GitHub "
        "tarball is the only active corpus source"
    )
