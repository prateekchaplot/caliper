"""Chunkers: the V1 heading chunker and the V2/V3 fixed-window chunker (§7.4, FR-02).

`make_chunker(cfg)` is ingest's §7.4 factory: it reads `cfg.chunking.unit` /
`size_tokens` / `overlap_tokens` (§9.7 presets) and returns the matching
`Chunker`. Chunkers consume `parse.ParsedPage` and emit `Chunk`s carrying
exactly the `db.ChunkRow` fields minus `chunk_hash` (hashing is T07's job).

Binding decisions (recorded here for §7.4 and the T07 freeze):

- **Token estimator:** `tokens = ceil(len(text) / 4)` (`count_tokens`, the
  4-chars-per-token rule of thumb). Zero dependencies, fully deterministic;
  every bound below is expressed and enforced in these tokens.
- **Hard cap:** every chunk is ≤ `MAX_CHUNK_TOKENS = 1900` tokens (embedding
  input limit is 2,048, §7.4). A chunk that would exceed the cap is **split
  further, never truncated**: an oversized single section becomes sequential
  whitespace-aligned windows of ≤ `size_tokens` tokens (no overlap — the
  heading unit defines none). The `fixed` unit is structurally bounded: its
  body budget is `size_tokens` minus the prefix, so every chunk is ≤
  `size_tokens` ≤ 1900 by construction.
- **`heading` (V1):** content is grouped at H2 boundaries (H3+ subsections
  stay attached to their H2; the intro under the title is its own group).
  Adjacent groups merge greedily while the merged body stays within
  `size_tokens` (`max_tokens`). A single group larger than `size_tokens` is
  emitted **alone and whole** up to the 1,900 cap (the §14.2 "single section
  exceeds it" exception); only beyond the cap is it split as above.
  `overlap_tokens` is ignored by this unit (§7.4 defines no overlap for it;
  presets set 0) — a warning is logged if a config sets it > 0. A merged
  chunk's heading path is the path of its first section (where it starts).
- **`fixed` (V2/V3):** the page body is windowed at `size_tokens` tokens per
  chunk **counting the `Page: <url>\\nHeading: <path>\\n\\n` prefix** (the
  window body budget is `size_tokens` minus the prefix's tokens), advancing
  by window length minus `overlap_tokens` between windows. A chunk's heading
  path is the path as of the window's first character. Windows cut at the
  last line/whitespace break inside the budget (hard cuts only as a last
  resort), so windows are ~`size_tokens` tokens but never more.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from caliper.config import CaliperConfig
from caliper.ingest.parse import ParsedPage

_LOGGER = logging.getLogger(__name__)

#: Chars per estimated token (binding decision: tokens = ceil(len / 4)).
CHARS_PER_TOKEN = 4

#: §7.4 hard cap: every chunk stays within this many estimated tokens
#: (the embedding input limit is 2,048).
MAX_CHUNK_TOKENS = 1900


def count_tokens(text: str) -> int:
    """Estimated token count: `ceil(len(text) / CHARS_PER_TOKEN)` (§7.4 note)."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _chunk_prefix(page_url: str, heading_path: str) -> str:
    """The fixed unit's per-chunk prefix (§7.4: `Page:` + `Heading:` lines)."""
    return f"Page: {page_url}\nHeading: {heading_path}\n\n"


def _find_cut(text: str, start: int, max_chars: int) -> int:
    """Exclusive end of a window `text[start : cut]` at most `max_chars` long.

    Cuts at the last whitespace inside the budget (paragraph break, line
    break, or space, whichever lands latest); a hard cut inside a token is
    the last resort.
    """
    limit = start + max_chars
    if limit >= len(text):
        return len(text)
    best = -1
    for separator in ("\n\n", "\n", " "):
        cut = text.rfind(separator, start, limit)
        if cut > best:
            best = cut
    if best > start:
        return best
    return limit


def _split_to_token_windows(text: str, max_tokens: int) -> list[str]:
    """Split `text` into sequential windows of ≤ `max_tokens` tokens each.

    Used only by the cap splitter for oversized single sections; whitespace
    between windows is dropped (never content).
    """
    windows: list[str] = []
    start = 0
    while start < len(text):
        cut = _find_cut(text, start, max_tokens * CHARS_PER_TOKEN)
        window = text[start:cut].rstrip()
        if window:
            windows.append(window)
        start = cut
        while start < len(text) and text[start] in " \n":
            start += 1
    return windows


@dataclass(frozen=True, slots=True)
class Chunk:
    """One index chunk: the §8.3 row payload minus `chunk_hash` (T07's job).

    Field order mirrors `db.ChunkRow` so T07 builds rows positionally:
    `db.ChunkRow(page_url, heading_path, body, body_tokens, chunk_hash, ordinal)`.
    """

    page_url: str
    heading_path: str
    body: str
    body_tokens: int
    ordinal: int


@runtime_checkable
class Chunker(Protocol):
    """§7.4 chunker interface: one parsed page → bounded chunks."""

    def chunk_page(self, page: ParsedPage) -> list[Chunk]: ...


def _h2_groups(page: ParsedPage) -> list[tuple[str, str]]:
    """`(heading_path, group body)` pairs: a new group per level ≤ 2 segment.

    H3+ subsections attach to the current H2 group; the intro segment under
    the title (level 1) is its own group and may merge with the next one.
    """
    groups: list[tuple[str, list[str]]] = []
    for segment in page.segments:
        if not segment.body:
            continue
        if segment.level <= 2 or not groups:
            groups.append((segment.heading_path, [segment.body]))
        else:
            groups[-1][1].append(segment.body)
    return [(path, "\n\n".join(parts)) for path, parts in groups]


class HeadingChunker:
    """V1 unit: H2-boundary groups, undersized neighbors merged up to max."""

    def __init__(self, max_tokens: int) -> None:
        self._max_tokens = max_tokens

    def chunk_page(self, page: ParsedPage) -> list[Chunk]:
        """Group at H2 boundaries, merge undersized neighbors, cap oversized ones."""
        chunks: list[Chunk] = []
        merged: list[str] = []
        merged_path: str | None = None

        def flush() -> None:
            nonlocal merged, merged_path
            if merged and merged_path is not None:
                body = "\n\n".join(merged)
                chunks.append(
                    Chunk(
                        page_url=page.page_url,
                        heading_path=merged_path,
                        body=body,
                        body_tokens=count_tokens(body),
                        ordinal=len(chunks),
                    )
                )
            merged = []
            merged_path = None

        for path, body in _h2_groups(page):
            tokens = count_tokens(body)
            if tokens > MAX_CHUNK_TOKENS:
                # Hard cap: split further, never truncate (§7.4). The window
                # size stays the unit's own bound (min with the cap).
                flush()
                budget = min(self._max_tokens, MAX_CHUNK_TOKENS)
                for window in _split_to_token_windows(body, budget):
                    chunks.append(
                        Chunk(
                            page_url=page.page_url,
                            heading_path=path,
                            body=window,
                            body_tokens=count_tokens(window),
                            ordinal=len(chunks),
                        )
                    )
                continue
            candidate = [*merged, body]
            if merged and count_tokens("\n\n".join(candidate)) > self._max_tokens:
                flush()
            if merged_path is None:
                merged_path = path
            merged.append(body)
        flush()
        return chunks


def _page_layout(page: ParsedPage) -> tuple[list[int], list[str], str]:
    """The page's concatenated body plus each segment's (offset, heading path)."""
    bodies = [segment.body for segment in page.segments if segment.body]
    full = "\n\n".join(bodies)
    offsets: list[int] = []
    paths: list[str] = []
    position = 0
    for segment in page.segments:
        if not segment.body:
            continue
        offsets.append(position)
        paths.append(segment.heading_path)
        position += len(segment.body) + 2  # the "\n\n" separator added above
    return offsets, paths, full


class FixedChunker:
    """V2/V3 unit: sliding windows with overlap, prefix counted in bounds."""

    def __init__(self, size_tokens: int, overlap_tokens: int) -> None:
        self._size_tokens = size_tokens
        self._overlap_tokens = overlap_tokens

    def chunk_page(self, page: ParsedPage) -> list[Chunk]:
        """Window the whole page body; heading path as of each window's start."""
        offsets, paths, full = _page_layout(page)
        if not full:
            return []
        chunks: list[Chunk] = []
        overlap_chars = self._overlap_tokens * CHARS_PER_TOKEN
        start = 0
        while start < len(full):
            index = bisect_right(offsets, start) - 1  # offsets[0] == 0 ≤ start
            path = paths[index]
            prefix = _chunk_prefix(page.page_url, path)
            budget_tokens = max(1, self._size_tokens - count_tokens(prefix))
            cut = _find_cut(full, start, budget_tokens * CHARS_PER_TOKEN)
            body = prefix + full[start:cut]
            chunks.append(
                Chunk(
                    page_url=page.page_url,
                    heading_path=path,
                    body=body,
                    body_tokens=count_tokens(body),
                    ordinal=len(chunks),
                )
            )
            if cut >= len(full):
                break
            if cut - start > overlap_chars:
                start = cut - overlap_chars
            else:
                # Degenerate config (overlap ≈ size): still guarantee progress.
                start = start + max(1, (cut - start) // 2)
        return chunks


def make_chunker(cfg: CaliperConfig) -> Chunker:
    """§7.4 factory: pick the chunker from `cfg.chunking` (§9.7 presets).

    `heading` (v1) → `HeadingChunker`; `fixed` (v2/v3) → `FixedChunker`.
    Raises `ValueError` on an unknown unit (unreachable via the config model,
    whose `unit` field is a Literal).
    """
    chunking = cfg.chunking
    if chunking.unit == "heading":
        if chunking.overlap_tokens > 0:
            _LOGGER.warning(
                "chunking.unit 'heading' ignores overlap_tokens (%d): "
                "the §7.4 heading unit defines no overlap",
                chunking.overlap_tokens,
            )
        return HeadingChunker(max_tokens=chunking.size_tokens)
    if chunking.unit == "fixed":
        return FixedChunker(
            size_tokens=chunking.size_tokens, overlap_tokens=chunking.overlap_tokens
        )
    raise ValueError(f"unknown chunking.unit {chunking.unit!r}")
