"""Unit tests: ingest/chunkers.py — heading/fixed token bounds, overlap, 1,900 cap.

Segment bodies are crafted with exact character counts so token bounds
(ceil(len/4)) are asserted precisely (§7.4, FR-02, §14.2 T06). Everything is
inline fixtures: no network, no patch().
"""

from pathlib import Path

import pytest

from caliper.config import CaliperConfig, ChunkingConfig, load_config
from caliper.ingest.chunkers import (
    MAX_CHUNK_TOKENS,
    Chunker,
    FixedChunker,
    HeadingChunker,
    count_tokens,
    make_chunker,
)
from caliper.ingest.parse import BodySegment, ParsedPage

URL = "https://learn.microsoft.com/en-us/azure/azure-functions/functions-timer"

V1_PRESET = """\
[chunking]
unit = "heading"
size_tokens = 800
overlap_tokens = 0
"""

V2_PRESET = """\
[chunking]
unit = "fixed"
size_tokens = 350
overlap_tokens = 50
"""


def _page(*segments: tuple[str, int, str]) -> ParsedPage:
    return ParsedPage(
        rel_path="articles/azure-functions/functions-timer.md",
        page_url=URL,
        title="Timer triggers",
        segments=[BodySegment(path, level, body) for path, level, body in segments],
    )


def _long_body(lines: int, filler: str = "x") -> str:
    return "\n".join(f"{filler}-line-{i:04d} {filler * 30}" for i in range(lines))


# --- token estimator ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("a", 1),
        ("abcd", 1),
        ("abcde", 2),
        ("a" * 400, 100),
    ],
)
def test_count_tokens_is_ceiling_of_len_over_four(text: str, expected: int) -> None:
    assert count_tokens(text) == expected


# --- heading chunker (V1) ------------------------------------------------------------


def test_heading_chunker_merges_undersized_neighbors_up_to_max() -> None:
    chunker = HeadingChunker(max_tokens=55)
    page = _page(("T > A", 2, "a" * 100), ("T > B", 2, "b" * 100), ("T > C", 2, "c" * 100))

    chunks = chunker.chunk_page(page)

    # A + B merged (202 chars → 51 tokens ≤ 55); C would push past max.
    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert chunks[0].heading_path == "T > A"  # path of the chunk's first section
    assert chunks[0].body == "a" * 100 + "\n\n" + "b" * 100
    assert chunks[0].body_tokens == count_tokens(chunks[0].body) == 51
    assert chunks[1].heading_path == "T > C"
    assert chunks[1].body == "c" * 100
    assert chunks[1].body_tokens == 25


def test_heading_chunker_never_merges_beyond_max_tokens() -> None:
    chunker = HeadingChunker(max_tokens=30)
    page = _page(("T > A", 2, "a" * 100), ("T > B", 2, "b" * 100), ("T > C", 2, "c" * 100))

    chunks = chunker.chunk_page(page)

    assert len(chunks) == 3
    assert [chunk.heading_path for chunk in chunks] == ["T > A", "T > B", "T > C"]
    assert all(chunk.body_tokens == 25 for chunk in chunks)


def test_heading_chunker_groups_h3_sections_into_their_h2() -> None:
    chunker = HeadingChunker(max_tokens=800)
    page = _page(
        ("T", 1, "intro"),
        ("T > A", 2, "alpha"),
        ("T > A > Sub", 3, "sub"),
        ("T > B", 2, "beta"),
    )

    chunks = chunker.chunk_page(page)

    # Everything is undersized at 800: one merged chunk anchored at the intro.
    assert len(chunks) == 1
    assert chunks[0].heading_path == "T"
    assert all(needle in chunks[0].body for needle in ("intro", "alpha", "sub", "beta"))


def test_heading_chunker_splits_at_h2_boundaries_when_groups_are_large() -> None:
    chunker = HeadingChunker(max_tokens=800)
    page = _page(("T", 1, "intro"), ("T > A", 2, "a" * 2800), ("T > B", 2, "b" * 2800))

    chunks = chunker.chunk_page(page)

    assert [chunk.heading_path for chunk in chunks] == ["T", "T > B"]
    assert chunks[0].body_tokens == count_tokens("intro\n\n" + "a" * 2800) == 702
    assert chunks[1].body_tokens == 700


def test_heading_chunker_keeps_a_single_oversized_section_whole_under_the_cap() -> None:
    # 3,600 chars → 900 tokens: above size_tokens (30) but below the 1,900 cap.
    chunker = HeadingChunker(max_tokens=30)
    page = _page(("T > Big", 2, "d" * 3600))

    chunks = chunker.chunk_page(page)

    assert len(chunks) == 1
    assert chunks[0].heading_path == "T > Big"
    assert chunks[0].body == "d" * 3600
    assert chunks[0].body_tokens == 900
    assert chunks[0].body_tokens <= MAX_CHUNK_TOKENS


def test_heading_chunker_splits_sections_past_the_cap_never_truncates() -> None:
    chunker = HeadingChunker(max_tokens=30)  # windows of ≤ 30 tokens = 120 chars
    page = _page(("T > Huge", 2, "e" * 20000))

    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1
    assert all(chunk.body_tokens <= 30 for chunk in chunks)
    assert all(chunk.body_tokens <= MAX_CHUNK_TOKENS for chunk in chunks)
    assert "".join(chunk.body for chunk in chunks) == "e" * 20000  # nothing truncated


def test_heading_chunker_cap_enforced_with_max_config_size() -> None:
    chunker = HeadingChunker(max_tokens=MAX_CHUNK_TOKENS)
    page = _page(("T > Huge", 2, "e" * 20000))  # 5,000 tokens > 1,900

    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1
    assert all(chunk.body_tokens <= MAX_CHUNK_TOKENS for chunk in chunks)
    assert "".join(chunk.body for chunk in chunks) == "e" * 20000


def test_heading_chunker_returns_no_chunks_for_empty_pages() -> None:
    chunker = HeadingChunker(max_tokens=800)

    assert chunker.chunk_page(_page()) == []
    assert chunker.chunk_page(_page(("T", 1, ""), ("T > A", 2, ""))) == []


# --- fixed chunker (V2/V3) -------------------------------------------------------------


def _fixed() -> FixedChunker:
    chunker = make_chunker(
        CaliperConfig(chunking=ChunkingConfig(unit="fixed", size_tokens=350, overlap_tokens=50))
    )
    assert isinstance(chunker, FixedChunker)
    return chunker


def test_fixed_chunker_windows_are_bounded_with_prefix_counted() -> None:
    page = _page(("T > A", 2, _long_body(280)))  # ~12k chars ≈ 3k tokens

    chunks = _fixed().chunk_page(page)

    assert len(chunks) >= 5
    expected_prefix = f"Page: {URL}\nHeading: T > A\n\n"
    for chunk in chunks:
        assert chunk.body_tokens == count_tokens(chunk.body)
        assert chunk.body.startswith(expected_prefix)
        assert chunk.body_tokens <= 350  # prefix counted in the token count
        assert chunk.body_tokens <= MAX_CHUNK_TOKENS
    # windows are ~size_tokens, not fragments: every non-final chunk is ≥ 250 tokens
    assert all(chunk.body_tokens >= 250 for chunk in chunks[:-1])


def test_fixed_chunker_overlap_shares_the_window_tail() -> None:
    page = _page(("T > A", 2, _long_body(280)))
    chunks = _fixed().chunk_page(page)

    assert len(chunks) >= 2
    for first, second in zip(chunks, chunks[1:], strict=False):
        # overlap of 50 tokens = 200 chars: the next window starts 200 chars back
        assert first.body[-60:] in second.body


def test_fixed_chunker_heading_path_follows_the_window_position() -> None:
    page = _page(("T", 1, "a" * 400), ("T > Section Two", 2, _long_body(280)))

    chunks = _fixed().chunk_page(page)

    assert chunks[0].heading_path == "T"
    assert chunks[1].heading_path == "T > Section Two"
    assert all(chunk.heading_path == "T > Section Two" for chunk in chunks[1:])


def test_fixed_chunker_returns_no_chunks_for_empty_pages() -> None:
    assert _fixed().chunk_page(_page()) == []
    assert _fixed().chunk_page(_page(("T", 1, ""), ("T > A", 2, ""))) == []


def test_fixed_chunker_respects_the_cap_at_max_config_size() -> None:
    chunker = make_chunker(
        CaliperConfig(
            chunking=ChunkingConfig(unit="fixed", size_tokens=MAX_CHUNK_TOKENS, overlap_tokens=50)
        )
    )
    page = _page(("T > A", 2, _long_body(700)))

    chunks = chunker.chunk_page(page)

    assert len(chunks) >= 2
    assert all(chunk.body_tokens <= MAX_CHUNK_TOKENS for chunk in chunks)


# --- make_chunker factory (§9.7 presets) -------------------------------------------------


def _preset_config(tmp_path: Path, preset_toml: str) -> CaliperConfig:
    """A resolved config from an inline preset file (self-contained fixture)."""
    preset = tmp_path / "preset.toml"
    preset.write_text(preset_toml, encoding="utf-8")
    user = tmp_path / "user.toml"
    user.write_text("", encoding="utf-8")
    return load_config(user, preset)


def test_make_chunker_builds_the_heading_chunker_for_v1_preset_values(
    tmp_path: Path,
) -> None:
    cfg = _preset_config(tmp_path, V1_PRESET)

    chunker = make_chunker(cfg)

    assert isinstance(chunker, Chunker)
    assert isinstance(chunker, HeadingChunker)
    # 800-token max: two 700-token H2 sections must not merge.
    page = _page(("T > A", 2, "a" * 2800), ("T > B", 2, "b" * 2800))
    chunks = chunker.chunk_page(page)
    assert len(chunks) == 2
    assert all(chunk.body_tokens <= 800 for chunk in chunks)


def test_make_chunker_builds_the_fixed_chunker_for_v2_preset_values(
    tmp_path: Path,
) -> None:
    cfg = _preset_config(tmp_path, V2_PRESET)

    chunker = make_chunker(cfg)

    assert isinstance(chunker, Chunker)
    assert isinstance(chunker, FixedChunker)
    # 350-token windows with the prefix counted: long page → many bounded chunks.
    page = _page(("T > A", 2, _long_body(280)))
    chunks = chunker.chunk_page(page)
    assert len(chunks) >= 5
    assert all(
        chunk.body_tokens <= 350 and chunk.body.startswith(f"Page: {URL}") for chunk in chunks
    )


def test_make_chunker_defaults_to_the_heading_chunker() -> None:
    chunker = make_chunker(CaliperConfig())

    assert isinstance(chunker, HeadingChunker)
