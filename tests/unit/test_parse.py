"""Unit tests: ingest/parse.py — title, heading paths, body segments (§7.4).

Inline markdown fixtures only (§15): no file I/O over the cache, no network,
no patch() on internals. Out-of-contract paths surface T05's SourceError.
"""

import pytest

from caliper.ingest.parse import BodySegment, ParsedPage, ParseError, parse_page
from caliper.ingest.source import SourceError

REL_AZ = "articles/azure-functions/functions-timer.md"
URL_AZ = "https://learn.microsoft.com/en-us/azure/azure-functions/functions-timer"

SIMPLE_PAGE = """\
---
title: Timer triggers
ms.topic: reference
---

# Timer triggers

Intro paragraph about timer triggers.

## Cron expressions

Cron text under the first H2.

## Retry behavior

Retry text under the second H2.
"""

NESTED_PAGE = """\
# Page

intro

## One

one body

### Nested

nested body

## Two

two body
"""

DEEP_PAGE = """\
# Deep

deep intro

## A

a body

### B

b body

#### C

deep body
"""


def test_parse_page_extracts_title_url_and_h2_paths() -> None:
    page = parse_page(REL_AZ, SIMPLE_PAGE)

    assert isinstance(page, ParsedPage)
    assert page.rel_path == REL_AZ
    assert page.page_url == URL_AZ
    assert page.title == "Timer triggers"
    assert [segment.heading_path for segment in page.segments] == [
        "Timer triggers",
        "Timer triggers > Cron expressions",
        "Timer triggers > Retry behavior",
    ]
    assert [segment.body for segment in page.segments] == [
        "Intro paragraph about timer triggers.",
        "Cron text under the first H2.",
        "Retry text under the second H2.",
    ]
    assert [segment.level for segment in page.segments] == [1, 2, 2]
    assert all("ms.topic" not in segment.body for segment in page.segments)


def test_parse_page_h3_nests_under_h2_and_resets_on_next_h2() -> None:
    page = parse_page(REL_AZ, NESTED_PAGE)

    assert page.title == "Page"
    assert [(segment.heading_path, segment.level) for segment in page.segments] == [
        ("Page", 1),
        ("Page > One", 2),
        ("Page > One > Nested", 3),
        ("Page > Two", 2),
    ]
    assert [segment.body for segment in page.segments] == [
        "intro",
        "one body",
        "nested body",
        "two body",
    ]


def test_parse_page_h4_continues_the_same_path_pattern() -> None:
    page = parse_page(REL_AZ, DEEP_PAGE)

    assert [segment.heading_path for segment in page.segments] == [
        "Deep",
        "Deep > A",
        "Deep > A > B",
        "Deep > A > B > C",
    ]
    assert page.segments[-1].level == 4
    assert page.segments[-1].body == "deep body"


@pytest.mark.parametrize(
    "text",
    [
        "## Only an H2\n\nbody text\n",
        "",
        "\n\n",
        "---\ntitle: x\nms.topic: y\n---\n",
        "Title\n=====\n\nsetext underlines are not headings\n",
    ],
)
def test_parse_page_without_atx_h1_raises_parse_error(text: str) -> None:
    with pytest.raises(ParseError) as excinfo:
        parse_page(REL_AZ, text)

    assert excinfo.value.reason == "no H1"
    assert "no H1" in str(excinfo.value)
    assert REL_AZ in str(excinfo.value)


def test_parse_page_title_only_page_has_no_segments() -> None:
    page = parse_page(REL_AZ, "# Timer triggers\n")

    assert page.title == "Timer triggers"
    assert page.segments == []


def test_parse_page_heading_inside_code_fence_is_body_text() -> None:
    page = parse_page(REL_AZ, "# T\n\n```python\n# not a heading\nx = 1\n```\n\nafter\n")

    assert len(page.segments) == 1
    assert "# not a heading" in page.segments[0].body
    assert page.segments[0].heading_path == "T"


def test_parse_page_repeated_h1_becomes_a_divider() -> None:
    page = parse_page(REL_AZ, "# Title\n\nintro\n\n# Second\n\nbody\n")

    assert page.title == "Title"
    assert [segment.heading_path for segment in page.segments] == [
        "Title",
        "Title > Second",
    ]


def test_parse_page_drops_content_before_the_first_h1() -> None:
    page = parse_page(REL_AZ, "orphan lead\n# Title\n\nbody\n")

    assert page.title == "Title"
    assert [segment.body for segment in page.segments] == ["body"]


def test_parse_page_maps_service_bus_urls() -> None:
    page = parse_page(
        "articles/service-bus-messaging/service-bus-messaging-overview.md",
        "# Service Bus messaging overview\n\nbody\n",
    )

    assert page.page_url == (
        "https://learn.microsoft.com/en-us/azure/"
        "service-bus-messaging/service-bus-messaging-overview"
    )


def test_parse_page_rejects_out_of_tree_paths() -> None:
    with pytest.raises(SourceError):
        parse_page("README.md", "# Title\n\nbody\n")


def test_body_segment_is_frozen() -> None:
    segment = BodySegment(heading_path="T", level=1, body="text")

    with pytest.raises(AttributeError):
        segment.body = "other"  # type: ignore[misc]
