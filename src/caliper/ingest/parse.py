"""Markdown page parser: page title, heading paths, body segments (§7.4).

Pure text-in/text-out: callers (T07) hand in a corpus-relative page path (as
produced by `source.discover_pages`) and the page's text content; this module
performs no file I/O over the cache. The page URL comes from T05's frozen
`source.page_to_url` (A1 mapping); an out-of-contract path surfaces as
`SourceError` from that call.

Binding decisions (recorded here for §7.4 and the T07 freeze):

- **Title:** the first ATX H1 (a line starting `# ` in column 0). Setext
  underlines are not treated as headings (the corpus uses ATX exclusively).
- **Heading path:** the stack of headings open at a segment's position,
  joined with `" > "`: `"<page title> > <H2> > <H3>"` per §7.4. Deeper
  headings (H4+) continue the same pattern — the path simply grows one
  `" > <Hn>"` per open level (azure-docs pages rarely go past H3, but the
  rule stays uniform instead of truncating).
- **No H1:** raises `ParseError` with `reason="no H1"` — T07 reports the
  §9.4 skip reason `parse-error: no H1` from that attribute.
- **Repeated H1** (not expected in the corpus): treated as a level-2 divider,
  so the path becomes `"<title> > <text>"` and structure stays consistent.
- **Cleaning:** YAML front matter (a leading `---` block) is stripped; heading
  lines become structure, not body; fenced code blocks (` ``` `/` ~~~ `) are
  never scanned for headings (a `#` comment inside a sample is body text).
  Everything else is kept **verbatim**: bodies remain GitHub markdown, the
  canonical quote source (§17 — quote verification normalizes emphasis, so
  stripping emphasis here would break verbatim quotes).
- **Content before the first H1** is dropped with a DEBUG log (the corpus
  places the H1 right after front matter).
- **Empty body:** an H1 with no following content parses fine with zero
  segments (T07 counts the page OK and writes zero chunks for it).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from caliper.ingest.source import page_to_url

_LOGGER = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


class ParseError(Exception):
    """A page failed structural parsing (§11.3 typed component error).

    `reason` is a short stable slug (currently only `"no H1"`); T07 reports
    the §9.4 skip reason as `f"parse-error: {exc.reason}"`.
    """

    def __init__(self, rel_path: str, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{rel_path}: {reason}")


@dataclass(frozen=True, slots=True)
class BodySegment:
    """The text under one heading position (§7.4 body segment)."""

    heading_path: str
    level: int
    body: str


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """A parsed markdown page: title, URL, and body segments in order."""

    rel_path: str
    page_url: str
    title: str
    segments: list[BodySegment]


def _strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block; return the text unchanged otherwise."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[index + 1 :])
    return text  # unterminated front matter: keep the original text


def parse_page(rel_path: str, text: str) -> ParsedPage:
    """Parse one markdown page into title, URL, and heading-aligned segments.

    `rel_path` is a corpus-relative page path (forward slashes) as produced
    by `source.discover_pages`; `text` is the page's full content. Raises
    `ParseError` (reason "no H1") when the page has no first-level heading,
    and `SourceError` (from `page_to_url`) for out-of-contract paths.
    """
    page_url = page_to_url(rel_path)
    body_text = _strip_front_matter(text)

    title: str | None = None
    stack: list[tuple[int, str]] = []  # open headings: (level, text)
    segments: list[BodySegment] = []
    buf: list[str] = []
    current_path: str | None = None
    current_level: int = 0
    fence: str | None = None  # fence marker character while inside a block

    def flush() -> None:
        if current_path is None:
            buf.clear()
            return
        body = "\n".join(buf).strip()
        buf.clear()
        if body:
            segments.append(BodySegment(heading_path=current_path, level=current_level, body=body))

    for line in body_text.splitlines():
        if fence is not None:
            buf.append(line)
            if line.lstrip().startswith(fence * 3):
                fence = None
            continue
        fence_match = _FENCE_RE.match(line)
        if fence_match is not None:
            fence = fence_match.group(1)[0]
            buf.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading is None:
            buf.append(line)
            continue
        flush()
        level = len(heading.group(1))
        heading_text = heading.group(2).strip()
        if level == 1:
            if title is None:
                title = heading_text
                stack = [(1, heading_text)]
            else:
                stack = [(1, title), (2, heading_text)]
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
        current_path = " > ".join(open_text for _, open_text in stack)
        current_level = level
    flush()

    if title is None:
        _LOGGER.debug("page %s has no H1 heading; reporting parse error", rel_path)
        raise ParseError(rel_path, "no H1")
    return ParsedPage(rel_path=rel_path, page_url=page_url, title=title, segments=segments)
