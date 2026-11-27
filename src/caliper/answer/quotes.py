"""Quote verification (§7.7): normalize + substring check.

Normalize whitespace and strip markdown emphasis characters (`*`, `_`, `` ` ``)
from both the quote and the passage body, then require the normalized quote to
be a substring of the normalized body. Verification results feed the retry /
drop / confidence policy in `answer.__init__`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EMPHASIS_RE = re.compile(r"[*_`]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Return `text` with whitespace collapsed, stripped, and emphasis removed."""
    cleaned = _EMPHASIS_RE.sub("", text)
    return _WS_RE.sub(" ", cleaned).strip()


@dataclass(frozen=True, slots=True)
class QuoteVerification:
    """Outcome of verifying one `chunk`'s quote against its passage body."""

    chunk: int
    quote: str
    verified: bool  # True when normalized quote is a substring of normalized body


def verify_quotes(quotes: list[tuple[int, str]], bodies: dict[int, str]) -> list[QuoteVerification]:
    """Verify each `(chunk_id, quote)` against `bodies[chunk_id]` (§7.7).

    A quote verifies only when it is non-empty and its normalized form is a
    substring of the normalized body of its chunk. Missing chunks or empty
    quotes never verify.
    """
    results: list[QuoteVerification] = []
    for chunk, quote in quotes:
        body = bodies.get(chunk, "")
        normalized_quote = normalize(quote)
        normalized_body = normalize(body)
        verified = bool(normalized_quote) and normalized_quote in normalized_body
        results.append(QuoteVerification(chunk=chunk, quote=quote, verified=verified))
    return results
