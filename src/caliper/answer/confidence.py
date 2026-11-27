"""Confidence rules (§7.7): high / medium / low."""

from __future__ import annotations

from typing import Literal

Confidence = Literal["high", "medium", "low"]


def confidence_for(
    *,
    refused: bool,
    refusal_stage: str | None,
    verified_count: int,
    total_citations: int,
    best_similarity: float | None,
    high_confidence_threshold: float,
    dropped: bool = False,
) -> Confidence:
    """§7.7 confidence matrix.

    - `high`   = non-refusal with ≥1 verified citation and best similarity ≥
      `high_confidence_threshold`, no dropped citations
    - `medium` = non-refusal with ≥1 verified citation but weaker scores or a
      dropped citation
    - `low`    = all citations unverified, or refusal produced only by the
      pre-model threshold

    A model refusal (`refusal_stage == "model")` is a deliberate, grounded
    "I don't know" — §7.7's `low` rule covers threshold refusals only, so a
    model refusal is not `low`. Since it is not a non-refusal either, it falls
    in a gap the doc leaves open; recorded as a decision in docs/TASKS.md
    (resolved to `high`, consistent with the accepted default).
    """
    if refused:
        if refusal_stage == "threshold":
            return "low"
        # Model refusal: not threshold-driven, so not `low`.
        return "high"
    if verified_count == 0:
        return "low"
    if dropped or verified_count < total_citations:
        return "medium"
    if best_similarity is None or best_similarity < high_confidence_threshold:
        return "medium"
    return "high"
