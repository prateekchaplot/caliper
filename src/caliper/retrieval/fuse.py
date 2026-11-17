"""Reciprocal Rank Fusion (§7.5): combine two ranked candidate lists.

Pure module: no I/O, no config, no db access (§7.5 "rrf ... in fuse.py"). The
hybrid_rrf wiring in `caliper.retrieval` calls `rrf` with the vector KNN
top-`vector_top_k` and the FTS5 BM25 top-`fts_top_k` candidate lists and
stamps the fused passages `stage="rrf"`; the cross-encoder stage (T18) sits
on top of this fused list.
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf(
    vector_list: Sequence[int], fts_list: Sequence[int], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse two best-first ranked lists with Reciprocal Rank Fusion (§7.5).

    Textbook RRF: a document's fused score is the sum, over each input list
    that contains it, of `1 / (k + rank)` with 1-based ranks — a document in
    both lists collects both contributions, a document in one list only gets
    one. `k` dampens top ranks: with small `k` the first position dominates,
    while the preset `k = 60` (§9.7) makes appearing in both lists matter
    more than ranking first in one.

    Inputs are ordered best-first and must not repeat a document within one
    list (ranked candidate lists never do). Identity is the chunk id, so a
    document present in both lists is deduplicated into a single fused entry.

    Returns (chunk_id, fused_score) pairs sorted by descending fused score,
    ties broken by ascending chunk_id — the same deterministic tie rule as
    vector mode (T09). An empty pair of inputs yields []; one empty input
    degrades to the other list re-scored by RRF. No sentinel scores are
    fabricated; the fused score is always a positive float.
    """
    fused: dict[int, float] = {}
    for ranked in (vector_list, fts_list):
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))
