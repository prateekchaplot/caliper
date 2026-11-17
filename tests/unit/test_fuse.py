"""T10 unit tests: the pure Reciprocal Rank Fusion function (§7.5).

Hand-computed fixtures against the textbook formula
`score(d) = Σ 1/(k + rank_i(d))` over each input list containing d, ranks
1-based, lists ordered best-first. No db, no config, no fakes — fuse.py is
pure, so these tests are too.
"""

import pytest

from caliper.retrieval.fuse import rrf

K = 60


def _score(rank: int, k: int = K) -> float:
    """One list's RRF contribution for a 1-based rank."""
    return 1.0 / (k + rank)


# --- the textbook formula --------------------------------------------------------


def test_doc_in_both_lists_collects_both_contributions() -> None:
    # vector: 1st, 2nd, 3rd; fts: 3rd, 2nd, 4th — every fused score by hand.
    fused = rrf([1, 2, 3], [3, 2, 4], k=K)

    expected = {
        1: _score(1),  # vector rank 1 only
        2: _score(2) + _score(2),  # vector rank 2 + fts rank 2
        3: _score(3) + _score(1),  # vector rank 3 + fts rank 1
        4: _score(3),  # fts rank 3 only
    }
    assert [doc for doc, _ in fused] == [3, 2, 1, 4]
    for doc, score in fused:
        assert score == pytest.approx(expected[doc])


def test_doc_in_one_list_only_keeps_one_contribution() -> None:
    fused = rrf([7, 3, 5], [], k=K)

    assert [doc for doc, _ in fused] == [7, 3, 5]
    assert fused[0][1] == pytest.approx(_score(1))
    assert fused[1][1] == pytest.approx(_score(2))
    assert fused[2][1] == pytest.approx(_score(3))


def test_ranks_are_one_based_positions_of_the_input_lists() -> None:
    # Only the position in each list matters — not any upstream score.
    fused = rrf([100, 200], [300], k=K)
    by_doc = dict(fused)
    assert by_doc[100] == pytest.approx(_score(1))
    assert by_doc[200] == pytest.approx(_score(2))
    assert by_doc[300] == pytest.approx(_score(1))


# --- k semantics ------------------------------------------------------------------


def test_k_dampens_top_ranks_appearing_twice_beats_ranking_first_once() -> None:
    # doc 2: rank 2 in BOTH lists; docs 1 and 9: rank 1 in one list each;
    # doc 3: rank 3 in the vector list only. k = 60 (preset): two mid-rank
    # contributions outweigh one top rank → doc 2 first, docs 1/9 tie behind
    # it, and the vector-only doc 3 trails them all.
    fused = rrf([1, 2, 3], [9, 2], k=K)
    assert [doc for doc, _ in fused] == [2, 1, 9, 3]
    assert fused[0][1] == pytest.approx(2.0 / (K + 2))
    assert fused[1][1] == pytest.approx(1.0 / (K + 1))
    assert fused[2][1] == pytest.approx(1.0 / (K + 1))
    assert fused[3][1] == pytest.approx(1.0 / (K + 3))

    # k = 0: 1/rank makes a single first place worth 1.0, so docs 1/2/9 all
    # tie (doc 2 collects 1/2 + 1/2) — the same lists flip to id order. k is
    # what breaks the tie towards the document appearing in both lists.
    fused_k0 = rrf([1, 2, 3], [9, 2], k=0)
    assert [doc for doc, _ in fused_k0] == [1, 2, 9, 3]


def test_preset_k_60_default_matches_explicit_k() -> None:
    assert rrf([1, 2, 3], [3, 2, 4]) == rrf([1, 2, 3], [3, 2, 4], k=60)


# --- ties and determinism ---------------------------------------------------------


def test_ties_break_by_ascending_chunk_id() -> None:
    # Both documents collect identical contributions (1st + 2nd in reverse
    # lists) → identical fused scores → ascending chunk_id stabilizes order.
    fused = rrf([5, 2], [2, 5], k=K)
    assert [doc for doc, _ in fused] == [2, 5]
    assert fused[0][1] == pytest.approx(_score(1) + _score(2))
    assert fused[1][1] == pytest.approx(_score(1) + _score(2))


def test_rrf_is_deterministic_on_fixed_input() -> None:
    assert rrf([4, 8, 15], [16, 8], k=K) == rrf([4, 8, 15], [16, 8], k=K)


# --- graceful edges ----------------------------------------------------------------


def test_both_lists_empty_yields_empty() -> None:
    assert rrf([], []) == []


def test_empty_list_degrades_to_the_other_list_reranked() -> None:
    # One candidate source missing (e.g. zero FTS hits) is graceful: the other
    # list's order is kept, re-scored with RRF contributions (§9.3).
    fused = rrf([11, 22], [], k=K)
    assert [doc for doc, _ in fused] == [11, 22]
    assert fused[0][1] == pytest.approx(_score(1))
    assert fused[1][1] == pytest.approx(_score(2))
