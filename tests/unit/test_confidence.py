"""T11 unit tests: confidence rules (§7.7, confidence.py)."""

from caliper.answer.confidence import confidence_for


def test_high_when_verified_and_score_above_threshold():
    assert (
        confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=1,
            total_citations=1,
            best_similarity=0.9,
            high_confidence_threshold=0.60,
        )
        == "high"
    )


def test_medium_when_verified_but_score_below_threshold():
    assert (
        confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=1,
            total_citations=1,
            best_similarity=0.4,
            high_confidence_threshold=0.60,
        )
        == "medium"
    )


def test_medium_when_dropped_citation():
    assert (
        confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=1,
            total_citations=2,
            best_similarity=0.9,
            high_confidence_threshold=0.60,
        )
        == "medium"
    )


def test_low_when_all_citations_unverified():
    assert (
        confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=0,
            total_citations=1,
            best_similarity=0.9,
            high_confidence_threshold=0.60,
        )
        == "low"
    )


def test_low_when_threshold_refusal():
    assert (
        confidence_for(
            refused=True,
            refusal_stage="threshold",
            verified_count=0,
            total_citations=0,
            best_similarity=None,
            high_confidence_threshold=0.60,
        )
        == "low"
    )


def test_high_when_model_refusal_not_low():
    assert (
        confidence_for(
            refused=True,
            refusal_stage="model",
            verified_count=0,
            total_citations=0,
            best_similarity=None,
            high_confidence_threshold=0.60,
        )
        == "high"
    )


def test_boundary_similarity_equals_threshold_is_high():
    assert (
        confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=1,
            total_citations=1,
            best_similarity=0.60,
            high_confidence_threshold=0.60,
        )
        == "high"
    )
