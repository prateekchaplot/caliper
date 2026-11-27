"""T11 unit tests: quote verification (§7.7, quotes.py)."""

from caliper.answer.quotes import normalize, verify_quotes


def test_normalize_collapses_whitespace_and_strips_emphasis():
    assert normalize("  Hello   world  ") == "Hello world"
    assert normalize("*italic* and _underscore_ and `code`") == "italic and underscore and code"
    assert normalize("Mixed   CASE *with* emphasis") == "Mixed CASE with emphasis"


def test_verify_quotes_matches_normalized_substring():
    bodies = {1: "The quick brown fox jumps over the lazy dog."}
    results = verify_quotes([(1, "quick brown fox")], bodies)
    assert len(results) == 1
    assert results[0].verified is True
    assert results[0].chunk == 1
    assert results[0].quote == "quick brown fox"


def test_verify_quotes_whitespace_insensitive_case_sensitive():
    bodies = {1: "The quick   brown fox"}
    # Whitespace differences are normalized away.
    results = verify_quotes([(1, "The quick brown fox")], bodies)
    assert results[0].verified is True
    # Case differences are NOT normalized (§7.7: whitespace + emphasis only).
    results = verify_quotes([(1, "THE QUICK BROWN FOX")], bodies)
    assert results[0].verified is False


def test_verify_quotes_strips_emphasis_on_both_sides():
    bodies = {1: "The *quick* brown fox"}
    results = verify_quotes([(1, "quick brown")], bodies)
    assert results[0].verified is True


def test_verify_quotes_empty_quote_fails():
    bodies = {1: "anything"}
    results = verify_quotes([(1, "")], bodies)
    assert results[0].verified is False


def test_verify_quotes_missing_chunk_fails():
    bodies = {1: "anything"}
    results = verify_quotes([(2, "anything")], bodies)
    assert results[0].verified is False


def test_verify_quotes_not_a_substring_fails():
    bodies = {1: "the quick brown fox"}
    results = verify_quotes([(1, "the slow red fox")], bodies)
    assert results[0].verified is False


def test_verify_quotes_multiple_mixed():
    bodies = {1: "alpha beta gamma", 2: "delta epsilon zeta"}
    results = verify_quotes([(1, "alpha beta"), (1, "nope nowhere"), (2, "epsilon")], bodies)
    assert results[0].verified is True
    assert results[1].verified is False
    assert results[2].verified is True
