"""Unit tests: providers/base.py — cost estimation, normalization, batching,
bounded 429 backoff, factories, and env handling (§7.2, §9.8, §11.3)."""

import pytest

from caliper.config import CaliperConfig, ConfigError
from caliper.providers import (
    EMBEDDING_DIMENSIONS,
    OLLAMA_EMBED_MODEL,
    GeminiChat,
    GeminiEmbeddings,
    OllamaChat,
    OllamaEmbeddings,
    ProviderError,
    Usage,
    estimate_cost,
    get_chat,
    get_embeddings,
    l2_normalize,
    load_prices,
)
from caliper.providers import base as base_module

# --- estimate_cost (§9.8 figures) -----------------------------------------------


def test_estimate_cost_combines_input_and_output_per_mtok() -> None:
    prices = load_prices().entries
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    # §9.8: gemini-3.5-flash-lite $0.10/M in, $0.40/M out.
    assert estimate_cost(usage, "gemini-3.5-flash-lite", prices) == pytest.approx(0.50)


def test_estimate_cost_uses_embed_price_for_embed_tokens() -> None:
    prices = load_prices().entries
    usage = Usage(embed_tokens=1_000_000)
    # §9.8: gemini-embedding-001 $0.15/M in.
    assert estimate_cost(usage, "gemini-embedding-001", prices) == pytest.approx(0.15)


def test_estimate_cost_unknown_model_raises_provider_error() -> None:
    with pytest.raises(ProviderError, match="no price entry"):
        estimate_cost(Usage(prompt_tokens=1), "totally-unknown-model", load_prices().entries)


def test_load_prices_records_as_of_and_binding_figures() -> None:
    prices = load_prices()
    assert prices.as_of  # §9.8: as_of recorded
    flash_lite = prices.entries["gemini-3.5-flash-lite"]
    assert flash_lite["input_per_mtok"] == pytest.approx(0.10)
    assert flash_lite["output_per_mtok"] == pytest.approx(0.40)
    embedding = prices.entries["gemini-embedding-001"]
    assert embedding["embed_per_mtok"] == pytest.approx(0.15)


# --- l2_normalize (ADR-003) -------------------------------------------------------


def test_l2_normalize_scales_to_unit_norm_preserving_direction() -> None:
    normalized = l2_normalize([3.0, 4.0])
    assert normalized == pytest.approx([0.6, 0.8])
    assert sum(value * value for value in normalized) == pytest.approx(1.0)


def test_l2_normalize_zero_vector_raises() -> None:
    with pytest.raises(ProviderError, match="zero vector"):
        l2_normalize([0.0, 0.0, 0.0])


# --- iter_batches (§7.2: ≤100 texts per request) -----------------------------------


def test_iter_batches_caps_batch_size_at_100() -> None:
    batches = list(base_module.iter_batches([f"t{i}" for i in range(250)]))
    assert [len(batch) for batch in batches] == [100, 100, 50]


def test_iter_batches_empty_input_yields_nothing() -> None:
    assert list(base_module.iter_batches([])) == []


def test_iter_batches_custom_size() -> None:
    assert [len(b) for b in base_module.iter_batches(["a", "b", "c"], size=2)] == [2, 1]


# --- call_with_backoff (§7.2: max 5 retries on 429) ---------------------------------


class _Fake429(Exception):
    """Stand-in for an HTTP 429 error from either SDK."""


def _is_fake_429(exc: Exception) -> bool:
    return isinstance(exc, _Fake429)


def test_call_with_backoff_retries_429_with_exponential_delays() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def op() -> str:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise _Fake429("429")
        return "ok"

    result = base_module.call_with_backoff(
        op, describe="demo", is_rate_limited=_is_fake_429, sleep=sleeps.append
    )
    assert result == "ok"
    assert attempts["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_call_with_backoff_raises_quota_exhausted_after_max_retries() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def op() -> str:
        attempts["n"] += 1
        raise _Fake429("429")

    with pytest.raises(ProviderError, match="rate limited") as excinfo:
        base_module.call_with_backoff(
            op, describe="demo", is_rate_limited=_is_fake_429, sleep=sleeps.append
        )
    assert excinfo.value.code == "quota_exhausted"
    # 1 initial attempt + MAX_RETRIES retries; 1s → 2s → 4s → 8s → 16s.
    assert attempts["n"] == 1 + base_module.MAX_RETRIES
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_call_with_backoff_backoff_delay_is_capped() -> None:
    sleeps: list[float] = []

    def op() -> str:
        raise _Fake429("429")

    with pytest.raises(ProviderError):
        base_module.call_with_backoff(
            op,
            describe="demo",
            is_rate_limited=_is_fake_429,
            base_delay_s=30.0,
            max_delay_s=60.0,
            sleep=sleeps.append,
        )
    assert all(delay <= 60.0 for delay in sleeps)


def test_call_with_backoff_wraps_other_errors_without_retry() -> None:
    sleeps: list[float] = []
    attempts = {"n": 0}

    def op() -> str:
        attempts["n"] += 1
        raise ValueError("boom")

    with pytest.raises(ProviderError, match="ValueError: boom") as excinfo:
        base_module.call_with_backoff(
            op, describe="demo", is_rate_limited=_is_fake_429, sleep=sleeps.append
        )
    assert excinfo.value.code == "provider_unavailable"
    assert attempts["n"] == 1
    assert sleeps == []


# --- provider selection + factories via environment (§11.2) ------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CALIPER_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def test_provider_name_defaults_to_gemini(clean_env: None) -> None:
    assert base_module.provider_name() == "gemini"


def test_provider_name_reads_and_normalizes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIPER_PROVIDER", " Ollama ")
    assert base_module.provider_name() == "ollama"


def test_provider_name_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CALIPER_PROVIDER", "openai")
    with pytest.raises(ConfigError, match="CALIPER_PROVIDER"):
        base_module.provider_name()


def test_get_chat_gemini_requires_api_key(clean_env: None) -> None:
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        get_chat(CaliperConfig())


def test_get_chat_gemini_builds_gemini_backend(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert isinstance(get_chat(CaliperConfig()), GeminiChat)


def test_get_chat_ollama_builds_ollama_backend(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALIPER_PROVIDER", "ollama")
    assert isinstance(get_chat(CaliperConfig()), OllamaChat)


def test_get_embeddings_rejects_non_768_dimensions_before_any_env_access(
    clean_env: None,
) -> None:
    cfg = CaliperConfig()
    broken = cfg.model_copy(
        update={"embeddings": cfg.embeddings.model_copy(update={"dimensions": 512})}
    )
    with pytest.raises(ConfigError, match="embeddings.dimensions"):
        get_embeddings(broken)


def test_get_embeddings_gemini_uses_config_model(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    embeddings = get_embeddings(CaliperConfig())
    assert isinstance(embeddings, GeminiEmbeddings)
    # Reading the config flow that was passed in (single underscore attrs are
    # the backend's captured settings; ruff SLF rules are not enabled).
    assert embeddings._model == "gemini-embedding-001"
    assert embeddings._dimensions == EMBEDDING_DIMENSIONS


def test_get_embeddings_ollama_uses_pinned_ollama_model(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALIPER_PROVIDER", "ollama")
    embeddings = get_embeddings(CaliperConfig())
    assert isinstance(embeddings, OllamaEmbeddings)
    assert embeddings._model == OLLAMA_EMBED_MODEL
    assert embeddings._dimensions == EMBEDDING_DIMENSIONS
