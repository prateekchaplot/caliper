"""Provider primitives shared by every backend (§7.2).

This file is single-owner (T04, §14.4). It defines the two backend protocols,
the result/usage dataclasses, the typed component exception, L2 normalization,
the bounded 429 backoff helper, batching, and the price table (§9.8).

Providers are a leaf component above `config` only (§4.2): they must not know
about SQLite, prompts, or retrieval, and they never issue parallel requests.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

from caliper.config import ConfigError

_LOGGER = logging.getLogger(__name__)

# --- Environment contract (§11.2) --------------------------------------------

PROVIDER_ENV_VAR = "CALIPER_PROVIDER"
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
OLLAMA_BASE_URL_ENV_VAR = "OLLAMA_BASE_URL"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# --- Binding constants (§7.2, ADR-003) ----------------------------------------

#: Every embedding is 768 dims, L2-normalized client-side before storage.
EMBEDDING_DIMENSIONS = 768
#: Maximum texts per embedding request (§7.2 "batch ... at up to 100 texts").
EMBED_BATCH_SIZE = 100
#: Bounded retries on HTTP 429: max 5 retries, then raise (§7.2).
MAX_RETRIES = 5
#: Backoff schedule 1s → 2s → 4s → ... capped at 60s (§7.4, §10.3).
BACKOFF_BASE_DELAY_S = 1.0
BACKOFF_MAX_DELAY_S = 60.0


class ProviderError(Exception):
    """A provider backend failed after bounded retries (§11.3).

    `code` carries the §9 envelope code this error maps to at the CLI/web
    edges: `provider_unavailable` (503/exit 4) or `quota_exhausted`
    (503/exit 5, after 429 retries are exhausted).
    """

    def __init__(self, message: str, *, code: ProviderErrorCode = "provider_unavailable") -> None:
        super().__init__(message)
        self.code: ProviderErrorCode = code


ProviderErrorCode = Literal["provider_unavailable", "quota_exhausted"]

# --- Public result types (§9.8) ------------------------------------------------


@dataclass(frozen=True)
class ChatResult:
    """One chat completion: answer text plus usage capture (§7.2, §9.8)."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


@dataclass(frozen=True)
class Usage:
    """Token usage of one request, for the §11.3 cost estimator."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0


# --- Backend protocols (§7.2 public interface, binding) ------------------------


class ChatBackend(Protocol):
    def complete(
        self, *, system: str, user: str, model: str, temperature: float, seed: int | None
    ) -> ChatResult: ...


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# --- Price table (§9.8) --------------------------------------------------------


class PriceEntry(TypedDict):
    """USD per million tokens for one model."""

    input_per_mtok: float
    output_per_mtok: float
    embed_per_mtok: float


@dataclass(frozen=True)
class Prices:
    """The price table with its `as_of` date recorded (A3, §9.8)."""

    as_of: str
    entries: dict[str, PriceEntry]


def _price_entry(model: str, raw: object) -> PriceEntry:
    if not isinstance(raw, dict):
        raise ProviderError(f"prices.json: entry for {model!r} must be an object")
    values: list[float] = []
    for key in ("input_per_mtok", "output_per_mtok", "embed_per_mtok"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProviderError(f"prices.json: {model}.{key} must be a number")
        values.append(float(value))
    return PriceEntry(input_per_mtok=values[0], output_per_mtok=values[1], embed_per_mtok=values[2])


def load_prices() -> Prices:
    """Load `prices.json` shipped next to this module (§9.8, §6.1)."""
    path = Path(__file__).with_name("prices.json")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderError(f"cannot read price table {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProviderError("prices.json: top level must be an object")
    as_of = raw.get("as_of")
    if not isinstance(as_of, str) or not as_of:
        raise ProviderError("prices.json: 'as_of' must be a non-empty date string")
    table_raw = raw.get("prices")
    if not isinstance(table_raw, dict):
        raise ProviderError("prices.json: 'prices' must be an object")
    entries = {str(model): _price_entry(model, entry) for model, entry in table_raw.items()}
    return Prices(as_of=as_of, entries=entries)


def estimate_cost(usage: Usage, model: str, prices: Mapping[str, PriceEntry]) -> float:
    """(prompt*in + completion*out + embed*embed) / 1e6, USD (§7.2, §11.3).

    Raises ProviderError for a model missing from the table: a wrong price
    figure never invalidates logged results (A3), but a missing entry is a
    real gap and must fail loudly instead of logging $0.00 silently.
    """
    entry = prices.get(model)
    if entry is None:
        raise ProviderError(f"no price entry for model {model!r} in prices.json")
    total = (
        usage.prompt_tokens * entry["input_per_mtok"]
        + usage.completion_tokens * entry["output_per_mtok"]
        + usage.embed_tokens * entry["embed_per_mtok"]
    )
    return total / 1e6


# --- Vector math (ADR-003) ------------------------------------------------------


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Return the L2-normalized copy of `vector` (client-side, ADR-003).

    All 768-dim vectors are normalized here before storage or comparison;
    only 3072-dim output would be API-normalized, and we never use it.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ProviderError("cannot L2-normalize a zero vector")
    return [value / norm for value in vector]


def iter_batches(texts: Sequence[str], size: int = EMBED_BATCH_SIZE) -> Iterator[list[str]]:
    """Split `texts` into batches of at most `size` (≤100 per request, §7.2)."""
    for start in range(0, len(texts), size):
        yield list(texts[start : start + size])


# --- Bounded 429 backoff (§7.2 binding) -----------------------------------------


def call_with_backoff[T](
    op: Callable[[], T],
    *,
    describe: str,
    is_rate_limited: Callable[[Exception], bool],
    max_retries: int = MAX_RETRIES,
    base_delay_s: float = BACKOFF_BASE_DELAY_S,
    max_delay_s: float = BACKOFF_MAX_DELAY_S,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `op` with bounded exponential backoff on HTTP 429 (§7.2).

    429 responses are retried up to `max_retries` times with delays
    base*2**n capped at `max_delay_s`, then ProviderError(code=quota_exhausted)
    is raised. Any other exception is wrapped immediately as
    ProviderError(code=provider_unavailable) — no unbounded retries, never a
    silent swallow (§16 provider discipline). Raw tracebacks stay at DEBUG.
    """
    for attempt in range(max_retries + 1):
        try:
            return op()
        except Exception as exc:  # noqa: BLE001 - single typed wrap, never swallowed
            if is_rate_limited(exc) and attempt < max_retries:
                delay = min(base_delay_s * (2**attempt), max_delay_s)
                _LOGGER.warning(
                    "%s: rate limited (HTTP 429); retrying in %.1fs (retry %d/%d)",
                    describe,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                sleep(delay)
                continue
            if is_rate_limited(exc):
                _LOGGER.debug("%s: still rate limited after retries", describe, exc_info=True)
                raise ProviderError(
                    f"{describe} still rate limited after {max_retries} retries",
                    code="quota_exhausted",
                ) from exc
            _LOGGER.debug("%s failed", describe, exc_info=True)
            raise ProviderError(f"{describe} failed: {type(exc).__name__}: {exc}") from exc
    raise AssertionError("unreachable: call_with_backoff loop is exhaustive")


# --- Environment accessors (§11.2) ----------------------------------------------


def provider_name() -> Literal["gemini", "ollama"]:
    """The selected provider from CALIPER_PROVIDER (default: gemini)."""
    raw = os.environ.get(PROVIDER_ENV_VAR, "gemini").strip().lower()
    if raw == "gemini":
        return "gemini"
    if raw == "ollama":
        return "ollama"
    raise ConfigError(f"{PROVIDER_ENV_VAR} must be 'gemini' or 'ollama', got {raw!r}")


def required_env(name: str) -> str:
    """Read a required environment variable, failing with its name (§11.4)."""
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"environment variable {name} is required but not set (see .env)")
    return value


def gemini_api_key() -> str:
    """GEMINI_API_KEY, required iff provider=gemini (§11.2). Never logged."""
    return required_env(GEMINI_API_KEY_ENV_VAR)


def ollama_base_url() -> str:
    """OLLAMA_BASE_URL (default http://127.0.0.1:11434, §11.2)."""
    return os.environ.get(OLLAMA_BASE_URL_ENV_VAR, DEFAULT_OLLAMA_BASE_URL)
