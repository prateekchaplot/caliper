"""Provider factories (§7.2): `get_chat(cfg)` and `get_embeddings(cfg)`.

Provider selection and endpoints come from the environment (§11.2):
CALIPER_PROVIDER (gemini|ollama, default gemini), GEMINI_API_KEY,
OLLAMA_BASE_URL. Model IDs and generation parameters come from the resolved
`CaliperConfig` (T02) — never hardcoded here. Re-exports the public surface
of the component so callers can `from caliper.providers import ...`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from google import genai

from caliper.config import ConfigError
from caliper.providers.base import (
    EMBEDDING_DIMENSIONS,
    ChatBackend,
    ChatResult,
    EmbeddingBackend,
    PriceEntry,
    Prices,
    ProviderError,
    Usage,
    estimate_cost,
    gemini_api_key,
    l2_normalize,
    load_prices,
    ollama_base_url,
    provider_name,
)
from caliper.providers.gemini import GeminiChat, GeminiEmbeddings
from caliper.providers.ollama import (
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_HTTP_TIMEOUT_S,
    OllamaChat,
    OllamaEmbeddings,
)

if TYPE_CHECKING:
    from caliper.config import CaliperConfig

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "OLLAMA_CHAT_MODEL",
    "OLLAMA_EMBED_MODEL",
    "OLLAMA_HTTP_TIMEOUT_S",
    "ChatBackend",
    "ChatResult",
    "EmbeddingBackend",
    "GeminiChat",
    "GeminiEmbeddings",
    "OllamaChat",
    "OllamaEmbeddings",
    "PriceEntry",
    "Prices",
    "ProviderError",
    "Usage",
    "estimate_cost",
    "get_chat",
    "get_embeddings",
    "l2_normalize",
    "load_prices",
]


def get_chat(cfg: CaliperConfig) -> ChatBackend:
    """Build the configured chat backend (§7.2).

    The caller (answer) passes `cfg.answer.model` / temperature / seed to
    `complete()` on every call; this factory only selects the transport.
    """
    if provider_name() == "gemini":
        genai_client = genai.Client(api_key=gemini_api_key())
        return GeminiChat(genai_client.models)
    ollama_client = httpx.Client(
        base_url=ollama_base_url(), timeout=httpx.Timeout(OLLAMA_HTTP_TIMEOUT_S)
    )
    return OllamaChat(ollama_client)


def get_embeddings(cfg: CaliperConfig) -> EmbeddingBackend:
    """Build the configured embedding backend (§7.2).

    Both backends must emit 768-dim, client-side-L2-normalized vectors
    (ADR-003, dimension-compatible with the `vec0` schema and with the Ollama
    fallback), so a different `embeddings.dimensions` is rejected here before
    any model call could be made.
    """
    if cfg.embeddings.dimensions != EMBEDDING_DIMENSIONS:
        raise ConfigError(
            f"embeddings.dimensions must be {EMBEDDING_DIMENSIONS} "
            f"(ADR-003: vec0 schema + Ollama compatibility), got {cfg.embeddings.dimensions}"
        )
    if provider_name() == "gemini":
        genai_client = genai.Client(api_key=gemini_api_key())
        return GeminiEmbeddings(
            genai_client.models, model=cfg.embeddings.model, dimensions=cfg.embeddings.dimensions
        )
    ollama_client = httpx.Client(
        base_url=ollama_base_url(), timeout=httpx.Timeout(OLLAMA_HTTP_TIMEOUT_S)
    )
    return OllamaEmbeddings(
        ollama_client, model=OLLAMA_EMBED_MODEL, dimensions=cfg.embeddings.dimensions
    )
