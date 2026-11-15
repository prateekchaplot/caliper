"""Ollama backend: the local $0 fallback (§5, §7.2).

Uses `qwen2.5:7b-instruct` for chat and `nomic-embed-text` (768 dims) for
embeddings — the two pinned Ollama models (§7.2 binding). Same protocol, same
client-side L2 normalization, same bounded 429 backoff as the Gemini backend.
Transport errors (Ollama not running) raise ProviderError(code=
"provider_unavailable") immediately — only HTTP 429 is retried.

httpx.Client is injected through the public constructor, so tests drive this
backend with `httpx.MockTransport` (an injectable transport, §15) and never
touch the network or patch internals.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import httpx

from caliper.providers.base import (
    EMBEDDING_DIMENSIONS,
    ChatResult,
    ProviderError,
    call_with_backoff,
    iter_batches,
    l2_normalize,
)

#: §7.2 pin — the Ollama answer model (override answer.model in caliper.toml
#: when switching providers; the fingerprint gate catches any drift, FR-10).
OLLAMA_CHAT_MODEL = "qwen2.5:7b-instruct"
#: §7.2 pin — the Ollama embedding model, dimension-compatible at 768 (ADR-003).
OLLAMA_EMBED_MODEL = "nomic-embed-text"

#: Local model calls can be slow on CPU; generous but bounded.
OLLAMA_HTTP_TIMEOUT_S = 120.0


def _is_rate_limited(exc: Exception) -> bool:
    """True when httpx saw an HTTP 429 status."""
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def _token_count(data: Mapping[str, object], key: str) -> int:
    """Read an optional integer token-count field, defaulting to 0."""
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _vector_list(
    data: Mapping[str, object], key: str, batch_len: int, dimensions: int, describe: str
) -> list[list[float]]:
    """Extract, validate, and L2-normalize an embeddings list (ADR-003)."""
    raw = data.get(key)
    if not isinstance(raw, list):
        raise ProviderError(f"{describe}: response is missing a '{key}' list")
    if len(raw) != batch_len:
        raise ProviderError(f"{describe}: response has {len(raw)} vectors for {batch_len} inputs")
    vectors: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list):
            raise ProviderError(f"{describe}: '{key}' entries must be lists of numbers")
        values: list[float] = []
        for value in item:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProviderError(f"{describe}: '{key}' entries must be lists of numbers")
            values.append(float(value))
        if len(values) != dimensions:
            raise ProviderError(f"{describe}: vector has {len(values)} dims, expected {dimensions}")
        vectors.append(l2_normalize(values))
    return vectors


def _post_json(
    client: httpx.Client,
    path: str,
    payload: Mapping[str, object],
    *,
    describe: str,
    sleep: Callable[[float], None],
) -> dict[str, object]:
    """POST `payload` and return the decoded JSON object, with bounded 429 backoff."""

    def op() -> object:
        response = client.post(path, json=payload)
        response.raise_for_status()
        raw: object = response.json()
        return raw

    data = call_with_backoff(op, describe=describe, is_rate_limited=_is_rate_limited, sleep=sleep)
    if not isinstance(data, dict):
        raise ProviderError(f"{describe}: response body is not a JSON object")
    return data


class OllamaChat:
    """ChatBackend over Ollama's /api/chat (§7.2)."""

    def __init__(
        self, client: httpx.Client, *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self._client = client
        self._sleep = sleep

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, seed: int | None
    ) -> ChatResult:
        options: dict[str, float | int] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
        describe = f"Ollama chat call ({model})"
        data = _post_json(self._client, "/api/chat", payload, describe=describe, sleep=self._sleep)
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderError("Ollama chat response is missing 'message'")
        text = message.get("content")
        if not isinstance(text, str) or not text:
            raise ProviderError("Ollama chat response has no 'message.content' text")
        return ChatResult(
            text=text,
            prompt_tokens=_token_count(data, "prompt_eval_count"),
            completion_tokens=_token_count(data, "eval_count"),
            model=model,
        )


class OllamaEmbeddings:
    """EmbeddingBackend over Ollama's batched /api/embed (§7.2)."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        model: str = OLLAMA_EMBED_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._sleep = sleep

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in iter_batches(texts):
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, batch: Sequence[str]) -> list[list[float]]:
        if not batch:
            return []
        payload: dict[str, object] = {"model": self._model, "input": list(batch)}
        describe = f"Ollama embed call ({self._model}, {len(batch)} texts)"
        data = _post_json(self._client, "/api/embed", payload, describe=describe, sleep=self._sleep)
        return _vector_list(data, "embeddings", len(batch), self._dimensions, describe)
