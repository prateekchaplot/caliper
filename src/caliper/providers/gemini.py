"""Gemini backend via the `google-genai` SDK (§5, §7.2).

Chat runs whatever model the caller passes (the config-pinned answer model,
default `gemini-3.5-flash-lite` per OQ1). Embeddings use the config-pinned
embedding model at `output_dimensionality=768` with task types
RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY; every returned vector is L2-normalized
client-side (ADR-003 — only 3072-dim output is API-normalized).

HTTP 429 responses back off exponentially, bounded at MAX_RETRIES, then raise
ProviderError(code="quota_exhausted"). Nothing here is parallel and nothing is
cached (§16 provider discipline).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from google.genai.types import EmbedContentConfig, GenerateContentConfig

from caliper.providers.base import (
    EMBEDDING_DIMENSIONS,
    ChatResult,
    ProviderError,
    call_with_backoff,
    iter_batches,
    l2_normalize,
)

GEMINI_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
GEMINI_TASK_QUERY = "RETRIEVAL_QUERY"


class GeminiModels(Protocol):
    """The narrow slice of `google.genai.client.Models` we use.

    The real SDK client satisfies this structurally (checked under mypy
    strict); tests inject a fake with the same shape through the public
    constructor — no patch() anywhere (§15).
    """

    def generate_content(
        self, *, model: str, contents: str, config: GenerateContentConfig
    ) -> types.GenerateContentResponse: ...

    def embed_content(
        self, *, model: str, contents: Sequence[str], config: EmbedContentConfig
    ) -> types.EmbedContentResponse: ...


def _is_rate_limited(exc: Exception) -> bool:
    """True when the SDK raised an HTTP 429 (google.genai.errors.APIError)."""
    return isinstance(exc, genai_errors.APIError) and exc.code == 429


class GeminiSession:
    """Owns a `genai.Client` so its `.models` handle stays usable (§7.2).

    The google-genai SDK's `Models` object holds only a weak reference to its
    `Client`: once the `Client` is garbage-collected, every request fails with
    "Cannot send a request, as the client has been closed". Factories and the
    smoke script therefore build a `GeminiSession` and hand it to the backends,
    which store it — keeping the client alive for the backend's lifetime.
    """

    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.models = self.client.models


class GeminiChat:
    """ChatBackend over the Gemini API (§7.2). Model is passed per call."""

    def __init__(
        self,
        models: GeminiModels,
        *,
        sleep: Callable[[float], None] = time.sleep,
        keepalive: object | None = None,
    ) -> None:
        self._models = models
        self._sleep = sleep
        # Holds the GeminiSession (and thus the genai.Client) referenced for
        # this backend's lifetime; see GeminiSession's docstring.
        self._keepalive = keepalive

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, seed: int | None
    ) -> ChatResult:
        config = GenerateContentConfig(
            system_instruction=system, temperature=temperature, seed=seed
        )
        response = call_with_backoff(
            lambda: self._models.generate_content(model=model, contents=user, config=config),
            describe=f"Gemini chat call ({model})",
            is_rate_limited=_is_rate_limited,
            sleep=self._sleep,
        )
        text = response.text
        if not text:
            raise ProviderError(f"Gemini chat returned no text for model {model}")
        usage = response.usage_metadata
        return ChatResult(
            text=text,
            prompt_tokens=int(usage.prompt_token_count or 0) if usage else 0,
            completion_tokens=int(usage.candidates_token_count or 0) if usage else 0,
            model=model,
        )


class GeminiEmbeddings:
    """EmbeddingBackend over the Gemini API, batched at ≤100 texts (§7.2)."""

    def __init__(
        self,
        models: GeminiModels,
        *,
        model: str,
        dimensions: int = EMBEDDING_DIMENSIONS,
        sleep: Callable[[float], None] = time.sleep,
        keepalive: object | None = None,
    ) -> None:
        self._models = models
        self._model = model
        self._dimensions = dimensions
        self._sleep = sleep
        # Holds the GeminiSession (and thus the genai.Client) referenced for
        # this backend's lifetime; see GeminiSession's docstring.
        self._keepalive = keepalive

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in iter_batches(texts):
            vectors.extend(self._embed_batch(batch, GEMINI_TASK_DOCUMENT))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text], GEMINI_TASK_QUERY)[0]

    def _embed_batch(self, batch: Sequence[str], task_type: str) -> list[list[float]]:
        if not batch:
            return []
        response = call_with_backoff(
            lambda: self._models.embed_content(
                model=self._model,
                contents=batch,
                config=EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dimensions
                ),
            ),
            describe=f"Gemini embed call ({self._model}, {len(batch)} texts, {task_type})",
            is_rate_limited=_is_rate_limited,
            sleep=self._sleep,
        )
        embeddings = response.embeddings
        if embeddings is None or len(embeddings) != len(batch):
            found = len(embeddings) if embeddings is not None else 0
            raise ProviderError(
                f"Gemini embedding response has {found} vectors for {len(batch)} inputs"
            )
        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = embedding.values
            if values is None:
                raise ProviderError("Gemini embedding response is missing values")
            if len(values) != self._dimensions:
                raise ProviderError(
                    f"Gemini embedding returned {len(values)} dims, expected {self._dimensions}"
                )
            # Client-side L2 normalization: mandatory at 768 dims (ADR-003).
            vectors.append(l2_normalize(values))
        return vectors
