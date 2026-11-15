"""Unit tests: providers/gemini.py — chat parameters/usage, embedding batching
(≤100), task types, client-side normalization, and 429 backoff.

The SDK client seam is injected through the public constructor (§15): a fake
`GeminiModels` object with scripted responses and exceptions. No network, no
patch() on internals.
"""

import math

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from caliper.providers import EMBEDDING_DIMENSIONS, ProviderError
from caliper.providers.gemini import (
    GEMINI_TASK_DOCUMENT,
    GEMINI_TASK_QUERY,
    GeminiChat,
    GeminiEmbeddings,
)

# --- test doubles (constructor-injected, like tests/fakes.py but SDK-shaped) -----


def _chat_response(
    text: str, *, prompt_tokens: int = 10, completion_tokens: int = 5
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_construct(
        candidates=[
            types.Candidate.model_construct(
                content=types.Content.model_construct(parts=[types.Part.model_construct(text=text)])
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata.model_construct(
            prompt_token_count=prompt_tokens, candidates_token_count=completion_tokens
        ),
        model_version="fake",
    )


def _empty_chat_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_construct(candidates=[])


def _embed_response(vectors: list[list[float]]) -> types.EmbedContentResponse:
    return types.EmbedContentResponse.model_construct(
        embeddings=[types.ContentEmbedding(values=vector) for vector in vectors]
    )


def _rate_limited() -> genai_errors.APIError:
    return genai_errors.APIError(429, {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"})


def _server_error() -> genai_errors.APIError:
    return genai_errors.APIError(500, {"message": "internal", "status": "INTERNAL"})


def _sleep_recorder() -> tuple[list[float], object]:
    sleeps: list[float] = []

    def record(seconds: float) -> None:
        sleeps.append(seconds)

    return sleeps, record


class FakeGeminiModels:
    """Fake of the narrow `GeminiModels` protocol with scripted results.

    Unscripted calls auto-reply: chat with a canned text response, embeddings
    with a 768-dim constant vector per input (so normalization is a no-op).
    """

    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.embed_calls: list[dict[str, object]] = []
        self.generate_results: list[object] = []  # responses or exceptions
        self.embed_results: list[object] = []

    def generate_content(
        self, *, model: str, contents: str, config: types.GenerateContentConfig
    ) -> types.GenerateContentResponse:
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        if self.generate_results:
            item = self.generate_results.pop(0)
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, types.GenerateContentResponse)
            return item
        return _chat_response("auto reply")

    def embed_content(
        self, *, model: str, contents: list[str], config: types.EmbedContentConfig
    ) -> types.EmbedContentResponse:
        self.embed_calls.append({"model": model, "contents": list(contents), "config": config})
        if self.embed_results:
            item = self.embed_results.pop(0)
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, types.EmbedContentResponse)
            return item
        return _embed_response([[2.0] * EMBEDDING_DIMENSIONS for _ in contents])


# --- chat (§7.2 complete) ---------------------------------------------------------


def test_complete_sends_system_temperature_seed_and_returns_usage() -> None:
    models = FakeGeminiModels()
    models.generate_results = [
        _chat_response("the answer", prompt_tokens=321, completion_tokens=44)
    ]
    chat = GeminiChat(models)

    result = chat.complete(
        system="be terse",
        user="question text",
        model="gemini-3.5-flash-lite",
        temperature=0.0,
        seed=42,
    )

    assert result.text == "the answer"
    assert result.prompt_tokens == 321
    assert result.completion_tokens == 44
    assert result.model == "gemini-3.5-flash-lite"
    assert len(models.generate_calls) == 1
    call = models.generate_calls[0]
    assert call["model"] == "gemini-3.5-flash-lite"
    assert call["contents"] == "question text"
    config = call["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.system_instruction == "be terse"
    assert config.temperature == 0.0
    assert config.seed == 42


def test_complete_seed_none_is_passed_through() -> None:
    models = FakeGeminiModels()
    chat = GeminiChat(models)
    chat.complete(system="s", user="u", model="m", temperature=0.7, seed=None)
    config = models.generate_calls[0]["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.seed is None


def test_complete_empty_response_raises_provider_error() -> None:
    models = FakeGeminiModels()
    models.generate_results = [_empty_chat_response()]
    chat = GeminiChat(models)
    with pytest.raises(ProviderError, match="no text"):
        chat.complete(system="s", user="u", model="m", temperature=0.0, seed=None)


def test_complete_retries_429_once_then_succeeds() -> None:
    models = FakeGeminiModels()
    models.generate_results = [_rate_limited(), _chat_response("recovered")]
    sleeps, sleep = _sleep_recorder()
    chat = GeminiChat(models, sleep=sleep)

    result = chat.complete(system="s", user="u", model="m", temperature=0.0, seed=None)

    assert result.text == "recovered"
    assert len(models.generate_calls) == 2
    assert sleeps == [1.0]


def test_complete_raises_quota_exhausted_after_max_retries() -> None:
    models = FakeGeminiModels()
    models.generate_results = [_rate_limited()] * 10
    sleeps, sleep = _sleep_recorder()
    chat = GeminiChat(models, sleep=sleep)

    with pytest.raises(ProviderError, match="rate limited") as excinfo:
        chat.complete(system="s", user="u", model="m", temperature=0.0, seed=None)

    assert excinfo.value.code == "quota_exhausted"
    assert len(models.generate_calls) == 6  # 1 initial + 5 retries
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_complete_does_not_retry_non_429_errors() -> None:
    models = FakeGeminiModels()
    models.generate_results = [_server_error()]
    sleeps, sleep = _sleep_recorder()
    chat = GeminiChat(models, sleep=sleep)

    with pytest.raises(ProviderError) as excinfo:
        chat.complete(system="s", user="u", model="m", temperature=0.0, seed=None)

    assert excinfo.value.code == "provider_unavailable"
    assert len(models.generate_calls) == 1
    assert sleeps == []


# --- embeddings (§7.2 binding: batch ≤100, task types, client-side normalization) --


def test_embed_documents_batches_at_most_100_per_request() -> None:
    models = FakeGeminiModels()
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")

    vectors = embeddings.embed_documents([f"text {i}" for i in range(250)])

    assert len(models.embed_calls) == 3
    batch_sizes = [len(call["contents"]) for call in models.embed_calls]  # type: ignore[arg-type]
    assert batch_sizes == [100, 100, 50]
    assert len(vectors) == 250


def test_embed_documents_uses_retrieval_document_task_and_768_dimensions() -> None:
    models = FakeGeminiModels()
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")
    embeddings.embed_documents(["chunk body"])

    config = models.embed_calls[0]["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type == GEMINI_TASK_DOCUMENT == "RETRIEVAL_DOCUMENT"
    assert config.output_dimensionality == EMBEDDING_DIMENSIONS == 768
    assert models.embed_calls[0]["model"] == "gemini-embedding-001"


def test_embed_documents_output_is_l2_normalized() -> None:
    models = FakeGeminiModels()
    models.embed_results = [_embed_response([[3.0, 4.0] + [0.0] * 766])]
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")

    (vector,) = embeddings.embed_documents(["chunk body"])

    assert vector[0] == pytest.approx(0.6)
    assert vector[1] == pytest.approx(0.8)
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_embed_documents_empty_input_makes_no_calls() -> None:
    models = FakeGeminiModels()
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")
    assert embeddings.embed_documents([]) == []
    assert models.embed_calls == []


def test_embed_query_uses_retrieval_query_task_type() -> None:
    models = FakeGeminiModels()
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")

    vector = embeddings.embed_query("how do retries work?")

    config = models.embed_calls[0]["config"]
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type == GEMINI_TASK_QUERY == "RETRIEVAL_QUERY"
    assert models.embed_calls[0]["contents"] == ["how do retries work?"]
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_embed_wrong_dimensionality_raises_provider_error() -> None:
    models = FakeGeminiModels()
    models.embed_results = [_embed_response([[2.0] * 512])]
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")
    with pytest.raises(ProviderError, match="512 dims, expected 768"):
        embeddings.embed_query("q")


def test_embed_vector_count_mismatch_raises_provider_error() -> None:
    models = FakeGeminiModels()
    models.embed_results = [_embed_response([[2.0] * EMBEDDING_DIMENSIONS])]
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001")
    with pytest.raises(ProviderError, match="1 vectors for 2 inputs"):
        embeddings.embed_documents(["a", "b"])


def test_embed_retries_429_with_backoff_then_succeeds() -> None:
    models = FakeGeminiModels()
    models.embed_results = [_rate_limited(), _rate_limited(), _embed_response([[2.0] * 768])]
    sleeps, sleep = _sleep_recorder()
    embeddings = GeminiEmbeddings(models, model="gemini-embedding-001", sleep=sleep)

    vector = embeddings.embed_query("q")

    assert len(models.embed_calls) == 3
    assert sleeps == [1.0, 2.0]
    assert math.isclose(sum(v * v for v in vector), 1.0)
