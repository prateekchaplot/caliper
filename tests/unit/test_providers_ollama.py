"""Unit tests: providers/ollama.py — chat payload/usage parsing, embed
batching/normalization, 429 backoff, and transport-error mapping.

Network is exercised through httpx's injectable MockTransport (§15) — no real
requests, no patch() on internals.
"""

import json
from collections.abc import Callable

import httpx
import pytest

from caliper.providers import EMBEDDING_DIMENSIONS, ProviderError
from caliper.providers.ollama import (
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OllamaChat,
    OllamaEmbeddings,
)

BASE_URL = "http://ollama.test"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url=BASE_URL)


def _sleep_recorder() -> tuple[list[float], object]:
    sleeps: list[float] = []

    def record(seconds: float) -> None:
        sleeps.append(seconds)

    return sleeps, record


def _chat_handler(
    content: str, *, prompt_eval_count: int = 21, eval_count: int = 7
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": content},
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
            },
            request=request,
        )

    return handler


def _embed_handler() -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"embeddings": [[1.0] * EMBEDDING_DIMENSIONS for _ in body["input"]]},
            request=request,
        )

    return handler


# --- chat (§7.2 complete) ---------------------------------------------------------


def test_complete_posts_chat_payload_and_parses_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_handler("Set the retry block.")(request)

    chat = OllamaChat(_client(handler))
    result = chat.complete(
        system="be terse",
        user="question text",
        model=OLLAMA_CHAT_MODEL,
        temperature=0.0,
        seed=42,
    )

    assert result.text == "Set the retry block."
    assert result.prompt_tokens == 21
    assert result.completion_tokens == 7
    assert result.model == OLLAMA_CHAT_MODEL
    assert seen["path"] == "/api/chat"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["model"] == OLLAMA_CHAT_MODEL
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "question text"},
    ]
    assert body["options"] == {"temperature": 0.0, "seed": 42}


def test_complete_omits_seed_when_none() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return _chat_handler("ok")(request)

    chat = OllamaChat(_client(handler))
    chat.complete(system="s", user="u", model=OLLAMA_CHAT_MODEL, temperature=0.5, seed=None)
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["options"] == {"temperature": 0.5}


def test_complete_missing_content_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {}}, request=request)

    chat = OllamaChat(_client(handler))
    with pytest.raises(ProviderError, match="message.content"):
        chat.complete(system="s", user="u", model=OLLAMA_CHAT_MODEL, temperature=0.0, seed=None)


# --- embeddings (§7.2 binding) -----------------------------------------------------


def test_embed_documents_batches_at_most_100_and_normalizes() -> None:
    requests: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(len(body["input"]))
        assert request.url.path == "/api/embed"
        assert body["model"] == OLLAMA_EMBED_MODEL
        return httpx.Response(
            200,
            json={"embeddings": [[3.0, 4.0] + [0.0] * 766 for _ in body["input"]]},
            request=request,
        )

    embeddings = OllamaEmbeddings(_client(handler))
    vectors = embeddings.embed_documents([f"text {i}" for i in range(250)])

    assert requests == [100, 100, 50]
    assert len(vectors) == 250
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)
    assert sum(v * v for v in vectors[0]) == pytest.approx(1.0)


def test_embed_query_sends_single_text_and_normalizes() -> None:
    embeddings = OllamaEmbeddings(_client(_embed_handler()))
    vector = embeddings.embed_query("how do retries work?")
    assert len(vector) == EMBEDDING_DIMENSIONS
    assert sum(v * v for v in vector) == pytest.approx(1.0)


def test_embed_documents_empty_input_makes_no_requests() -> None:
    count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        count["n"] += 1
        return _embed_handler()(request)

    embeddings = OllamaEmbeddings(_client(handler))
    assert embeddings.embed_documents([]) == []
    assert count["n"] == 0


def test_embed_wrong_dimensionality_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0] * 512]}, request=request)

    embeddings = OllamaEmbeddings(_client(handler))
    with pytest.raises(ProviderError, match="512 dims, expected 768"):
        embeddings.embed_query("q")


# --- backoff + error mapping (§7.2, §10.1) ------------------------------------------


def test_complete_retries_429_with_backoff_then_succeeds() -> None:
    state = {"n": 0}
    sleeps, sleep = _sleep_recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] <= 2:
            return httpx.Response(429, json={"error": "rate limited"}, request=request)
        return _chat_handler("recovered")(request)

    chat = OllamaChat(_client(handler), sleep=sleep)
    result = chat.complete(
        system="s", user="u", model=OLLAMA_CHAT_MODEL, temperature=0.0, seed=None
    )

    assert result.text == "recovered"
    assert state["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_complete_raises_quota_exhausted_after_max_retries() -> None:
    state = {"n": 0}
    sleeps, sleep = _sleep_recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(429, json={"error": "rate limited"}, request=request)

    chat = OllamaChat(_client(handler), sleep=sleep)
    with pytest.raises(ProviderError, match="rate limited") as excinfo:
        chat.complete(system="s", user="u", model=OLLAMA_CHAT_MODEL, temperature=0.0, seed=None)

    assert excinfo.value.code == "quota_exhausted"
    assert state["n"] == 6  # 1 initial + 5 retries
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_transport_error_maps_to_provider_unavailable_without_retry() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        raise httpx.ConnectError("connection refused")

    chat = OllamaChat(_client(handler))
    with pytest.raises(ProviderError, match="connection refused") as excinfo:
        chat.complete(system="s", user="u", model=OLLAMA_CHAT_MODEL, temperature=0.0, seed=None)

    assert excinfo.value.code == "provider_unavailable"
    assert state["n"] == 1


def test_http_500_maps_to_provider_unavailable_without_retry() -> None:
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return httpx.Response(500, json={"error": "boom"}, request=request)

    chat = OllamaChat(_client(handler))
    with pytest.raises(ProviderError, match="500"):
        chat.complete(system="s", user="u", model=OLLAMA_CHAT_MODEL, temperature=0.0, seed=None)

    assert state["n"] == 1


def test_unexpected_json_shape_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": True}, request=request)

    embeddings = OllamaEmbeddings(_client(handler))
    with pytest.raises(ProviderError, match="embeddings"):
        embeddings.embed_query("q")
