"""Fake backends for offline tests (§15) — the ONLY mocks in this suite.

Two implementations, both stable public seams for later tasks (T09–T14):

- `FakeChat`: a ChatBackend with a scripted response list. Callers push the
  responses they need, in order; every call is recorded. Includes a
  paraphrased-quote case and a refusal case (§15) shaped like the §7.7 reply
  schema, so answer/quote-verification tests can script them directly.
- `FakeEmbeddings`: an EmbeddingBackend with deterministic hash-based
  embeddings (768-dim by default). Vectors are derived per-token (a small
  hashing-trick projection), so texts sharing vocabulary get correlated
  vectors — retrieval tests can rely on relatedness, not just determinism.

No network, no patch(), no knowledge of any component's internals.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from caliper.providers.base import EMBEDDING_DIMENSIONS, ChatResult, l2_normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# --- FakeChat -------------------------------------------------------------------


@dataclass(frozen=True)
class CompleteCall:
    """One recorded complete() call."""

    system: str
    user: str
    model: str
    temperature: float
    seed: int | None


@dataclass
class FakeChat:
    """Scripted ChatBackend: pops one response per complete() call, in order."""

    responses: list[ChatResult]
    calls: list[CompleteCall] = field(default_factory=list)

    def complete(
        self, *, system: str, user: str, model: str, temperature: float, seed: int | None
    ) -> ChatResult:
        self.calls.append(
            CompleteCall(system=system, user=user, model=model, temperature=temperature, seed=seed)
        )
        if not self.responses:
            raise RuntimeError("FakeChat: scripted responses exhausted")
        return self.responses.pop(0)


def chat_result(
    text: str,
    *,
    model: str = "fake-chat-model",
    prompt_tokens: int = 120,
    completion_tokens: int = 40,
) -> ChatResult:
    """Build a ChatResult with plausible default token counts."""
    return ChatResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


def scripted_chat(*responses: ChatResult) -> FakeChat:
    """A FakeChat playing the given responses in order."""
    return FakeChat(responses=list(responses))


def paraphrased_quote_result(
    *, model: str = "fake-chat-model", prompt_tokens: int = 120, completion_tokens: int = 40
) -> ChatResult:
    """Scripted answer whose citation quote PARAPHRASES the passage (§15).

    Downstream quote verification (T11) must retry/drop this one — the quote
    is not a verbatim substring of any chunk body.
    """
    text = json.dumps(
        {
            "answer": "Set the retry policy on a timer trigger via the retry block.",
            "citations": [{"chunk": 1, "quote": "you can configure a fixed retry delay"}],
            "refused": False,
        }
    )
    return chat_result(
        text, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


def refusal_result(
    *, model: str = "fake-chat-model", prompt_tokens: int = 120, completion_tokens: int = 15
) -> ChatResult:
    """Scripted refusal per §7.7 mechanism (1): `refused: true`, no citations."""
    text = json.dumps(
        {
            "answer": "Not found in the provided passages.",
            "citations": [],
            "refused": True,
        }
    )
    return chat_result(
        text, model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )


# --- FakeEmbeddings --------------------------------------------------------------


@dataclass(frozen=True)
class EmbedCall:
    """One recorded embed call: the input texts and the task kind."""

    texts: tuple[str, ...]
    task_type: str  # "document" (embed_documents) or "query" (embed_query)


def _token_positions(token: str, dimensions: int) -> list[tuple[int, float]]:
    """Deterministic (index, sign) projections for one token (hashing trick)."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    positions: list[tuple[int, float]] = []
    for half in (digest[:8], digest[8:16]):
        value = int.from_bytes(half, "big")
        index = value % dimensions
        sign = 1.0 if (value >> 63) & 1 else -1.0
        positions.append((index, sign))
    return positions


def fake_embedding(text: str, dimensions: int = EMBEDDING_DIMENSIONS) -> list[float]:
    """Deterministic 768-dim embedding of `text`, L2-normalized.

    Tokens sharing vocabulary land on the same coordinates, so texts about
    the same topic have cosine > 0 and unrelated texts ≈ 0. Same text always
    yields the same vector (stable across runs and processes).
    """
    accumulated = [0.0] * dimensions
    tokens = _TOKEN_RE.findall(text.lower()) or ["<empty>"]
    for token in tokens:
        for index, sign in _token_positions(token, dimensions):
            accumulated[index] += sign
    return l2_normalize(accumulated)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (unit test helper)."""
    numerator = sum(x * y for x, y in zip(a, b, strict=True))
    return numerator / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@dataclass
class FakeEmbeddings:
    """Deterministic hash-based EmbeddingBackend (§15)."""

    dimensions: int = EMBEDDING_DIMENSIONS
    calls: list[EmbedCall] = field(default_factory=list)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(EmbedCall(texts=tuple(texts), task_type="document"))
        return [fake_embedding(text, self.dimensions) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(EmbedCall(texts=(text,), task_type="query"))
        return fake_embedding(text, self.dimensions)
