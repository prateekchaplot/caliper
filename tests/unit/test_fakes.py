"""Unit tests: tests/fakes.py — the suite's two (and only) mock backends (§15).

Verifies the contract later tasks (T09–T14) rely on: scripted chat responses
including the paraphrased-quote and refusal cases, plus deterministic,
L2-normalized, semantically-correlated hash embeddings.
"""

import json

import pytest
from fakes import (
    EmbedCall,
    FakeChat,
    FakeEmbeddings,
    chat_result,
    cosine,
    fake_embedding,
    paraphrased_quote_result,
    refusal_result,
    scripted_chat,
)

# --- FakeChat ---------------------------------------------------------------------


def test_fake_chat_plays_scripted_responses_in_order() -> None:
    first = chat_result("one")
    second = chat_result("two")
    chat = scripted_chat(first, second)

    assert chat.complete(system="s", user="u", model="m", temperature=0.0, seed=1) is first
    assert chat.complete(system="s", user="u", model="m", temperature=0.0, seed=1) is second


def test_fake_chat_records_every_call() -> None:
    chat = scripted_chat(chat_result("ok"))
    chat.complete(system="sys", user="usr", model="m1", temperature=0.0, seed=42)

    (call,) = chat.calls
    assert call.system == "sys"
    assert call.user == "usr"
    assert call.model == "m1"
    assert call.temperature == 0.0
    assert call.seed == 42


def test_fake_chat_raises_when_script_exhausted() -> None:
    chat = scripted_chat(chat_result("only"))
    chat.complete(system="s", user="u", model="m", temperature=0.0, seed=None)
    with pytest.raises(RuntimeError, match="exhausted"):
        chat.complete(system="s", user="u", model="m", temperature=0.0, seed=None)


def test_fake_chat_can_be_scripted_empty_for_protocol_shaped_use() -> None:
    chat: FakeChat = scripted_chat()
    assert chat.responses == []


def test_paraphrased_quote_case_is_a_paraphrasing_citation() -> None:
    result = paraphrased_quote_result()
    payload = json.loads(result.text)
    assert payload["refused"] is False
    (citation,) = payload["citations"]
    # §15: this quote must NOT be trusted verbatim by downstream verification.
    assert isinstance(citation["quote"], str)
    assert "fixed retry delay" in citation["quote"]


def test_refusal_case_refuses_with_no_citations() -> None:
    result = refusal_result()
    payload = json.loads(result.text)
    assert payload["refused"] is True
    assert payload["citations"] == []


# --- FakeEmbeddings ----------------------------------------------------------------


def test_fake_embedding_is_deterministic_and_unit_norm() -> None:
    first = fake_embedding("How do I configure retries on a timer trigger?")
    second = fake_embedding("How do I configure retries on a timer trigger?")

    assert len(first) == 768
    assert first == second
    assert sum(v * v for v in first) == pytest.approx(1.0)


def test_fake_embedding_related_texts_are_closer_than_unrelated() -> None:
    question = "How do I configure retries on a timer trigger?"
    related = "retry configuration for timer triggers"
    unrelated = "blob storage account properties"

    assert cosine(fake_embedding(question), fake_embedding(related)) > cosine(
        fake_embedding(question), fake_embedding(unrelated)
    )


def test_fake_embedding_empty_text_still_unit_norm() -> None:
    vector = fake_embedding("")
    assert sum(v * v for v in vector) == pytest.approx(1.0)


def test_fake_embeddings_records_batches_with_task_kind() -> None:
    embeddings = FakeEmbeddings()
    vectors = embeddings.embed_documents(["alpha", "beta"])
    query = embeddings.embed_query("gamma")

    assert len(vectors) == 2
    assert vectors[0] == fake_embedding("alpha")
    assert query == fake_embedding("gamma")
    assert embeddings.calls == [
        EmbedCall(texts=("alpha", "beta"), task_type="document"),
        EmbedCall(texts=("gamma",), task_type="query"),
    ]


def test_fake_embeddings_custom_dimensions() -> None:
    embeddings = FakeEmbeddings(dimensions=64)
    (vector,) = embeddings.embed_documents(["text"])
    assert len(vector) == 64
    assert sum(v * v for v in vector) == pytest.approx(1.0)
