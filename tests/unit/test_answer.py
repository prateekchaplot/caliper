"""T11 unit tests: answer + quotes + confidence (§7.7).

Seeds a per-fingerprint index through T03's accessors and exercises only the
public `answer_question` interface with the suite's fakes — fakes are the only
mocks (§15): no network, no patch()/monkeypatch.
"""

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fakes import FakeEmbeddings, chat_result, refusal_result, scripted_chat

from caliper.answer import AnswerError, answer_question
from caliper.config import CaliperConfig, ConfigError, fp8, load_config
from caliper.db import (
    ChunkRow,
    connect,
    ensure_chunks_tables,
    ensure_schema,
    insert_chunk_embeddings,
    insert_chunks,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"

QUESTION = "how do i configure retries on a timer trigger"

BODIES = [
    "how do i configure retries on a timer trigger",  # exact match → similarity 1.0
    "retry configuration for timer triggers",  # shares vocabulary → second
    "azure functions hosting plans and scaling",  # unrelated
    "service bus queue message lock duration",  # unrelated
    "blob storage account properties and tiers",  # unrelated
]


@pytest.fixture()
def conn() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    ensure_schema(connection)
    yield connection
    connection.close()


def _load_cfg(tmp_path: Path, user_toml: str = "") -> CaliperConfig:
    user = tmp_path / "user.toml"
    user.write_text(user_toml, encoding="utf-8")
    return load_config(user, CONFIGS / "v1.toml")


def _seed_index(conn: sqlite3.Connection, cfg: CaliperConfig, bodies: list[str]) -> list[int]:
    suffix = fp8(cfg)
    ensure_chunks_tables(conn, suffix)
    chunk_ids = insert_chunks(
        conn,
        suffix,
        [
            ChunkRow(
                page_url=f"https://learn.microsoft.com/en-us/azure/azure-functions/page-{i}",
                heading_path=f"Docs > Section {i}",
                body=body,
                body_tokens=len(body.split()),
                chunk_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                ordinal=i - 1,
            )
            for i, body in enumerate(bodies, start=1)
        ],
    )
    vectors = FakeEmbeddings().embed_documents(bodies)
    insert_chunk_embeddings(conn, suffix, list(zip(chunk_ids, vectors, strict=True)))
    return chunk_ids


# --- happy path: verified citation, high confidence -------------------------


def test_answer_cited_with_verified_quote_and_high_confidence(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    # Body 0 matches the question exactly → chunk_id 1 ranks first, similarity 1.0.
    chat = scripted_chat(
        chat_result(
            '{"answer": "Configure the retry block on the trigger.", "citations": '
            '[{"chunk": 1, "quote": "configure retries"}], "refused": false}'
        )
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "answer"
    assert result.confidence == "high"
    assert result.citations == [
        {
            "chunk": 1,
            "page_url": "https://learn.microsoft.com/en-us/azure/azure-functions/page-1",
            "heading_path": "Docs > Section 1",
            "quote": "configure retries",
            "verified": 1,
        }
    ]
    assert result.refusal_stage is None
    assert result.request_id is not None
    assert chat.calls  # exactly one chat call
    assert len(chat.calls) == 1


# --- threshold refusal: no chat call ----------------------------------------


def test_threshold_refusal_without_chat_call(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    # Unrelated bodies → best similarity below refusal_threshold (0.35).
    _seed_index(conn, cfg, BODIES[2:])
    chat = scripted_chat()  # must never be called

    result = answer_question(
        "completely unrelated topic xyzzy", conn, cfg, chat=chat, embeddings=FakeEmbeddings()
    )

    assert result.answer_type == "refusal"
    assert result.confidence == "low"
    assert result.refusal_stage == "threshold"
    assert result.citations == []
    assert result.answer == "Not found in the docs."
    assert len(chat.calls) == 0  # quota saved


# --- model refusal: high confidence, not low --------------------------------


def test_model_refusal_is_high_confidence(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(refusal_result())

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "refusal"
    assert result.confidence == "high"  # model refusal is not threshold-driven
    assert result.refusal_stage == "model"
    assert result.citations == []


# --- paraphrased quote → retry → drop → medium confidence -------------------


def test_paraphrased_quote_retried_then_downgraded(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    # First reply paraphrases (not verbatim); second reply fixes the quote.
    chat = scripted_chat(
        chat_result(
            '{"answer": "Set the retry policy via the retry block.", "citations": '
            '[{"chunk": 1, "quote": "you can configure a fixed retry delay"}], "refused": false}'
        ),
        chat_result(
            '{"answer": "Configure the retry block on the trigger.", "citations": '
            '[{"chunk": 1, "quote": "configure retries"}], "refused": false}'
        ),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "answer"
    assert result.confidence == "medium"  # a citation was dropped on retry
    assert result.citations == [
        {
            "chunk": 1,
            "page_url": "https://learn.microsoft.com/en-us/azure/azure-functions/page-1",
            "heading_path": "Docs > Section 1",
            "quote": "configure retries",
            "verified": 1,
        }
    ]
    assert len(chat.calls) == 2


# --- all citations fail → model-stage refusal -------------------------------


def test_all_citations_fail_after_retry_low_confidence_answer(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    # Both replies fail quote verification → answer kept, confidence low (§7.7).
    chat = scripted_chat(
        chat_result(
            '{"answer": "Something.", "citations": ['
            '{"chunk": 1, "quote": "nope nope"}], "refused": false}'
        ),
        chat_result(
            '{"answer": "Still something.", "citations": ['
            '{"chunk": 1, "quote": "also wrong"}], "refused": false}'
        ),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "answer"  # kept per §7.7, not converted to refusal
    assert result.confidence == "low"  # all citations unverified
    assert result.refusal_stage is None
    assert result.citations == []


# --- empty index → refusal (caller edge) ------------------------------------


def test_empty_index_refuses_without_chat_call(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    ensure_chunks_tables(conn, fp8(cfg))  # trio exists, no chunks
    chat = scripted_chat()

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "refusal"
    assert result.confidence == "low"
    assert result.refusal_stage == "threshold"
    assert len(chat.calls) == 0


# --- template hash mismatch → ConfigError before any provider call ----------


def test_template_hash_mismatch_is_config_error(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    drifted = cfg.model_copy(
        update={"answer": cfg.answer.model_copy(update={"prompt_template_sha256": "0" * 64})}
    )
    chat = scripted_chat()

    with pytest.raises(ConfigError):
        answer_question(QUESTION, conn, drifted, chat=chat, embeddings=FakeEmbeddings())
    assert len(chat.calls) == 0


# --- empty question → AnswerError -------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\t\n"])
def test_empty_question_raises(conn: sqlite3.Connection, tmp_path: Path, question: str) -> None:
    cfg = _load_cfg(tmp_path)
    with pytest.raises(AnswerError, match="non-empty"):
        answer_question(question, conn, cfg, chat=scripted_chat(), embeddings=FakeEmbeddings())


# --- run_id is persisted unchanged ------------------------------------------


def test_run_id_persisted_in_trace(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "Configure the retry block.", "citations": '
            '[{"chunk": 1, "quote": "configure retries"}], "refused": false}'
        )
    )
    # requests.run_id references runs(run_id) (§8.4): create a run row first.
    from caliper.db import RunRow, create_run, get_request_trace

    run_id = create_run(
        conn,
        RunRow(
            version_label="v1",
            fingerprint=fp8(cfg),
            golden_hash="x" * 64,
            seed=7,
        ),
    )

    result = answer_question(
        QUESTION, conn, cfg, run_id=run_id, chat=chat, embeddings=FakeEmbeddings()
    )

    assert result.request_id is not None
    trace = get_request_trace(conn, result.request_id)
    assert trace.request.run_id == run_id
