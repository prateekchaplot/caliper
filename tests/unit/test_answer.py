"""T11 unit tests: answer + quotes + confidence (§7.7).

Seeds a per-fingerprint index through T03's accessors and exercises only the
public `answer_question` interface with the suite's fakes — fakes are the only
mocks (§15): no network, no patch()/monkeypatch.
"""

import hashlib
import json
import os
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

FILLERS = [f"filler document number {i} about an unrelated topic" for i in range(30)]


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
            "chunk_id": 1,
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


# --- prompt labels are citation ids, not chunk_ids (§7.7, §8.5) -------------


def test_citation_label_resolves_to_the_real_chunk_id(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """The model cites the prompt's `[n]` labels; the trace stores chunk_ids.

    The seeded index holds 30 filler chunks first, so the passages that answer
    the question carry chunk_ids 31..35 — never the labels 1..5 the model sees.
    """
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, FILLERS)
    chunk_ids = _seed_index(conn, cfg, BODIES)
    assert chunk_ids[0] != 1  # the label and the chunk_id genuinely differ
    chat = scripted_chat(
        chat_result(
            '{"answer": "Configure the retry block on the trigger.", "citations": '
            '[{"chunk": 1, "quote": "configure retries"}], "refused": false}'
        )
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "answer"
    assert result.confidence == "high"  # the quote verified against the passage
    assert result.citations[0]["chunk_id"] == chunk_ids[0]
    assert result.citations[0]["quote"] == "configure retries"
    stored = conn.execute(
        "SELECT chunk_id, verified FROM request_citations ORDER BY ordinal"
    ).fetchall()
    assert [tuple(row) for row in stored] == [(chunk_ids[0], 1)]


def test_prompt_numbers_the_passages_from_one(conn: sqlite3.Connection, tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "Configure the retry block on the trigger.", "citations": '
            '[{"chunk": 2, "quote": "retry configuration"}], "refused": false}'
        )
    )

    answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    system = chat.calls[0].system
    passage_lines = [line for line in system.splitlines() if line.startswith("[")]
    assert [line.split(" ", 1)[0] for line in passage_lines] == ["[1]", "[2]", "[3]", "[4]", "[5]"]
    assert passage_lines[0] == f"[1] {BODIES[0]}"  # the exact match ranks first
    assert "retry configuration" in passage_lines[1]  # the cited passage is [2]
    assert chat.calls[0].user == QUESTION


def test_citation_of_an_unknown_passage_is_dropped(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """A label outside the final set cannot be stored (its chunk_id is unknown)."""
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "Something.", "citations": [{"chunk": 99, "quote": "anything"}], '
            '"refused": false}'
        ),
        chat_result(
            '{"answer": "Something.", "citations": [{"chunk": 99, "quote": "anything"}], '
            '"refused": false}'
        ),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.citations == []
    assert result.confidence == "low"
    assert conn.execute("SELECT COUNT(*) FROM request_citations").fetchone()[0] == 0


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


# --- a partial index refuses (§9.3: the caller refuses) ----------------------


def test_index_with_fewer_than_final_k_rows_refuses_without_chat_call(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES[:3])  # 3 rows < final_k (5)
    chat = scripted_chat()

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "refusal"
    assert result.refusal_stage == "threshold"
    assert result.confidence == "low"
    assert len(chat.calls) == 0


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


def test_model_refusal_on_the_quote_retry_is_a_refusal(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """The correction reply may refuse; §7.7's refusal mechanism still applies."""
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "A.", "citations": [{"chunk": 1, "quote": "not in body"}], '
            '"refused": false}'
        ),
        refusal_result(),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer_type == "refusal"
    assert result.refusal_stage == "model"
    assert result.citations == []
    assert conn.execute("SELECT COUNT(*) FROM request_citations").fetchone()[0] == 0


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
            "chunk_id": 1,
            "page_url": "https://learn.microsoft.com/en-us/azure/azure-functions/page-1",
            "heading_path": "Docs > Section 1",
            "quote": "configure retries",
            "verified": 1,
        }
    ]
    assert len(chat.calls) == 2


def test_dropped_citation_is_recorded_with_verified_zero(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    """§7.7: the second failure records `verified=0` in `request_citations`."""
    cfg = _load_cfg(tmp_path)
    chunk_ids = _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "Something.", "citations": '
            '[{"chunk": 1, "quote": "nope nope"}], "refused": false}'
        ),
        chat_result(
            '{"answer": "Still something.", "citations": '
            '[{"chunk": 1, "quote": "also wrong"}], "refused": false}'
        ),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.citations == []  # dropped from the answer
    stored = conn.execute(
        "SELECT chunk_id, quote, verified FROM request_citations ORDER BY ordinal"
    ).fetchall()
    assert [tuple(row) for row in stored] == [(chunk_ids[0], "also wrong", 0)]


def test_answer_text_comes_from_the_final_reply(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """The answer and its citations must come from the same (final) reply."""
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "first answer", "citations": '
            '[{"chunk": 1, "quote": "not in body"}], "refused": false}'
        ),
        chat_result(
            '{"answer": "second answer", "citations": '
            '[{"chunk": 1, "quote": "configure retries"}], "refused": false}'
        ),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert result.answer == "second answer"
    assert result.citations[0]["quote"] == "configure retries"


# --- all citations fail → low-confidence answer ------------------------------


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


# --- the trace records what was actually sent (NFR-01) -----------------------


def test_payload_json_records_the_correction_prompt_and_the_final_reply(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    second = (
        '{"answer": "B.", "citations": [{"chunk": 1, "quote": "configure retries"}], '
        '"refused": false}'
    )
    chat = scripted_chat(
        chat_result(
            '{"answer": "A.", "citations": [{"chunk": 1, "quote": "not in body"}], '
            '"refused": false}'
        ),
        chat_result(second),
    )

    answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    raw = conn.execute("SELECT payload_json FROM requests").fetchone()[0]
    payload = json.loads(raw)
    assert payload["user"] != QUESTION  # the correction prompt is the effective one
    assert "failed quote verification" in payload["user"]
    assert payload["reply"] == second
    assert len(payload["attempts"]) == 2
    assert payload["attempts"][0]["user"] == QUESTION
    assert payload["attempts"][1]["user"] == payload["user"]


# --- token + cost accounting (§8.4, §11.3) ----------------------------------


def test_tokens_count_every_chat_call(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Two chat calls (schema retry + quote retry) are both billed."""
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "A.", "citations": [{"chunk": 1, "quote": "not in body"}], '
            '"refused": false}'
        ),
        chat_result(
            '{"answer": "B.", "citations": [{"chunk": 1, "quote": "configure retries"}], '
            '"refused": false}'
        ),
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    assert len(chat.calls) == 2
    assert result.prompt_tokens == 240  # 2 × 120
    assert result.completion_tokens == 80  # 2 × 40


def test_cost_prices_embed_tokens_with_the_embedding_model(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "A.", "citations": [{"chunk": 1, "quote": "configure retries"}], '
            '"refused": false}'
        )
    )

    result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())

    # chat: 120 × $0.10/M + 40 × $0.40/M; embed: ceil(47/4) = 12 × $0.15/M.
    assert result.embed_tokens == 12
    assert result.cost_usd == pytest.approx((120 * 0.1 + 40 * 0.4 + 12 * 0.15) / 1e6)


# --- the trace is durable (§8.4, FR-07) -------------------------------------


def test_trace_survives_the_connection_closing(tmp_path: Path) -> None:
    cfg = _load_cfg(tmp_path)
    db_path = tmp_path / "caliper.db"
    connection = connect(db_path)
    ensure_schema(connection)
    _seed_index(connection, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "A.", "citations": [{"chunk": 1, "quote": "configure retries"}], '
            '"refused": false}'
        )
    )

    result = answer_question(QUESTION, connection, cfg, chat=chat, embeddings=FakeEmbeddings())
    connection.close()

    reopened = connect(db_path)
    try:
        assert reopened.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
        assert reopened.execute("SELECT COUNT(*) FROM request_citations").fetchone()[0] == 1
    finally:
        reopened.close()
    assert result.request_id is not None


# --- the frozen template is found regardless of the working directory -------


def test_template_is_found_outside_the_repository_root(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    cfg = _load_cfg(tmp_path)
    _seed_index(conn, cfg, BODIES)
    chat = scripted_chat(
        chat_result(
            '{"answer": "A.", "citations": [{"chunk": 1, "quote": "configure retries"}], '
            '"refused": false}'
        )
    )
    previous = Path.cwd()
    os.chdir(tmp_path)
    try:
        result = answer_question(QUESTION, conn, cfg, chat=chat, embeddings=FakeEmbeddings())
    finally:
        os.chdir(previous)

    assert result.answer_type == "answer"


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
