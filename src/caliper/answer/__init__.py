"""Answer + quotes + confidence (§7.7): question → cited answer or refusal.

`answer_question(question, conn, cfg, *, run_id=None, chat=None,
embeddings=None) -> AnswerResult` — the positional §9.3-style contract
unchanged; keyword-only `chat`/`embeddings` seams accept any
`ChatBackend`/`EmbeddingBackend` (tests inject `FakeChat`/`FakeEmbeddings`,
§15 — no `patch()` anywhere).

Binding behaviors (§7.7):
- Frozen prompt from `src/caliper/answer/prompt_template.md`, verified by
  SHA-256 against `cfg.answer.prompt_template_sha256` at load; mismatch is a
  `ConfigError` before any provider call.
- Pre-model threshold refusal: best similarity below
  `cfg.answer.refusal_threshold` → refusal without a chat call.
- Chat with `temperature=0`, `seed=cfg.answer.seed`; numbered passages passed
  as `{chunk: <n>, body}` — numbers are the citation IDs.
- Quote verification (`quotes.verify_quotes`); on failure retry once with a
  correction message listing the failed quotes; on second failure drop failed
  citations, log them `verified=0`, downgrade confidence.
- Persist `requests` + `request_retrievals` + `request_citations` (incl.
  refusals) via `db`; return `AnswerResult` with §9.2 fields plus token /
  latency / cost internals.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from caliper.answer.confidence import confidence_for
from caliper.answer.quotes import verify_quotes
from caliper.config import CaliperConfig, ConfigError, fp8
from caliper.db import (
    CitationRow,
    RequestRow,
    RetrievalRow,
    insert_citations,
    insert_request,
    insert_retrievals,
)
from caliper.providers import ChatBackend, ChatResult, EmbeddingBackend, get_chat, get_embeddings
from caliper.providers.base import Usage, estimate_cost, load_prices
from caliper.retrieval import Passage, RetrievalError, retrieve

log = logging.getLogger("caliper.answer")

PromptPath = "src/caliper/answer/prompt_template.md"

_ANSWER_SCHEMA_KEYS = ("answer", "citations", "refused")


class AnswerError(Exception):
    """Answer path failed hard on its inputs or the index.

    Raised for caller-contract violations: empty question, unimplemented mode,
    non-768-dim embedding, template hash mismatch, invalid config. Mapped to
    the §9 error envelope at the CLI/web edges (later tasks). Dependency
    failures keep their own typed exceptions (`DbError`, `ProviderError`,
    `RetrievalError`, `ConfigError`) and propagate unchanged.
    """


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """One answered question (§7.7 public result, §8.4 inputs).

    `answer_type` is `"answer"` or `"refusal"`. `citations` is empty for
    refusals and for answers whose citations all failed verification. Token
    and latency fields mirror `requests` (§8.4); `cost_usd` is the §11.3
    estimate (chat + embed).
    """

    request_id: int | None
    question: str
    answer_type: str
    answer: str
    citations: list[dict[str, Any]]
    confidence: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    embed_tokens: int
    cost_usd: float
    refusal_stage: str | None


def _load_prompt_template(cfg: CaliperConfig) -> str:
    """Read the frozen template and assert its SHA-256 (§7.7, §11.2)."""
    template_path = Path(PromptPath)
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"frozen prompt template not readable: {PromptPath}: {exc}") from exc
    actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if actual != cfg.answer.prompt_template_sha256:
        raise ConfigError(
            "answer.prompt_template_sha256 mismatch: template on disk hashes to "
            f"{actual}, config expects {cfg.answer.prompt_template_sha256} — "
            f"{PromptPath} drifted from the preset (FR-10)"
        )
    return template


def _similarity(passage: Passage) -> float | None:
    """Cosine *similarity* from a passage's score (§7.7 threshold reads similarity).

    Vector mode stores cosine distance (lower = more similar); hybrid stores an
    RRF score whose polarity differs (§7.5). Vector mode is the only mode with
    a defined threshold contract today (§9.7 V1/V2), so this reads distance as
    similarity only there; hybrid returns None and the threshold is skipped.
    """
    if passage.stage != "vector":
        return None
    return 1.0 - passage.score


def _premol_refusal(passages: list[Passage], cfg: CaliperConfig) -> str | None:
    """Best-similarity below refusal_threshold → refuse without a chat call (§7.7).

    An empty passage set (never-ingested or empty index, fewer than
    `final_k`) also refuses: there is nothing to answer from (§9.3, §11.5).
    """
    if not passages:
        return "Not found in the docs."
    similarities = [_similarity(p) for p in passages]
    best = max((s for s in similarities if s is not None), default=None)
    if best is None:
        return None
    if best < cfg.answer.refusal_threshold:
        return "Not found in the docs."
    return None


def _number_passages(passages: list[Passage]) -> str:
    """Render numbered passages for the prompt body (§7.7)."""
    return "\n".join(f"[{i + 1}] {p.body}" for i, p in enumerate(passages))


def _build_system_prompt(template: str, passages: list[Passage]) -> str:
    """Fill {{passages}} with numbered passages (§7.7)."""
    return template.replace("{{passages}}", _number_passages(passages))


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first {...} object from `text`; raise ValueError on none."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in model reply")
    return cast(dict[str, Any], json.loads(text[start : end + 1]))


def _validate_reply(payload: dict[str, Any]) -> None:
    """Enforce the strict reply schema keys (§7.7 prompt)."""
    if not isinstance(payload, dict):
        raise ValueError("model reply is not an object")
    unknown = set(payload) - set(_ANSWER_SCHEMA_KEYS)
    if unknown:
        raise ValueError(f"unexpected keys in model reply: {sorted(unknown)}")
    if "refused" not in payload or not isinstance(payload["refused"], bool):
        raise ValueError("model reply missing boolean 'refused'")
    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("'citations' must be a list")
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("citation is not an object")
        if "chunk" not in citation or not isinstance(citation["chunk"], int):
            raise ValueError("citation missing integer 'chunk'")
        if "quote" not in citation or not isinstance(citation["quote"], str):
            raise ValueError("citation missing string 'quote'")


def _parse_chat_reply(text: str) -> tuple[bool, list[tuple[int, str]]]:
    """Parse `{answer, citations, refused}` from one chat reply (§7.7)."""
    payload = _extract_json(text)
    _validate_reply(payload)
    citations = [(int(c["chunk"]), str(c["quote"])) for c in payload.get("citations", [])]
    return bool(payload.get("refused", False)), citations


def _record_trace(
    conn: sqlite3.Connection,
    *,
    request: RequestRow,
    passages: list[Passage],
    citations: list[dict[str, Any]],
    answer_text: str,
    confidence: str,
    refusal_stage: str | None,
    payload_json: str,
) -> int:
    """Persist §8.4 + §8.5 rows for one request (answer or refusal)."""
    request_id = insert_request(conn, request)
    retrieval_rows = [
        RetrievalRow(
            ordinal=rank,
            chunk_id=p.chunk_id,
            score=p.score,
            stage=p.stage,
        )
        for rank, p in enumerate(passages, start=1)
    ]
    insert_retrievals(conn, request_id, retrieval_rows)
    if citations:
        citation_rows = [
            CitationRow(
                ordinal=ordinal + 1,
                chunk_id=c["chunk_id"],
                page_url=c["page_url"],
                heading_path=c["heading_path"],
                quote=c["quote"],
                verified=int(c["verified"]),
            )
            for ordinal, c in enumerate(citations)
        ]
        insert_citations(conn, request_id, citation_rows)
    return request_id


def _estimate_embed_tokens(question: str) -> int:
    """Embed-token estimate when the provider exposes none (§11.3 fallback)."""
    return max(1, math.ceil(len(question) / 4))


def _cost_usd(prompt_tokens: int, completion_tokens: int, embed_tokens: int, model: str) -> float:
    """§11.3 cost estimate via prices.json; missing entry fails loudly."""
    usage = Usage(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, embed_tokens=embed_tokens
    )
    prices = load_prices()
    return estimate_cost(usage, model, prices.entries)


def _payload_json(
    prompt_template: str,
    passages: list[Passage],
    question: str,
    reply_text: str | None = None,
) -> str:
    """Serialize the exact request payload for NFR-01 (§8.4)."""
    system_prompt = _build_system_prompt(prompt_template, passages)
    payload: dict[str, Any] = {"system": system_prompt, "user": question}
    if reply_text is not None:
        payload["reply"] = reply_text
    return json.dumps(payload, sort_keys=True)


def answer_question(
    question: str,
    conn: sqlite3.Connection,
    cfg: CaliperConfig,
    *,
    run_id: int | None = None,
    chat: ChatBackend | None = None,
    embeddings: EmbeddingBackend | None = None,
) -> AnswerResult:
    """§7.7 end-to-end: retrieve → (threshold check) → chat → verify quotes →
    compute confidence → persist trace → return AnswerResult.

    `chat` / `embeddings` are keyword-only seams: tests inject
    `FakeChat`/`FakeEmbeddings` through them; production defaults to
    `get_chat(cfg)` / `get_embeddings(cfg)` (§15, no `patch()`).
    `run_id` links eval-driven calls to `run_results` (§8.4) and does not
    change behavior (FR-10 invariant).
    """
    if not question.strip():
        raise AnswerError("question must be a non-empty string")
    started_at = time.time()
    prompt_template = _load_prompt_template(cfg)

    chat_backend = chat if chat is not None else get_chat(cfg)
    passage_embeddings = embeddings if embeddings is not None else get_embeddings(cfg)

    try:
        passages = retrieve(question, conn, cfg, embeddings=passage_embeddings)
    except RetrievalError as exc:
        raise AnswerError(str(exc)) from exc

    # --- Pre-model threshold refusal (§7.7 mechanism 2) -------------------
    refusal_text = _premol_refusal(passages, cfg)
    if refusal_text is not None:
        elapsed_ms = int((time.time() - started_at) * 1000)
        embed_tokens = _estimate_embed_tokens(question)
        cost = _cost_usd(0, 0, embed_tokens, cfg.answer.model)
        payload_json = _payload_json(prompt_template, passages, question)
        request = RequestRow(
            question=question,
            fingerprint=fp8(cfg),
            run_id=run_id,
            answer_type="refusal",
            answer_text=refusal_text,
            confidence="low",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            embed_tokens=embed_tokens,
            cost_usd=cost,
            refusal_stage="threshold",
            payload_json=payload_json,
        )
        request_id = _record_trace(
            conn,
            request=request,
            passages=passages,
            citations=[],
            answer_text=refusal_text,
            confidence="low",
            refusal_stage="threshold",
            payload_json=payload_json,
        )
        log.info("refusal(threshold): %s", question)
        return AnswerResult(
            request_id=request_id,
            question=question,
            answer_type="refusal",
            answer=refusal_text,
            citations=[],
            confidence="low",
            latency_ms=elapsed_ms,
            prompt_tokens=0,
            completion_tokens=0,
            embed_tokens=embed_tokens,
            cost_usd=cost,
            refusal_stage="threshold",
        )

    # --- Chat (initial + one schema retry) ---------------------------------
    system_prompt = _build_system_prompt(prompt_template, passages)
    chat_result: ChatResult | None = None
    reply_text: str | None = None
    user_prompt = question

    for attempt in range(2):
        chat_result = chat_backend.complete(
            system=system_prompt,
            user=user_prompt,
            model=cfg.answer.model,
            temperature=cfg.answer.temperature,
            seed=cfg.answer.seed,
        )
        try:
            _validate_reply(_extract_json(chat_result.text))
            reply_text = chat_result.text
            break
        except (ValueError, KeyError, TypeError) as exc:
            if attempt == 0:
                user_prompt = (
                    f"{question}\n\nThe previous reply could not be parsed as "
                    f"the required schema: {exc}."
                )
                log.debug("answer retry (schema): %s", exc)
                continue
            raise AnswerError(f"model reply failed schema on retry: {exc}") from exc

    if reply_text is None:
        raise AnswerError("model produced no parseable reply")

    payload_json = _payload_json(prompt_template, passages, question, reply_text)

    # --- Parse + quote verification + one correction retry -----------------
    refused, raw_citations = _parse_chat_reply(reply_text)

    bodies = {p.chunk_id: p.body for p in passages}
    verification = verify_quotes(raw_citations, bodies)
    had_failed = bool([v for v in verification if not v.verified])

    if had_failed:
        correction = (
            "The following citations failed quote verification — each quote "
            "must be an exact substring (whitespace-normalized, emphasis "
            "stripped) of its passage's body: "
            + "; ".join(f"chunk {v.chunk}: {v.quote!r}" for v in verification if not v.verified)
            + ". Re-answer with only verified citations, or refused: true."
        )
        retry_result = chat_backend.complete(
            system=system_prompt,
            user=f"{question}\n\n{correction}",
            model=cfg.answer.model,
            temperature=cfg.answer.temperature,
            seed=cfg.answer.seed,
        )
        try:
            _validate_reply(_extract_json(retry_result.text))
            retry_refused, retry_citations = _parse_chat_reply(retry_result.text)
        except (ValueError, KeyError, TypeError) as exc:
            log.debug("answer retry (quotes) failed to parse: %s", exc)
            retry_refused, retry_citations = refused, []
        # Re-verify the retry's citations (§7.7: a second failure drops them).
        verification = verify_quotes(retry_citations, bodies)
        retry_failed = [v for v in verification if not v.verified]
        had_failed = had_failed or bool(retry_failed)
        if retry_failed:
            # Second failure: drop the failed citations; keep any that verify.
            raw_citations = [(v.chunk, v.quote) for v in verification if v.verified]
            if raw_citations:
                log.warning("dropped unverified citations after retry")
            else:
                log.info("all citations dropped after retry; low-confidence answer")
        else:
            raw_citations = retry_citations
            refused = retry_refused

    # --- Confidence + answer text ------------------------------------------
    best_similarity = max(
        (s for s in (_similarity(p) for p in passages) if s is not None),
        default=None,
    )
    if refused:
        confidence = "high"  # model refusal is not threshold-driven (§7.7)
        refusal_stage = "model"
        citations_out: list[dict[str, Any]] = []
        answer_text = "Not found in the docs."
    else:
        refusal_stage = None
        verified_count = len(raw_citations)
        confidence = confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=verified_count,
            total_citations=verified_count,
            best_similarity=best_similarity,
            high_confidence_threshold=cfg.answer.high_confidence_threshold,
            dropped=had_failed and verified_count > 0,
        )
        passage_by_id = {p.chunk_id: p for p in passages}
        citations_out = [
            {
                "chunk_id": chunk,
                "page_url": passage_by_id[chunk].page_url,
                "heading_path": passage_by_id[chunk].heading_path,
                "quote": quote,
                "verified": 1,
            }
            for chunk, quote in raw_citations
            if chunk in passage_by_id
        ]
        answer_text = _extract_json(reply_text).get("answer", "")

    # --- Persist + return --------------------------------------------------
    elapsed_ms = int((time.time() - started_at) * 1000)
    usage = chat_result or ChatResult(
        text="", prompt_tokens=0, completion_tokens=0, model=cfg.answer.model
    )
    embed_tokens = _estimate_embed_tokens(question)
    cost = _cost_usd(usage.prompt_tokens, usage.completion_tokens, embed_tokens, cfg.answer.model)

    request = RequestRow(
        question=question,
        fingerprint=fp8(cfg),
        run_id=run_id,
        answer_type="refusal" if refused else "answer",
        answer_text=answer_text,
        confidence=confidence,
        latency_ms=elapsed_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        embed_tokens=embed_tokens,
        cost_usd=cost,
        refusal_stage=refusal_stage,
        payload_json=payload_json,
    )
    request_id = _record_trace(
        conn,
        request=request,
        passages=passages,
        citations=citations_out,
        answer_text=answer_text,
        confidence=confidence,
        refusal_stage=refusal_stage,
        payload_json=payload_json,
    )

    return AnswerResult(
        request_id=request_id,
        question=question,
        answer_type="refusal" if refused else "answer",
        answer=answer_text,
        citations=[
            {
                "chunk": c["chunk_id"],
                "page_url": c["page_url"],
                "heading_path": c["heading_path"],
                "quote": c["quote"],
                "verified": c["verified"],
            }
            for c in citations_out
        ],
        confidence=confidence,
        latency_ms=elapsed_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        embed_tokens=embed_tokens,
        cost_usd=cost,
        refusal_stage=refusal_stage,
    )
