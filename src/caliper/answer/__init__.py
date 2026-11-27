"""Answer + quotes + confidence (§7.7): question → cited answer or refusal.

`answer_question(question, conn, cfg, *, run_id=None, chat=None,
embeddings=None) -> AnswerResult` — the positional §9.3-style contract
unchanged; keyword-only `chat`/`embeddings` seams accept any
`ChatBackend`/`EmbeddingBackend` (tests inject `FakeChat`/`FakeEmbeddings`,
§15 — no `patch()` anywhere).

Binding behaviors (§7.7):
- Frozen prompt from `src/caliper/answer/prompt_template.md`, verified by
  SHA-256 against `cfg.answer.prompt_template_sha256` at load; mismatch is a
  `ConfigError` before any provider call. The file is resolved next to this
  module (like `db.py`'s `schema.sql`), so it never depends on the process's
  working directory.
- Pre-model refusal, no chat call: fewer than `cfg.retrieval.final_k`
  passages come back (§9.3 — the caller refuses), or the best similarity is
  below `cfg.answer.refusal_threshold`.
- Chat with `temperature=0`, `seed=cfg.answer.seed`; the passages are
  numbered `[1]..[final_k]` in the prompt and the model cites those labels —
  a label is resolved back to the passage's real `chunk_id` for the trace
  (§8.5).
- Quote verification (`quotes.verify_quotes`); on failure retry once with a
  correction message listing the failed quotes; on second failure drop the
  failed citations, record them `verified=0`, downgrade confidence.
- Persist `requests` + `request_retrievals` + `request_citations` (incl.
  refusals) via `db` and commit — `db`'s accessors never commit, and a trace
  that vanishes with the caller's connection would break FR-07/NFR-01.
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
from caliper.answer.quotes import QuoteVerification, verify_quotes
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

# The frozen template ships next to this module — the same way `db.py` finds
# `schema.sql` and `providers/base.py` finds `prices.json` — so the path does
# not depend on the working directory of the process.
_TEMPLATE_PATH = Path(__file__).with_name("prompt_template.md")

_ANSWER_SCHEMA_KEYS = ("answer", "citations", "refused")

REFUSAL_TEXT = "Not found in the docs."


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

    `answer_type` is `"answer"` or `"refusal"`. `citations` follows the §9.2
    response schema (`chunk_id`, `page_url`, `heading_path`, `quote`,
    `verified`) and lists only the citations that passed verification; the
    dropped ones stay in `request_citations` with `verified=0` (§7.7).
    Token and latency fields mirror `requests` (§8.4); `cost_usd` is the §11.3
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


@dataclass(frozen=True, slots=True)
class _Citation:
    """One model citation resolved against the numbered prompt passages."""

    label: int
    chunk_id: int
    page_url: str
    heading_path: str
    quote: str
    verified: bool


def _load_prompt_template(cfg: CaliperConfig) -> str:
    """Read the frozen template and assert its SHA-256 (§7.7, §11.2)."""
    try:
        template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"frozen prompt template not readable: {_TEMPLATE_PATH}: {exc}") from exc
    actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if actual != cfg.answer.prompt_template_sha256:
        raise ConfigError(
            "answer.prompt_template_sha256 mismatch: template on disk hashes to "
            f"{actual}, config expects {cfg.answer.prompt_template_sha256} — "
            f"{_TEMPLATE_PATH} drifted from the preset (FR-10)"
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
    """Refuse before any chat call: too few passages, or weak best similarity.

    - Fewer than `cfg.retrieval.final_k` passages (§9.3: an index holding < 5
      rows returns fewer passages and the caller refuses; an empty or
      never-ingested index returns none at all).
    - Best similarity below `cfg.answer.refusal_threshold` (vector mode only —
      hybrid_rrf scores are RRF values with a different polarity).

    Both are deterministic, pre-model refusals: `refusal_stage="threshold"`,
    `confidence="low"` (§7.7).
    """
    if len(passages) < cfg.retrieval.final_k:
        return REFUSAL_TEXT
    similarities = [_similarity(p) for p in passages]
    best = max((s for s in similarities if s is not None), default=None)
    if best is None:
        return None
    if best < cfg.answer.refusal_threshold:
        return REFUSAL_TEXT
    return None


def _number_passages(passages: list[Passage]) -> str:
    """Render numbered passages for the prompt body (§7.7).

    The numbers are the citation ids the model must use: passage `[1]` is the
    first passage of the final set, `[2]` the second, and so on.
    """
    return "\n".join(f"[{label}] {p.body}" for label, p in enumerate(passages, start=1))


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
    if "answer" not in payload or not isinstance(payload["answer"], str):
        raise ValueError("model reply missing string 'answer'")
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


def _parse_citations(payload: dict[str, Any]) -> list[tuple[int, str]]:
    """Extract the `(passage label, quote)` pairs from one validated reply."""
    return [(int(c["chunk"]), str(c["quote"])) for c in payload.get("citations", [])]


def _resolve_citations(
    verification: list[QuoteVerification], passages_by_label: dict[int, Passage]
) -> list[_Citation]:
    """Map verified quotes onto the passages they cite (§7.7, §8.5).

    The model cites the passage numbers shown in the prompt (`[1]`, `[2]`, …);
    the trace stores the passage's real `chunk_id` (§8.5). A citation whose
    label matches no passage cannot be stored — its `chunk_id` is unknown — so
    it is dropped with a warning.
    """
    resolved: list[_Citation] = []
    for item in verification:
        passage = passages_by_label.get(item.chunk)
        if passage is None:
            log.warning("citation cites unknown passage %s; dropped", item.chunk)
            continue
        resolved.append(
            _Citation(
                label=item.chunk,
                chunk_id=passage.chunk_id,
                page_url=passage.page_url,
                heading_path=passage.heading_path,
                quote=item.quote,
                verified=item.verified,
            )
        )
    return resolved


def _record_trace(
    conn: sqlite3.Connection,
    *,
    request: RequestRow,
    passages: list[Passage],
    citations: list[_Citation],
) -> int:
    """Persist §8.4 + §8.5 rows for one request (answer or refusal)."""
    request_id = insert_request(conn, request)
    insert_retrievals(
        conn,
        request_id,
        [
            RetrievalRow(
                ordinal=rank,
                chunk_id=p.chunk_id,
                score=p.score,
                stage=p.stage,
            )
            for rank, p in enumerate(passages, start=1)
        ],
    )
    if citations:
        insert_citations(
            conn,
            request_id,
            [
                CitationRow(
                    ordinal=ordinal,
                    chunk_id=c.chunk_id,
                    page_url=c.page_url,
                    heading_path=c.heading_path,
                    quote=c.quote,
                    verified=int(c.verified),
                )
                for ordinal, c in enumerate(citations, start=1)
            ],
        )
    return request_id


def _estimate_embed_tokens(question: str) -> int:
    """Embed-token estimate when the provider exposes none (§11.3 fallback)."""
    return max(1, math.ceil(len(question) / 4))


def _cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    embed_tokens: int,
    chat_model: str,
    embed_model: str,
) -> float:
    """§11.3 cost estimate via prices.json; a missing model entry fails loudly.

    Chat tokens are priced with the chat model and embed tokens with the
    embedding model — §11.3's `embed_tokens·embed` term is the embedding
    model's rate, not the chat model's.
    """
    prices = load_prices()
    chat_cost = estimate_cost(
        Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            embed_tokens=0,
        ),
        chat_model,
        prices.entries,
    )
    embed_cost = estimate_cost(
        Usage(prompt_tokens=0, completion_tokens=0, embed_tokens=embed_tokens),
        embed_model,
        prices.entries,
    )
    return chat_cost + embed_cost


def _payload_json(
    system_prompt: str,
    user_prompt: str,
    reply_text: str | None = None,
    attempts: list[dict[str, str]] | None = None,
) -> str:
    """Serialize the exact request payload for NFR-01 (§8.4).

    `user_prompt`/`reply_text` are the effective pair — the request that
    produced the recorded answer. `attempts` keeps every chat call made (the
    schema retry, the quote retry) so the correction prompts stay in the
    trace instead of being lost.
    """
    payload: dict[str, Any] = {"system": system_prompt, "user": user_prompt}
    if reply_text is not None:
        payload["reply"] = reply_text
    payload["attempts"] = attempts or []
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

    The trace is committed before returning: `db`'s accessors never commit and
    the caller owns the transaction boundary, but a request row that vanishes
    when the caller closes the connection would break FR-07/NFR-01.
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

    # --- Pre-model refusal (§7.7 mechanism 2): no chat call ----------------
    refusal_text = _premol_refusal(passages, cfg)
    if refusal_text is not None:
        elapsed_ms = int((time.time() - started_at) * 1000)
        embed_tokens = _estimate_embed_tokens(question)
        cost = _cost_usd(0, 0, embed_tokens, cfg.answer.model, cfg.embeddings.model)
        system_prompt = _build_system_prompt(prompt_template, passages)
        payload_json = _payload_json(system_prompt, question)
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
        request_id = _record_trace(conn, request=request, passages=passages, citations=[])
        conn.commit()
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

    # --- Chat --------------------------------------------------------------
    system_prompt = _build_system_prompt(prompt_template, passages)
    passages_by_label = {label: passage for label, passage in enumerate(passages, start=1)}
    bodies = {label: passage.body for label, passage in passages_by_label.items()}

    attempts: list[dict[str, str]] = []
    prompt_tokens = 0
    completion_tokens = 0

    def _complete(user_prompt: str) -> ChatResult:
        """One chat call; tokens accumulate over every call (FR-07)."""
        nonlocal prompt_tokens, completion_tokens
        result = chat_backend.complete(
            system=system_prompt,
            user=user_prompt,
            model=cfg.answer.model,
            temperature=cfg.answer.temperature,
            seed=cfg.answer.seed,
        )
        attempts.append({"user": user_prompt, "reply": result.text})
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        return result

    # First reply, with one schema retry (§7.7: strict JSON).
    user_prompt = question
    reply_text: str | None = None
    reply_payload: dict[str, Any] | None = None
    for attempt in range(2):
        result = _complete(user_prompt)
        try:
            payload = _extract_json(result.text)
            _validate_reply(payload)
        except (ValueError, KeyError, TypeError) as exc:
            if attempt == 0:
                user_prompt = (
                    f"{question}\n\nThe previous reply could not be parsed as "
                    f"the required schema: {exc}."
                )
                log.debug("answer retry (schema): %s", exc)
                continue
            raise AnswerError(f"model reply failed schema on retry: {exc}") from exc
        reply_text = result.text
        reply_payload = payload
        break

    if reply_text is None:
        raise AnswerError("model produced no parseable reply")
    if reply_payload is None:
        raise AnswerError("model produced no parseable reply")

    # --- Parse + quote verification + one correction retry (§7.7) ----------
    refused = bool(reply_payload["refused"])
    effective_citations: list[_Citation] = []
    dropped = False

    if refused:
        answer_text = REFUSAL_TEXT
    else:
        first_verification = verify_quotes(_parse_citations(reply_payload), bodies)
        failed = [v for v in first_verification if not v.verified]
        if not failed:
            effective_citations = _resolve_citations(first_verification, passages_by_label)
            answer_text = str(reply_payload["answer"])
        else:
            # One correction retry listing the failed quotes (§7.7).
            correction = (
                "The following citations failed quote verification — each quote "
                "must be an exact substring (whitespace-normalized, emphasis "
                "stripped) of its passage's body: "
                + "; ".join(f"chunk {v.chunk}: {v.quote!r}" for v in failed)
                + ". Re-answer with only verified citations, or refused: true."
            )
            retry_user = f"{question}\n\n{correction}"
            retry_result = _complete(retry_user)
            try:
                retry_payload = _extract_json(retry_result.text)
                _validate_reply(retry_payload)
            except (ValueError, KeyError, TypeError) as exc:
                log.debug("answer retry (quotes) failed to parse: %s", exc)
                retry_payload = None
            if retry_payload is not None and retry_payload["refused"]:
                # The model refused on the correction — a grounded refusal (§7.7).
                refused = True
                answer_text = REFUSAL_TEXT
                effective_citations = []
                user_prompt = retry_user
                reply_text = retry_result.text
                reply_payload = retry_payload
            elif retry_payload is not None:
                # The retry's citations are the effective set; the first
                # attempt's failed ones count as dropped (§7.7).
                retry_verification = verify_quotes(_parse_citations(retry_payload), bodies)
                effective_citations = _resolve_citations(retry_verification, passages_by_label)
                dropped = True
                answer_text = str(retry_payload["answer"])
                user_prompt = retry_user
                reply_text = retry_result.text
                reply_payload = retry_payload
            else:
                # The correction reply was unusable: keep the first reply's
                # answer, drop the quotes that failed verification (§7.7).
                effective_citations = _resolve_citations(first_verification, passages_by_label)
                dropped = True
                answer_text = str(reply_payload["answer"])

    # --- Confidence + answer text ------------------------------------------
    best_similarity = max(
        (s for s in (_similarity(p) for p in passages) if s is not None),
        default=None,
    )
    if refused:
        confidence = "high"  # model refusal is not threshold-driven (§7.7)
        refusal_stage = "model"
        answer_text = REFUSAL_TEXT
        effective_citations = []  # refusal rows write no citations (§8.5)
    else:
        refusal_stage = None
        verified = [c for c in effective_citations if c.verified]
        confidence = confidence_for(
            refused=False,
            refusal_stage=None,
            verified_count=len(verified),
            total_citations=len(effective_citations),
            best_similarity=best_similarity,
            high_confidence_threshold=cfg.answer.high_confidence_threshold,
            dropped=dropped,
        )

    # --- Persist + return --------------------------------------------------
    elapsed_ms = int((time.time() - started_at) * 1000)
    embed_tokens = _estimate_embed_tokens(question)
    cost = _cost_usd(
        prompt_tokens, completion_tokens, embed_tokens, cfg.answer.model, cfg.embeddings.model
    )
    payload_json = _payload_json(system_prompt, user_prompt, reply_text, attempts)

    request = RequestRow(
        question=question,
        fingerprint=fp8(cfg),
        run_id=run_id,
        answer_type="refusal" if refused else "answer",
        answer_text=answer_text,
        confidence=confidence,
        latency_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embed_tokens=embed_tokens,
        cost_usd=cost,
        refusal_stage=refusal_stage,
        payload_json=payload_json,
    )
    request_id = _record_trace(
        conn,
        request=request,
        passages=passages,
        citations=effective_citations,
    )
    conn.commit()

    return AnswerResult(
        request_id=request_id,
        question=question,
        answer_type="refusal" if refused else "answer",
        answer=answer_text,
        citations=[
            {
                "chunk_id": c.chunk_id,
                "page_url": c.page_url,
                "heading_path": c.heading_path,
                "quote": c.quote,
                "verified": 1,
            }
            for c in effective_citations
            if c.verified
        ],
        confidence=confidence,
        latency_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        embed_tokens=embed_tokens,
        cost_usd=cost,
        refusal_stage=refusal_stage,
    )
