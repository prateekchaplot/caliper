"""P0 live smoke test (T04, §14.2, §15) — run by a human, NEVER in CI.

Exercises BOTH provider backends live:
  1. one chat call (gemini: cfg.answer.model; ollama: qwen2.5:7b-instruct),
  2. one 768-dim embedding per backend, printed with its L2 norm (≈ 1.0,
     proving the client-side normalization of ADR-003),
  3. the bounded 429 backoff path (no live rate limit needed: the helper runs
     against a scripted twice-429 operation and logs the same WARNING events
     production calls produce).

Reads config and GEMINI_API_KEY exactly like the runtime does — environment
variables (§11.2), optionally populated from a repo-root `.env`. Any failure
exits nonzero with the §9 error envelope on stderr.

Usage:
    uv run python scripts/smoke.py

Setup (docs/SETUP.md lands with T22):
    cp .env.example .env   # then set GEMINI_API_KEY=...
    # for the ollama leg: install Ollama, then
    #   ollama pull qwen2.5:7b-instruct && ollama pull nomic-embed-text
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path

import httpx

from caliper.config import CaliperConfig, ConfigError, fp8, load_config
from caliper.providers.base import (
    Prices,
    ProviderError,
    Usage,
    call_with_backoff,
    estimate_cost,
    gemini_api_key,
    load_prices,
    ollama_base_url,
)
from caliper.providers.gemini import GeminiChat, GeminiEmbeddings
from caliper.providers.ollama import (
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_HTTP_TIMEOUT_S,
    OllamaChat,
    OllamaEmbeddings,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("caliper.smoke")

SMOKE_SYSTEM = "You are a documentation assistant. Answer in one short sentence."
SMOKE_QUESTION = "In one sentence: what is an Azure Functions timer trigger?"
SMOKE_EMBED_TEXT = "How do I configure retries on an Azure Functions timer trigger?"

# §9 error envelope exit codes.
EXIT_CODES: dict[str, int] = {
    "config_invalid": 2,
    "provider_unavailable": 4,
    "quota_exhausted": 5,
}


class BackendPair:
    """One provider's chat + embedding backends plus its pinned chat model."""

    def __init__(
        self,
        name: str,
        chat: GeminiChat | OllamaChat,
        embeddings: GeminiEmbeddings | OllamaEmbeddings,
        chat_model: str,
    ) -> None:
        self.name = name
        self.chat = chat
        self.embeddings = embeddings
        self.chat_model = chat_model


def _load_env_file(path: Path) -> None:
    """Populate os.environ from .env (KEY=VALUE lines) without overriding."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _build_backend(name: str, cfg: CaliperConfig) -> BackendPair:
    """Build one backend pair directly (smoke tests BOTH, ignoring CALIPER_PROVIDER)."""
    if name == "gemini":
        from google import genai

        models = genai.Client(api_key=gemini_api_key()).models
        return BackendPair(
            name,
            GeminiChat(models),
            GeminiEmbeddings(
                models, model=cfg.embeddings.model, dimensions=cfg.embeddings.dimensions
            ),
            chat_model=cfg.answer.model,
        )
    client = httpx.Client(base_url=ollama_base_url(), timeout=httpx.Timeout(OLLAMA_HTTP_TIMEOUT_S))
    return BackendPair(
        name,
        OllamaChat(client),
        OllamaEmbeddings(client, model=OLLAMA_EMBED_MODEL, dimensions=cfg.embeddings.dimensions),
        chat_model=OLLAMA_CHAT_MODEL,
    )


def _run_backend(backend: BackendPair, prices: Prices) -> None:
    """One chat call + one embed per backend; prints norms and cost (§14.2)."""
    result = backend.chat.complete(
        system=SMOKE_SYSTEM,
        user=SMOKE_QUESTION,
        model=backend.chat_model,
        temperature=0.0,
        seed=42,
    )
    print(
        f"[{backend.name}] chat ok: model={result.model} "
        f"tokens(in/out)={result.prompt_tokens}/{result.completion_tokens} "
        f"text={result.text!r}"
    )
    if backend.chat_model in prices.entries:
        cost = estimate_cost(
            Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            ),
            backend.chat_model,
            prices.entries,
        )
        print(f"[{backend.name}] estimated cost (prices as_of {prices.as_of}): ${cost:.6f}")
    else:
        print(f"[{backend.name}] estimated cost: $0.000000 (local model, no price entry)")

    vector = backend.embeddings.embed_query(SMOKE_EMBED_TEXT)
    norm = math.sqrt(sum(v * v for v in vector))
    if abs(norm - 1.0) > 1e-6:
        raise ProviderError(
            f"[{backend.name}] embedding norm {norm:.6f} != 1.0 — client-side L2 "
            "normalization (ADR-003) is broken"
        )
    print(
        f"[{backend.name}] embed ok: dims={len(vector)} norm={norm:.6f} (L2-normalized client-side)"
    )


def _demonstrate_backoff() -> None:
    """Log the same WARNING events a live 429 produces, with recorded sleeps."""
    attempts = {"n": 0}
    slept: list[float] = []

    class _Simulated429(Exception):
        """Stand-in for an HTTP 429 from the provider."""

    def op() -> str:
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise _Simulated429("simulated HTTP 429")
        return "succeeded"

    outcome = call_with_backoff(
        op,
        describe="smoke 429 backoff demo",
        is_rate_limited=lambda exc: isinstance(exc, _Simulated429),
        sleep=slept.append,
    )
    print(f"429 backoff demo: {attempts['n']} attempts, sleeps={slept} -> {outcome}")


def _fail(code: str, message: str) -> int:
    print(json.dumps({"error": {"code": code, "message": message}}), file=sys.stderr)
    return EXIT_CODES[code]


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("CALIPER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    _load_env_file(REPO_ROOT / ".env")
    print("== caliper live smoke (T04) ==")

    try:
        user_config = Path(
            os.environ.get("CALIPER_CONFIG", REPO_ROOT / "configs" / "caliper.sample.toml")
        )
        cfg = load_config(user_config, REPO_ROOT / "configs" / "v1.toml")
    except ConfigError as exc:
        return _fail("config_invalid", str(exc))
    print(f"config fingerprint (fp8): {fp8(cfg)}")

    prices = load_prices()
    print(f"price table as_of: {prices.as_of} (re-verify at P0, §9.8)")

    failures: list[ProviderError] = []
    for name in ("gemini", "ollama"):
        try:
            backend = _build_backend(name, cfg)
        except ConfigError as exc:
            failures.append(ProviderError(str(exc), code="config_invalid"))
            continue
        except ProviderError as exc:
            failures.append(exc)
            continue
        try:
            _run_backend(backend, prices)
        except ProviderError as exc:
            failures.append(exc)

    _demonstrate_backoff()

    if failures:
        first = failures[0]
        for extra in failures[1:]:
            LOGGER.warning("additional backend failure: %s", extra)
        LOGGER.debug("first failure traceback", exc_info=first)
        return _fail(first.code, str(first))
    print("SMOKE OK — chat + 768-dim normalized embed per backend, backoff path logged.")
    print(
        "OQ1 reminder: confirm gemini-3.5-flash-lite free-tier RPD in AI Studio "
        "and record the decision in the decision log."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
