"""Configuration loading, merging, and fingerprinting (§7.1).

Single definition of the config model and of the canonical fingerprint used
everywhere (§9.7 contents binding, §11.2 fingerprint scope). This module is a
leaf: it has no knowledge of the DB layout or providers, and performs no I/O
beyond reading TOML files (§7.1 "Must not").

Merge order (§11.2 ambiguity resolution, recorded in docs/TASKS.md Notes):
the version preset is merged first, then the user config is merged over it —
the user config wins on conflicts. This is what makes the A2 fallback path
("flip CALIPER_PROVIDER=ollama") workable without editing committed presets,
and it stays safe for FR-10 because comparability is enforced by the
fingerprint gate (§7.8), not by the merge order.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

# SHA-256 of the empty string: placeholder for the frozen prompt template's
# hash until T11 stores the real template file. Changing this value changes
# the fingerprint, so T11's replacement is fingerprint-visible by construction.
PROMPT_TEMPLATE_SHA256_PLACEHOLDER = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


class ConfigError(ValueError):
    """A config file is missing, unreadable, or invalid (§11.4).

    The message always names the offending key path. Mapped to the
    `config_invalid` envelope code at the CLI/web edges (§9).
    """


class ChunkingConfig(BaseModel):
    """Segmentation settings (§9.7). The 1,900-token cap matches §7.4."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    unit: Literal["heading", "fixed"] = "heading"
    size_tokens: int = Field(default=800, ge=1, le=1900)
    overlap_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _overlap_below_size(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.size_tokens:
            # The ValueError message names the offending key path (§11.4);
            # pydantic prefixes it with the section name in ValidationError.loc.
            raise ValueError("chunking.overlap_tokens must be smaller than chunking.size_tokens")
        return self


class RetrievalConfig(BaseModel):
    """Search/ranking settings (§7.5, §9.7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["vector", "hybrid_rrf"] = "vector"
    vector_top_k: int = Field(default=5, ge=1)
    fts_top_k: int = Field(default=0, ge=0)
    rrf_k: int = Field(default=60, ge=1)
    rerank: Literal["none", "cross-encoder"] = "none"
    final_k: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _mode_consistency(self) -> RetrievalConfig:
        # Stated rule (§11.4): fts_top_k=0 iff mode="vector".
        if self.mode == "vector" and self.fts_top_k != 0:
            raise ValueError("retrieval.fts_top_k must be 0 when mode is 'vector'")
        if self.mode == "hybrid_rrf" and self.fts_top_k == 0:
            raise ValueError("retrieval.fts_top_k must be > 0 when mode is 'hybrid_rrf'")
        # A reranker only makes sense over a fused candidate list (§7.5).
        if self.rerank != "none" and self.mode != "hybrid_rrf":
            raise ValueError("retrieval.rerank other than 'none' requires mode 'hybrid_rrf'")
        return self


class AnswerConfig(BaseModel):
    """Fixed answerer settings (§7.7, §9.7). Identical across V1–V3 (FR-10)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = "gemini-3.5-flash-lite"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = Field(default=42, ge=0)
    # Initial 0.35; re-verified once against V1 eval data, then frozen (§7.7, §9.7).
    refusal_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    prompt_template_sha256: str = PROMPT_TEMPLATE_SHA256_PLACEHOLDER


class EmbeddingsConfig(BaseModel):
    """Embedding model settings (§5, §9.7)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = "gemini-embedding-001"
    dimensions: int = Field(default=768, ge=1)


class CaliperConfig(BaseModel):
    """The full resolved Caliper configuration (§9.7).

    Contains only fingerprint-relevant fields; paths, provider selection, and
    secrets are environment variables (§11.2) and live outside this model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    answer: AnswerConfig = AnswerConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep merge of `override` on top of `base` (override wins)."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _format_validation_error(exc: ValidationError) -> str:
    """Render pydantic errors as 'section.key: message' key paths (§11.4)."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(piece) for piece in err["loc"])
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


def load_config(user_path: Path, preset: Path) -> CaliperConfig:
    """Load the user config and the version preset, resolve to concrete values.

    The preset is loaded first and the user config is merged over it (user
    wins; see module docstring). Any unknown key, wrong type, or cross-field
    violation raises ConfigError naming the offending key path (§11.4) before
    any provider call could be made.
    """
    preset_raw = _read_toml(preset)
    user_raw = _read_toml(user_path)
    merged = _deep_merge(preset_raw, user_raw)
    try:
        return CaliperConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid config: {_format_validation_error(exc)}") from exc


def fingerprint_payload(cfg: CaliperConfig) -> dict[str, Any]:
    """The fingerprinted fields, per §11.2: model IDs, embedding model +
    dimensions, prompt template text hash, temperature, seed, thresholds,
    chunking, retrieval mode and k-values — deliberately excluding paths and
    the API key (neither is representable in this model)."""
    return {
        "chunking": cfg.chunking.model_dump(),
        "retrieval": cfg.retrieval.model_dump(),
        "answer": cfg.answer.model_dump(),
        "embeddings": cfg.embeddings.model_dump(),
    }


def fingerprint(cfg: CaliperConfig) -> str:
    """SHA-256 hex of the canonical JSON of the binding fields (§11.2).

    Stable across runs on identical inputs; changes on any binding-field
    change, which by design starts a new index/eval identity (FR-09).
    """
    canonical = json.dumps(fingerprint_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fp8(cfg: CaliperConfig) -> str:
    """First 8 hex chars of the fingerprint — the per-fingerprint table suffix (§6.2)."""
    return fingerprint(cfg)[:8]
