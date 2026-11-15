"""T02 unit tests: config loading/merging and fingerprinting (§7.1, §9.7, §11.2, §11.4)."""

import re
from pathlib import Path
from typing import Any

import pytest

from caliper.config import (
    CaliperConfig,
    ConfigError,
    fingerprint,
    fingerprint_payload,
    fp8,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"

# §9.7 binding contents, keyed by preset (answer/embeddings shared by all three).
_V1_TABLE: dict[str, Any] = {
    "chunking": {"unit": "heading", "size_tokens": 800, "overlap_tokens": 0},
    "retrieval": {
        "mode": "vector",
        "vector_top_k": 5,
        "fts_top_k": 0,
        "rrf_k": 60,
        "rerank": "none",
        "final_k": 5,
    },
}
_V2_TABLE: dict[str, Any] = {
    "chunking": {"unit": "fixed", "size_tokens": 350, "overlap_tokens": 50},
    "retrieval": {
        "mode": "vector",
        "vector_top_k": 5,
        "fts_top_k": 0,
        "rrf_k": 60,
        "rerank": "none",
        "final_k": 5,
    },
}
_V3_TABLE: dict[str, Any] = {
    "chunking": {"unit": "fixed", "size_tokens": 350, "overlap_tokens": 50},
    "retrieval": {
        "mode": "hybrid_rrf",
        "vector_top_k": 20,
        "fts_top_k": 20,
        "rrf_k": 60,
        "rerank": "cross-encoder",
        "final_k": 5,
    },
}
_SHARED_TABLE: dict[str, Any] = {
    "answer": {
        "model": "gemini-3.5-flash-lite",
        "temperature": 0.0,
        "seed": 42,
        "refusal_threshold": 0.35,
        "high_confidence_threshold": 0.60,
    },
    "embeddings": {"model": "gemini-embedding-001", "dimensions": 768},
}
PRESET_TABLES: dict[str, dict[str, Any]] = {"v1": _V1_TABLE, "v2": _V2_TABLE, "v3": _V3_TABLE}


def _write_user(tmp_path: Path, toml_text: str) -> Path:
    path = tmp_path / "user.toml"
    path.write_text(toml_text, encoding="utf-8")
    return path


def _load(tmp_path: Path, preset: str, user_toml: str = "") -> CaliperConfig:
    return load_config(_write_user(tmp_path, user_toml), CONFIGS / f"{preset}.toml")


def _assert_matches_table(cfg: CaliperConfig, table: dict[str, Any]) -> None:
    for section, values in table.items():
        sub = getattr(cfg, section)
        for key, expected in values.items():
            assert getattr(sub, key) == expected, f"{section}.{key}"


# --- §9.7: preset contents binding -------------------------------------------


@pytest.mark.parametrize("preset", ["v1", "v2", "v3"])
def test_preset_matches_section_9_7_table(tmp_path: Path, preset: str) -> None:
    cfg = _load(tmp_path, preset)
    _assert_matches_table(cfg, {**PRESET_TABLES[preset], **_SHARED_TABLE})


def test_presets_share_the_fixed_answerer(tmp_path: Path) -> None:
    # FR-10 invariant: answer settings identical across V1/V2/V3.
    answer_tables = [fingerprint_payload(_load(tmp_path, v))["answer"] for v in ("v1", "v2", "v3")]
    assert answer_tables[0] == answer_tables[1] == answer_tables[2]


def test_sample_config_resolves_to_v1_values(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "caliper.sample.toml", CONFIGS / "v1.toml")
    _assert_matches_table(cfg, {**_V1_TABLE, **_SHARED_TABLE})


# --- merge semantics ---------------------------------------------------------


def test_user_config_merges_over_preset(tmp_path: Path) -> None:
    user = '[answer]\nmodel = "qwen2.5:7b-instruct"\n'
    cfg = _load(tmp_path, "v1", user)
    assert cfg.answer.model == "qwen2.5:7b-instruct"
    # Deep merge: sibling keys from the preset survive the override.
    assert cfg.answer.seed == 42
    _assert_matches_table(cfg, {**_V1_TABLE, "embeddings": _SHARED_TABLE["embeddings"]})


def test_merge_is_deep_not_section_replacing(tmp_path: Path) -> None:
    user = "[retrieval]\nvector_top_k = 20\n"
    cfg = _load(tmp_path, "v1", user)
    assert cfg.retrieval.vector_top_k == 20
    assert cfg.retrieval.mode == "vector"
    assert cfg.retrieval.final_k == 5


# --- fingerprint (§11.2) -----------------------------------------------------


def test_fingerprint_stable_across_reloads(tmp_path: Path) -> None:
    a = _load(tmp_path, "v1")
    b = _load(tmp_path, "v1")
    assert fingerprint(a) == fingerprint(b)
    assert len(fingerprint(a)) == 64


def test_fingerprint_differs_across_versions(tmp_path: Path) -> None:
    fps = {v: fingerprint(_load(tmp_path, v)) for v in ("v1", "v2", "v3")}
    assert len(set(fps.values())) == 3


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("chunking", "unit", "fixed"),
        ("chunking", "size_tokens", 400),
        ("chunking", "overlap_tokens", 10),
        ("retrieval", "mode", "hybrid_rrf"),
        ("retrieval", "vector_top_k", 20),
        ("retrieval", "fts_top_k", 20),
        ("retrieval", "rrf_k", 61),
        ("retrieval", "rerank", "cross-encoder"),
        ("retrieval", "final_k", 7),
        ("answer", "model", "other-model"),
        ("answer", "temperature", 0.5),
        ("answer", "seed", 43),
        ("answer", "refusal_threshold", 0.4),
        ("answer", "high_confidence_threshold", 0.7),
        ("answer", "prompt_template_sha256", "0" * 64),
        ("embeddings", "model", "nomic-embed-text"),
        ("embeddings", "dimensions", 512),
    ],
)
def test_fingerprint_changes_on_every_binding_field(
    tmp_path: Path, section: str, field: str, value: Any
) -> None:
    # model_copy bypasses validation on purpose: the fingerprint must react to
    # every binding-field change, regardless of whether the combo is valid.
    base = _load(tmp_path, "v1")
    mutated_section = getattr(base, section).model_copy(update={field: value})
    mutated = base.model_copy(update={section: mutated_section})
    assert fingerprint(mutated) != fingerprint(base), f"{section}.{field}"


def test_fingerprint_payload_covers_exactly_the_binding_sections(tmp_path: Path) -> None:
    payload = fingerprint_payload(_load(tmp_path, "v1"))
    assert set(payload) == {"chunking", "retrieval", "answer", "embeddings"}
    assert set(payload["answer"]) == {
        "model",
        "temperature",
        "seed",
        "refusal_threshold",
        "high_confidence_threshold",
        "prompt_template_sha256",
    }


def test_fp8_is_first_8_hex_chars(tmp_path: Path) -> None:
    cfg = _load(tmp_path, "v1")
    assert fp8(cfg) == fingerprint(cfg)[:8]
    assert re.fullmatch(r"[0-9a-f]{8}", fp8(cfg))


# --- validation with key paths (§11.4) ---------------------------------------


def test_unknown_key_rejected_with_key_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"chunking\.units"):
        _load(tmp_path, "v1", '[chunking]\nunits = "fixed"\n')


def test_wrong_type_rejected_with_key_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"answer\.temperature"):
        _load(tmp_path, "v1", '[answer]\ntemperature = "hot"\n')


def test_fts_top_k_must_be_zero_in_vector_mode(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"retrieval\.fts_top_k"):
        _load(tmp_path, "v1", "[retrieval]\nfts_top_k = 20\n")


def test_vector_mode_requires_zero_fts_top_k(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"retrieval\.fts_top_k"):
        _load(tmp_path, "v3", '[retrieval]\nmode = "vector"\n')


def test_rerank_requires_hybrid_mode(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"retrieval\.rerank"):
        _load(tmp_path, "v1", '[retrieval]\nrerank = "cross-encoder"\n')


def test_overlap_must_be_smaller_than_size(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"chunking\.overlap_tokens"):
        _load(tmp_path, "v1", "[chunking]\noverlap_tokens = 800\n")


def test_missing_user_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "missing.toml", CONFIGS / "v1.toml")


def test_missing_preset_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(_write_user(tmp_path, ""), tmp_path / "missing.toml")
