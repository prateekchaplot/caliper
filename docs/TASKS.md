# Caliper — Task Tracker

Source of truth for the task definitions: `docs/ARCHITECTURE.md` §14.2 (task breakdown), §14.3 (parallelization map). This file only tracks status; the definitions below are summaries — the architecture document is binding.

**Working agreement (hold-after-every-task):** one task per cycle — implement, verify acceptance criteria, commit, move the task to **Done** with the commit hash, then STOP for review. No task starts before its reviewer approves the previous one and the task's dependencies are Done.

Statuses: ✅ done · 🔄 doing · ⬜ remaining · ✂️ cut (stretch only, per the P5 cut order: FR-13 first, then FR-12)

---

## Done

| ID | Commit(s) | Verified |
|---|---|---|
| T01 | `c76a2bc` (tracker), `d0446f6` (scaffold) | `uv sync` OK (CPython 3.12.16); `uv run caliper --help` renders; `ruff check` + `ruff format --check` + `mypy` clean; 2 tests pass. CI green on push/PR and on `main` after PR #1 merge (run 43717112766). |
| T02 | `489c035` | `uv run pytest` — 38 passed (36 new config tests); `ruff check` + `ruff format --check` clean; `mypy` strict clean. Verified: user-over-preset deep merge; §9.7 contents binding for v1/v2/v3 + shared fixed answerer; fingerprint stable across reloads, differs across versions, and reacts to every binding-field change (parametrized, 17 fields); invalid configs rejected with key path. |

## Doing

_(none — next up: **T03**, pending review; T04 is parallelizable with it per §14.3)_

## Remaining

### P0 — Foundation

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T02 | Config + fingerprint | `src/caliper/config.py`, `configs/caliper.sample.toml`, `configs/v1.toml`, `configs/v2.toml`, `configs/v3.toml` | T01 | Load merges user+preset; fingerprint stable on identical inputs, changes on every §7.8-listed field change; invalid config rejected with key path |
| T03 | Schema + db accessors | `src/caliper/schema.sql`, `src/caliper/db.py` | T02 | `ensure_schema` idempotent; append-only triggers raise on UPDATE/DELETE of `runs`, `run_results`, `spot_checks`, `feedback`; `connect` loads sqlite-vec on Windows; accessors unit-tested |
| T04 | Providers + live smoke test | `src/caliper/providers/*`, `scripts/smoke.py`, `tests/fakes.py` | T02 | `smoke.py` completes one chat call + one 768-dim embed per backend, demonstrates normalization, logs a 429 backoff; OQ1 verified and decision logged |

### P1 — Ingestion

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T05 | Corpus source | `src/caliper/ingest/source.py` | T04 | Tarball fetched once, cached, ref pinned; file→URL mapping unit-tested against 5 known pages; exclusions applied |
| T06 | Parsers + chunkers | `src/caliper/ingest/parse.py`, `src/caliper/ingest/chunkers.py` | T05 | Heading paths correct on fixture corpus; both chunkers hit token bounds; 1,900-token cap enforced |
| T07 | Ingest orchestration | `src/caliper/ingest/__init__.py`, `src/caliper/cli.py` (ingest) | T03, T06 | Double-ingest on mini-corpus yields identical counts; orphan chunks removed; summary JSON matches §9.4 |
| T08 | Embedding batcher + resume | edits within `src/caliper/ingest/` only | T07 | Batch ≤100/request; 429 backoff via providers; interrupt + resume completes without re-embedding finished batches |

### P2 — Ask path

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T09 | Retrieval core | `src/caliper/retrieval/__init__.py` | T03, T04 | `vector` mode returns exactly 5 passages with cosine scores; deterministic on fixed input |
| T10 | RRF fusion | `src/caliper/retrieval/fuse.py` | T09 | RRF of two ranked lists matches the textbook formula on fixture vectors; `hybrid_rrf` wired behind config |
| T11 | Answer + quotes + confidence | `src/caliper/answer/*` | T09 | Refusal on out-of-scope fixture question, no hallucinated citations; quote verifier passes/retries/drops per §7.7; confidence rules unit-tested; full trace persisted |

### P3 — Evaluation

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T12 | Ask CLI + logging | `src/caliper/cli.py` (ask) | T11 | `caliper ask --json` matches §9.2 schema; every call writes `requests` + children with latency/tokens/cost |
| T13 | Golden loader | `src/caliper/eval/golden.py`, `data/golden/GOLDEN.md` | T03 | Rejects tampered file (hash), wrong composition, duplicate IDs |
| T14 | Eval harness + metrics + report fragment | `src/caliper/eval/__init__.py`, `metrics.py`, `report.py` | T12, T13 | 60-question fake run produces all §9.6 metrics incl. median/p95; fingerprint mismatch aborts before any run row; two runs coexist append-only |
| T15 | Spot-check tooling | `src/caliper/eval/spotcheck.py` | T14 | Deterministic seeded sample of 20 answerable questions; judgments append-only; metrics use latest judgment |
| T16 | V1 baseline run | `docs/EVALUATION.md` (initial), run records in `data/caliper.db` | T08, T14, T15 + real corpus | Run v1 complete on real corpus; refusal threshold re-verified once and frozen; results committed. **Reviewer gate: golden set freeze verified before run starts** |

### P4 — Versions

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T17 | V2 segmentation run | `configs/v2.toml`, re-ingest, run records | T16 | v2 index coexists with v1 (per-fingerprint tables); prior traces still reconstructable |
| T18 | V3 reranker + run | optional dep group in `pyproject.toml`, `src/caliper/retrieval/` rerank wiring, run records | T17 | Cross-encoder reranks top-20→5; v3 run recorded; deltas vs v2 visible |
| T19 | Comparison table | `docs/EVALUATION.md`, `caliper report` | T16–T18 | One table with all three runs' metrics incl. deltas; generated by command, committed |

### P5 — Stretch (cut order if slipping: FR-13 → FR-12)

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T20 | Ask page | `src/caliper/web/*` (app, routes, templates, `app.js`) | T12 | Browser ask→answer with citations + confidence; refusal path renders; errors render §9 envelope |
| T21 | Review screen + feedback + export | `src/caliper/web/` additions, `caliper report --feedback csv` | T20 | Feedback stored (idempotent), visible on review page, CSV export round-trips |

### P6 — Setup & wrap-up

| ID | Task | Files | Deps | Acceptance criteria (summary) |
|---|---|---|---|---|
| T22 | Setup command + docs | `src/caliper/cli.py` (setup, serve), `docs/SETUP.md`, `README.md` | T07, T20 | Fresh clone + sample config + one command → working ask end-to-end |
| T23 | Acceptance sweep + demo notes | `tests/integration/acceptance*.py`, `docs/LIMITATIONS.md` | T19, T22 | Acceptance criteria 1–10 mapped to tests (11 = demo); LIMITATIONS.md complete; stretch delivered-or-cut recorded |

---

## Notes

- Max concurrency: 2 agents; ownership rules in ARCHITECTURE §6.3 keep parallel tasks disjoint.
- Contract-touching tasks follow the shared-file protocol (§14.4): PR touches only the owned file + its tests, updates the binding §7–§9 section in the same change, and names dependent tasks.
- Never edit `data/golden/questions.jsonl` after the first measured run; never commit `data/caliper.db`, `data/cache/`, `caliper.toml`, `.env`.

### Decisions recorded during the build

- **T02 — merge order (§11.2 ambiguity):** the version preset is loaded first and the user config is deep-merged *over* it (user wins on conflicts). Rationale: this is the only reading consistent with A2's fallback path ("flip `CALIPER_PROVIDER=ollama`" requires overriding `answer.model`/`embeddings.model`, which preset-wins would forbid since presets are committed). FR-10 comparability is not weakened: identity is enforced by the fingerprint gate (§7.8), computed from the resolved config, not by the merge order.
- **T02 — `answer.prompt_template_sha256`:** presets carry the SHA-256 of the empty string as a placeholder until T11 stores the frozen prompt template file; T11 must replace it in all four TOML files (and in `PROMPT_TEMPLATE_SHA256_PLACEHOLDER` in `config.py`'s default) — the change is fingerprint-visible by construction.
- **T02 — `high_confidence_threshold`:** not pinned in §9.7; initialized to 0.60 in all presets. To be re-verified alongside the refusal threshold at T16 before the V1 baseline run freezes it.
