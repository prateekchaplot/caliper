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

| ID | Branch | Status |
|---|---|---|
| T03 | `arch/caliper/t03` | Implemented + verified (commit `2409b2a`); awaiting review before PR |
| T04 | `arch/caliper/t04` (worktree) | Implemented + verified (commit `e6b5517`); awaiting review before PR; **live smoke pending user** (no `GEMINI_API_KEY`, Ollama not running) |

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
- **T03 — FTS5 flavor (§8.3 wording ambiguity):** §8.3's "contentless external-content rowid-mapped to `chunk_id`" mixes two FTS5 flavors. Implemented as an **external-content** FTS5 table (`content='chunks_<fp8>'`, `content_rowid='chunk_id'`) — it stores no text itself (contentless in effect) but can serve `snippet()` from the content table, and AFTER INSERT/DELETE triggers keep it in sync with orphan removal (FR-02/T07). Smallest choice consistent with §7.4's "rowid-mapped" requirement; recorded here rather than re-derived later.
- **T03 — `runs` lifecycle enforcement:** beyond §8.9's literal triggers, `ensure_schema` also installs `runs_insert_must_start_running` (INSERTs must use status `running`) so the §8.7 lifecycle ("running → terminal is the only permitted UPDATE") cannot be bypassed from row zero. `finish_run()` performs the single legal UPDATE (status + finished_at + summary_json in one statement while `OLD.status = 'running'`).
- **T03 — `ingest_runs` upsert semantics:** "idempotent upsert" (§8.3) implemented as one row per fingerprint (UNIQUE) with `ON CONFLICT DO UPDATE` that resets `finished_at`/`pages_*` until the new run finishes; matches §8.1's "latest per fingerprint".
- **T03 — chunk/embedding accessors added for T07/T08:** `insert_chunks` (returns ids, row-by-row since ids are reported), `get_chunks` (order-preserving, raises on missing id), `delete_chunks` (also deletes the vec rows; FTS follows via trigger), `insert_chunk_embeddings` (serializes via `sqlite_vec.serialize_float32`; validates 768 dims). No business logic added (§7.3).
- **T03 — `db` is a leaf (§7.3):** it imports nothing from `config`; callers pass fingerprints/`fp8` suffixes as strings. Suffix is validated as 8 lowercase hex chars before any SQL interpolation.
- **T03 — sqlite-vec binding detail:** embeddings must be bound as serialized float32 (the package's `serialize_float32`), not Python lists; `connect()` enables extension loading around `sqlite_vec.load()`.
- **T04 — `tests/conftest.py` added (outside the T04 file list):** puts `tests/` on `sys.path` so `tests/fakes.py` is importable from every test directory (§15's "fakes are the suite's only mocks"); smallest mechanism that avoids editing T01-owned `pyproject.toml`. Owned by T04 from here on.
- **T04 — `get_embeddings` rejects `embeddings.dimensions != 768` up front:** 768 is binding (§7.2, vec0 table in §8.3); failing before any model call is cheaper than corrupting the index.
- **T04 — smoke 429 demo is simulated:** a live 429 can't be forced deterministically, so `scripts/smoke.py` exercises the real `call_with_backoff` against a scripted twice-429 op with no-op sleeps; the live retry path is covered by injection-seam tests. Live leg still requires the human smoke run (§15).
- **T04 — `ChatResult.model` echoes the requested model** (not `response.model_version`) so the fingerprint stays consistent when the API appends a version suffix.
- **T04 — unknown price entries raise `ProviderError`:** A3 covers wrong figures, not absent ones — a missing entry must fail loudly. ⚠ `gemini-3.1-flash-lite` (OQ1 fallback) has **no price entry yet**; add its figure to `prices.json` when OQ1 resolves, or `estimate_cost` will raise if it is ever used.
- **T04 — frozen seams:** `tests/fakes.py` API (`FakeChat`, `FakeEmbeddings`, helpers) is the mock foundation for T09–T14; Ollama model pins `qwen2.5:7b-instruct` / `nomic-embed-text` are constants in `providers/ollama.py` (§7.2). **OQ1 still open:** free-tier RPD for `gemini-3.5-flash-lite` must be verified live (AI Studio) at the smoke run and logged in §18.
