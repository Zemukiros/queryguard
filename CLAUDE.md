# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

QueryGuard is Text-to-SQL with a hard read-only database boundary: an LLM (Anthropic SDK) turns a question
into SQL, a static guardrail gates it, a sandboxed executor runs it as a read-only role, and validation
layers (result sanity checks, back-translation, agreement, confidence) flag answers that are probably wrong.
Python 3.12 · uv · FastAPI · SQLAlchemy + psycopg 3 · sqlparse · pandas · PostgreSQL 16 in Docker.

README.md is the public overview; module docstrings are the design documentation — read the top of a module
before changing it.

## Setup and commands

```bash
cp .env.example .env                 # then fill in values; .env is gitignored, never commit it
docker compose up -d                 # Postgres; db/init/*.sql runs on FIRST start only (empty volume)
uv sync
uv run scripts/verify_db.py          # row counts + proof the read-only role cannot write

uv run pytest                        # full suite
uv run pytest tests/test_guardrails.py -k cte    # single file / test

uv run python -m queryguard.schema.introspect --refresh    # rebuild schema_cache.json
uv run python -m queryguard.generate "How many orders were cancelled?"   # SQL only  (API call)
uv run python -m queryguard.pipeline "how many orders last quarter?"     # end to end (API calls)
uv run python -m queryguard.api                       # HTTP API on :8000 (/docs); questions cost API calls
uv run python -m queryguard.api export-feedback       # incorrect feedback -> evals/feedback_candidates.yaml

make dev FAKE=1      # API (demo mode: llm/fake.py, $0) + Vite on :5173; plain `make dev` uses the real model
make gen-api         # regenerate web/src/api/schema.ts after any API model change, then commit it
make test            # pytest + vitest      make e2e   # Playwright vs demo mode      make check   # everything

uv run python -m evals.run_golden                     # execute golden SQL, record row counts/hashes
uv run python -m evals.mutations                      # build known-wrong negatives
uv run python -m evals.run_eval                       # dry run, fake client, no spend
uv run python -m evals.run_eval --live --run-id ID    # real API spend; same ID resumes
uv run python -m evals.recompute --run-id ID          # re-derive labels/confidence offline
uv run python -m evals.calibrate --run-id ID          # fit confidence weights offline (needs `uv sync --group eval`)
```

Changing `db/init/` requires `docker compose down -v` (wipes the volume) to take effect.

Tests that need Postgres skip with a message when the container is down; `tests/test_guardrails.py`
touches no database or network and must never skip.

## Architecture (`src/queryguard/`)

`pipeline.py` orchestrates: **generate → guard → execute → validate → score**, and keeps each outcome a
distinct type (`ClarificationNeeded`, `CannotAnswer`, guardrail rejection, execution result) — don't collapse them.
There is ONE code path: `stream_question` is an async generator of typed stage events (`events.py` documents the
sequence and payloads); `run_question` / `run_answer` only drain it. Blocking SDK/DB calls go through
`asyncio.to_thread`. Add a stage there, never in a wrapper.

- `config.py` — two database identities that must never be confused: `DATABASE_URL` (owner, used only for
  introspection) and `DATABASE_URL_READONLY` (`queryguard_ro`, used to execute generated SQL). Always go
  through `database_url(readonly=...)`; it pins psycopg 3 and the session time zone to UTC.
- `schema/introspect.py` — introspects the DB and renders a compact schema for the prompt; cached in
  `schema_cache.json` (gitignored, regenerable).
- `llm/` — `client.py` wraps the SDK with a process-wide request cap (`QUERYGUARD_MAX_REQUESTS`, default 50),
  cost estimation and a JSONL call log; `prompt.py` + `examples.py` build the cached system prompt and few-shot examples.
- `generate.py` — question → typed `GeneratedSQL` / `ClarificationNeeded` / `CannotAnswer`.
- `guardrails.py` — static gate on sqlparse **tokens, never raw substrings**: single statement, SELECT only,
  DML scanned tree-wide (data-modifying CTEs), forbidden functions/keywords, subquery depth, comments,
  row cap (auto-appends `LIMIT`). No DB or network access in this module.
- `executor.py` — four nested boundaries: read-only role → `SET TRANSACTION READ ONLY` + always rollback →
  transaction-local `statement_timeout` → `EXPLAIN` row-estimate check. Never raises; every outcome is an `ExecutionResult`.
- `validation/` — `sanity.py` (result-shape flags; advice, not a gate), `backtranslate.py`, `agreement.py`,
  `confidence.py` (weights load from `calibration.json`, written by `evals/calibrate.py`; v0 hand-set
  weights are the fallback). `pipeline.MAX_CALLS_PER_QUESTION` (4) bounds API calls per question.

- `api/` — FastAPI (`app.py`; `create_app()` is the factory, tests inject a schema and fake-backed clients).
  Admission before any API call: cache hit (free, no rate-limit use) → per-client rate limit (429) → daily spend
  ceiling read from the LLM call log, with a per-question reserve (503). History/feedback/cache live in SQLite
  (`store.py`, `data/app.db`) — never give the API a writable Postgres identity. Limits are env vars (`settings.py`).
  Stage exception text is logged, not returned.

`web/` — Vite + React + TS (strict), Tailwind v4, TanStack Query, CodeMirror 6. `src/api/schema.ts` is GENERATED
from the OpenAPI schema (`scripts/dump_openapi.py`, no server needed); use its types via `src/api/client.ts`, never
hand-write a server shape. `src/api/sse.ts` reads SSE over POST (fetch + ReadableStream). `src/lib/stages.ts` owns
the timeline model (order, verdicts: done / warn / blocked / skipped). Demo mode (`QUERYGUARD_FAKE_LLM=1`) answers
from the golden set and logs to `logs/fake_llm_calls.jsonl`, never the real ledger. Response models subclass
`events.Out` so defaulted fields are required in the generated types.

`db/init/` holds the schema, seed data, the `queryguard_ro` role and column comments. `evals/` holds the golden
set (`golden.yaml`), mutation negatives and the calibration harness; live run outputs are committed under
`evals/results/<run-id>.*`, dry runs are gitignored. Results write-up: `docs/EVAL_RESULTS.md`.

## Conventions

- **The database is the real boundary; the guardrail is the second one.** Every protection in `executor.py`
  must hold on its own — never rely on the guardrail having run first. Never execute generated SQL as the owner.
- **Anything that calls the API costs money.** Prefer the dry run / fake client; tests must not make live calls.
  Ask before a `--live` eval or any script that hits the API, and state the expected call count.
- Reported metrics come only from recorded runs in `evals/results/`; don't quote numbers without one.
- `logs/` (LLM call logs) and `.env` are gitignored — keep it that way; logs can contain question text and cost data.
- Determinism: sessions run in UTC, eval runs are resumable by `--run-id`, and labels are recomputable offline.
