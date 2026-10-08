# QueryGuard

Ask a database a question in plain English and get rows back, along with a calibrated estimate of
whether the answer is right. An LLM writes the SQL. A database role that can't write runs it. Three
independent checks then look for answers that are plausible but wrong. QueryGuard returns the SQL, the
rows, every check's verdict and a confidence score, and on a question with more than one reading it asks
for clarification instead of guessing.

**Status:** In development. The core pipeline, evals, HTTP API and web UI run locally; it is not deployed yet.

## Safety first: generated SQL can't write

Two independent layers, each of which holds without the other:

1. **The database.** Generated SQL runs only as `queryguard_ro`, a role with `SELECT` and nothing else,
   which also covers tables created later. Every query runs in a `READ ONLY` transaction that is always
   rolled back, under a transaction-local `statement_timeout`, after an `EXPLAIN` row-estimate check.
   `scripts/verify_db.py` proves that the role's writes fail.
2. **The guardrail.** A static gate on sqlparse tokens, never on raw substrings. It allows one statement and
   `SELECT` only, scans the whole tree for data-modifying CTEs, blocks forbidden functions and keywords,
   limits subquery depth, rejects comments, and adds a `LIMIT`. 81 tests, with no database or network.

The executor doesn't trust the guardrail to have run, and the guardrail doesn't trust the database to be
safe. The API's own state (history, feedback, cache) lives in SQLite, so it never needs a writable
Postgres identity.

## Results

From the recorded runs `full-2026-10-08` and `final-2026-10-08`, prompt version `p-02ea2fb3b358`: 50 questions
through the full pipeline, plus 104 known-wrong mutation queries and 40 golden queries through back-translation
and the judge. The full write-up, including which second queries were reused from earlier runs, is in
[docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md).

**These numbers are final for prompt version `p-02ea2fb3b358`.** The eval is frozen. More detector or prompt
tuning against this golden set would overfit to it.

| metric | result |
|---|---|
| Generation accuracy | **49/50** (39/40 answerable · 10/10 clarification/refusal) |
| Wrong answers flagged (confidence < 0.5) | **99.0%** (103/104) |
| False flags on correct answers | **7.6%** (6/79) |
| Brier score (out-of-fold, grouped 5-fold CV) | **0.038** (hand-set v0: 0.051) |
| Spend for these runs | $1.52, 633 API calls |

**Fixed after the first eval:** `refund_04`. In the first run (`live-2026-10-01`), "gross revenue before refunds" was
answered including unpaid orders, and every check passed it. None of the checks knew the *business rule*. A metric
glossary, shared by the generator, the schema comments and (for revenue questions) the alignment judge, plus a
`revenue_status` sanity check fixed it. refund_04 is now correct with alignment 1.0, and both mutations that had
slipped through are caught. See [Fixed after first eval](docs/EVAL_RESULTS.md#fixed-after-first-eval).

## Architecture

- **Generate** (`generate.py`): the question, a compact introspected schema and few-shot examples go to
  Claude, which returns typed SQL, a `ClarificationNeeded` with runnable readings, or a `CannotAnswer`.
- **Guard** (`guardrails.py`): the static token-level gate described above.
- **Execute** (`executor.py`): four nested database boundaries. It never raises; every outcome is a typed result.
- **Validate** (`validation/`): result-shape sanity checks, back-translation of the SQL into a question
  that is judged against the original, and a second, independently written query whose result must agree.
- **Score** (`confidence.py`): a logistic model over every signal, fitted on the eval run (`calibration.json`).
- **Serve** (`api/`, `web/`): FastAPI streams the pipeline as typed stage events, as JSON or Server-Sent
  Events, behind per-client rate limits, a daily spend ceiling and a response cache. The React UI shows the
  stages arriving live, explains the score signal by signal, and runs your own SQL through the same checks.

## Run it locally

```bash
cp .env.example .env            # fill in passwords and ANTHROPIC_API_KEY; never commit it
docker compose up -d            # Postgres 16 with seed data and the read-only role
uv sync
uv run scripts/verify_db.py     # row counts + proof that the read-only role can't write
uv run pytest                   # no API calls; Postgres tests skip if the container is down
uv run python -m queryguard.api # http://127.0.0.1:8000/docs
```

```bash
curl -N -X POST localhost:8000/v1/query/stream \
     -H 'content-type: application/json' -d '{"question": "How many orders were cancelled?"}'
```

### Web UI

```bash
make dev FAKE=1                 # API + UI, simulated model: $0, no API key needed
make dev                        # API + UI, real model (each new question costs API calls)
```

Open http://localhost:5173. In demo mode (`FAKE=1`) a simulated model answers the eval set's questions from
`evals/golden.yaml`. The guardrail, the read-only database and every check run for real. Try the three example
questions: a normal one, an ambiguous one (pick a reading and it runs), and a pasted `DROP TABLE` (blocked,
with the rule named).

Vite + React + TypeScript (strict), Tailwind, TanStack Query, CodeMirror 6. API types are generated from the
FastAPI OpenAPI schema (`make gen-api`); nothing is restated by hand. `make check` runs lint, typecheck,
Vitest, Playwright (against demo mode) and the production build.

Python 3.12 · uv · FastAPI · SQLAlchemy + psycopg 3 · sqlparse · pandas · PostgreSQL 16 · Anthropic SDK.
