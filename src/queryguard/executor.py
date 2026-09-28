"""Run guardrail-cleared SQL against the read-only role, inside a sandbox.

`guardrails.check()` decides whether SQL is *allowed*. This module decides what
it is allowed to *cost*, and it assumes nothing about what the guardrail let
through -- a caller can reach `execute()` directly, so every protection here has
to hold on its own rather than by arrangement with the layer above.

Four nested boundaries, outermost first:

1. The connection. Only `database_url(readonly=True)` -- the `queryguard_ro`
   role, which holds SELECT and nothing else. The owner URL is never built here.
2. The transaction. Explicitly `SET TRANSACTION READ ONLY`, and always rolled
   back, including on success. A SELECT has nothing to commit, so a rollback
   costs nothing and removes the question of whether anything could persist.
3. `statement_timeout`, transaction-local, set through `set_config` with a bind
   parameter -- the idiom `schema/introspect.py` already uses, so no value is
   ever interpolated into SQL.
4. `EXPLAIN (FORMAT JSON)` before execution. The planner's row estimate is the
   only warning available *before* paying for a query: a cross join over two
   15k-row tables estimates 225 million rows and is refused unexecuted, where a
   row cap alone would have let the server do all that work and then discarded
   it.

What layer 4 does *not* do is bound work, only rows returned. `SELECT count(*)
FROM order_items a CROSS JOIN order_items b` estimates one row, because one row
is what it returns, and the 225-million-row join happens anyway. A LIMIT has the
same effect from the other direction: the guardrail's appended `LIMIT 1000` turns
that cross join's root estimate into 1000 and the query is allowed -- correctly,
as it happens, since PostgreSQL stops the join early. `statement_timeout` is the
only thing here that bounds cost rather than volume, which is why it is not
optional. `tests/test_executor.py` pins both behaviours so neither is a surprise.

Layers 1 and 2 overlap on purpose, and the overlap is visible in what a write
returns. Reaching `execute()` with an INSERT fails with SQLSTATE 25006
(read-only transaction) rather than 42501 (insufficient privilege), because the
transaction check runs first -- the privilege boundary is still there, it is just
no longer the thing that fires. `verify_db.py` proves that inner layer
separately, and `tests/test_executor.py` asserts both.

No exception from the driver escapes: every outcome is an ExecutionResult, so a
caller cannot mistake a failure for an empty result set.

Nothing here calls an API.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from queryguard.config import REPO_ROOT, database_url
from queryguard.llm.client import append_log

# Five seconds is far longer than any question a person waits on and short
# enough that a pathological plan cannot hold a connection open.
DEFAULT_STATEMENT_TIMEOUT_MS = 5000

# Matches guardrails.DEFAULT_MAX_ROWS. Kept as its own constant because the two
# caps answer different questions -- the guardrail rewrites the SQL, this one
# bounds what is read out of the cursor even when the SQL was never rewritten.
DEFAULT_MAX_ROWS = 1000

# The planner's estimate, not a measurement. Generous enough that ordinary
# aggregate scans over the seeded tables pass, tight enough that an accidental
# cross join does not.
DEFAULT_MAX_ESTIMATED_ROWS = 100_000

OUTCOME_OK = "ok"
OUTCOME_REFUSED = "refused"
OUTCOME_FAILED = "failed"


def log_path() -> Path:
    """Where executions are logged. Overridable so tests never touch the real log."""
    import os

    override = os.getenv("QUERYGUARD_EXECUTOR_LOG")
    return Path(override) if override else REPO_ROOT / "logs" / "executions.jsonl"


def sql_hash(sql: str) -> str:
    """Stable fingerprint of a statement. The SQL itself is never logged.

    Same reasoning as `client.prompt_hash`: the log needs to correlate runs of
    the same query without keeping the query, which may embed values from a
    question a user considered private.
    """
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class ExecutorConfig:
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS
    max_rows: int = DEFAULT_MAX_ROWS
    max_estimated_rows: int = DEFAULT_MAX_ESTIMATED_ROWS

    # Off skips the pre-flight plan entirely, which also removes the row-estimate
    # refusal. Only for a caller that has its own reason to trust the query.
    explain_first: bool = True


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str


@dataclass(frozen=True)
class ExecutionResult:
    """One outcome: ok, refused before execution, or failed in the database.

    `rows` is None unless `outcome` is ok, so an empty DataFrame means "the query
    ran and matched nothing" and cannot be confused with "the query never ran".
    """

    outcome: str
    sql_sha256: str
    rows: pd.DataFrame | None = None
    columns: tuple[ColumnSpec, ...] = ()
    row_count: int = 0
    truncated: bool = False
    execution_ms: int = 0
    plan: Any = None
    estimated_rows: float | None = None
    reason: str | None = None
    error_class: str | None = None
    error_message: str | None = None
    sqlstate: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == OUTCOME_OK

    @property
    def column_names(self) -> list[str]:
        return [col.name for col in self.columns]


# ----------------------------------------------------------------- the sandbox


def _describe_error(exc: SQLAlchemyError) -> tuple[str, str, str | None]:
    """(class, first line, sqlstate) from whatever the driver raised.

    The driver exception is unwrapped for its name and SQLSTATE because
    "InsufficientPrivilege/42501" tells a caller what to do and
    "ProgrammingError" does not.
    """
    orig = getattr(exc, "orig", None)
    subject = orig if orig is not None else exc
    message = str(subject).strip().splitlines()
    return (
        type(subject).__name__,
        message[0] if message else type(subject).__name__,
        getattr(subject, "sqlstate", None),
    )


def _plan_estimate(plan: Any) -> float | None:
    """The root node's Plan Rows from EXPLAIN (FORMAT JSON) output."""
    try:
        return float(plan[0]["Plan"]["Plan Rows"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _explain(conn: Any, sql: str) -> tuple[Any, float | None]:
    """Plan the query without running it.

    The statement is concatenated rather than bound: EXPLAIN takes a query, not a
    value, so there is no parameter form of this. What makes it acceptable is
    everything around it -- the text has been through the guardrail, the role can
    only read, and the transaction is discarded either way.
    """
    plan = conn.execute(text(f"EXPLAIN (FORMAT JSON) {sql}")).scalar()
    return plan, _plan_estimate(plan)


def _frame(result: Any, max_rows: int) -> tuple[pd.DataFrame, bool]:
    """Read at most max_rows, and report whether more were waiting.

    One extra row is fetched to detect truncation. Counting the true total would
    mean reading everything the cap exists to avoid reading.
    """
    fetched = result.fetchmany(max_rows + 1)
    truncated = len(fetched) > max_rows
    if truncated:
        fetched = fetched[:max_rows]
    return pd.DataFrame(fetched, columns=list(result.keys())), truncated


def _columns(frame: pd.DataFrame) -> tuple[ColumnSpec, ...]:
    return tuple(
        ColumnSpec(name=str(name), dtype=str(dtype)) for name, dtype in frame.dtypes.items()
    )


def execute(
    sql: str, config: ExecutorConfig | None = None, *, log: Path | None = None
) -> ExecutionResult:
    """Run one statement in the sandbox and return a structured outcome.

    Never raises for anything the database refuses; a driver error becomes a
    failed ExecutionResult carrying the error class and SQLSTATE.
    """
    config = config or ExecutorConfig()
    digest = sql_hash(sql)

    # readonly=True is the whole point of this line. There is no branch here and
    # no parameter that could turn it into the owner connection.
    engine = create_engine(database_url(readonly=True))
    try:
        with engine.connect() as conn:
            try:
                return _run(conn, sql, digest, config, log)
            finally:
                # Always, including after a clean SELECT. Nothing this module
                # does is ever committed.
                conn.rollback()
    except SQLAlchemyError as exc:
        # Connecting failed, or the rollback did. Same contract as any other
        # database error: a result, not an exception.
        error_class, message, sqlstate = _describe_error(exc)
        return _log_and_return(
            ExecutionResult(
                outcome=OUTCOME_FAILED,
                sql_sha256=digest,
                error_class=error_class,
                error_message=message,
                sqlstate=sqlstate,
            ),
            config,
            log,
        )
    finally:
        engine.dispose()


def _run(
    conn: Any, sql: str, digest: str, config: ExecutorConfig, log: Path | None
) -> ExecutionResult:
    """The body of one sandboxed execution, inside an open transaction."""
    try:
        # Must precede every other statement in the transaction, or Postgres
        # rejects it with 25001.
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(
            text("SELECT set_config('statement_timeout', :ms, true)"),
            {"ms": str(config.statement_timeout_ms)},
        )

        plan: Any = None
        estimated: float | None = None
        if config.explain_first:
            plan, estimated = _explain(conn, sql)
            if estimated is not None and estimated > config.max_estimated_rows:
                return _log_and_return(
                    ExecutionResult(
                        outcome=OUTCOME_REFUSED,
                        sql_sha256=digest,
                        plan=plan,
                        estimated_rows=estimated,
                        reason=(
                            f"the planner estimates {int(estimated):,} rows, above the "
                            f"limit of {config.max_estimated_rows:,}; the query was not run"
                        ),
                    ),
                    config,
                    log,
                )

        started = time.perf_counter()
        result = conn.execute(text(sql))
        frame, truncated = _frame(result, config.max_rows)
        execution_ms = int((time.perf_counter() - started) * 1000)

        return _log_and_return(
            ExecutionResult(
                outcome=OUTCOME_OK,
                sql_sha256=digest,
                rows=frame,
                columns=_columns(frame),
                row_count=len(frame),
                truncated=truncated,
                execution_ms=execution_ms,
                plan=plan,
                estimated_rows=estimated,
            ),
            config,
            log,
        )

    except SQLAlchemyError as exc:
        error_class, message, sqlstate = _describe_error(exc)
        return _log_and_return(
            ExecutionResult(
                outcome=OUTCOME_FAILED,
                sql_sha256=digest,
                error_class=error_class,
                error_message=message,
                sqlstate=sqlstate,
            ),
            config,
            log,
        )


# -------------------------------------------------------------------- logging


def _log_and_return(
    result: ExecutionResult, config: ExecutorConfig, log: Path | None
) -> ExecutionResult:
    """One JSONL line per execution, whatever the outcome.

    A refusal and a failure are as worth recording as a success -- a run of
    refusals is how a bad prompt change first becomes visible.
    """
    append_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sql_sha256": result.sql_sha256,
            "outcome": result.outcome,
            "row_count": result.row_count,
            "truncated": result.truncated,
            "execution_ms": result.execution_ms,
            "estimated_rows": result.estimated_rows,
            "statement_timeout_ms": config.statement_timeout_ms,
            "max_rows": config.max_rows,
            "error_class": result.error_class,
            "sqlstate": result.sqlstate,
            "reason": result.reason,
        },
        log or log_path(),
    )
    return result
