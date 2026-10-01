"""Tests for the sandboxed executor.

These need a live database, because what is under test is not logic -- it is
whether PostgreSQL actually refuses what this project claims it refuses. A mock
that returns 42501 on cue proves only that the mock was written to agree with
the test. So every test here takes `live_database` and skips cleanly when the
container is down.

The read-only test deliberately bypasses `guardrails.check()` and hands the
executor an INSERT. That is the whole point of it: the guardrail is the layer
that is *supposed* to stop writes, so the only way to find out whether the
executor would stop one on its own is to remove the guardrail and look.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, text

from queryguard.config import database_url
from queryguard.executor import (
    DEFAULT_MAX_ROWS,
    OUTCOME_FAILED,
    OUTCOME_OK,
    OUTCOME_REFUSED,
    ExecutorConfig,
    execute,
    sql_hash,
)
from queryguard.guardrails import check
from queryguard.llm.examples import EXAMPLES

# The seed is pinned to a fixed date anchor, so these are exact. A change here
# means the seed moved, which is worth failing loudly over.
EXPECTED_ROW_COUNTS = {
    "example1": 1,  # one customer
    "example2": 5,  # LIMIT 5
    "example3": 12,  # one row per category
    "example4": 1,  # a single count
    "example5": 8,  # one row per country
    "example6": 5,  # LIMIT 5
    "example7-flagged_active": 1,
    "example7-purchased_recently": 1,
}


@pytest.fixture(autouse=True)
def _isolate_execution_log(tmp_path, monkeypatch):
    """Never append to the real logs/executions.jsonl from a test."""
    monkeypatch.setenv("QUERYGUARD_EXECUTOR_LOG", str(tmp_path / "executions.jsonl"))


def _example_queries() -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for number, example in enumerate(EXAMPLES, 1):
        if example.sql:
            queries.append((f"example{number}", example.sql))
        for label, sql in example.interpretations:
            queries.append((f"example{number}-{label}", sql))
    return queries


EXAMPLE_QUERIES = _example_queries()


def _count_categories() -> int:
    """Read the table through a connection the executor does not own."""
    engine = create_engine(database_url(readonly=True))
    try:
        with engine.connect() as conn:
            return int(conn.execute(text("SELECT count(*) FROM categories")).scalar())
    finally:
        engine.dispose()


# ------------------------------------------------------------------- identity


def test_the_readonly_url_is_a_different_role_from_the_owner() -> None:
    """No database needed: this is about which credential is configured."""
    assert database_url(readonly=True).username == "queryguard_ro"
    assert database_url().username != database_url(readonly=True).username


def test_the_executor_connects_as_queryguard_ro(live_database) -> None:
    """Asks the server who it is, rather than trusting the URL."""
    result = execute("SELECT current_user AS who")
    assert result.ok, result.error_message
    assert result.rows.iloc[0]["who"] == "queryguard_ro"


def test_the_executor_asks_for_the_readonly_url_and_never_the_owner(monkeypatch) -> None:
    """The owner URL must not be reachable from this module, even by accident."""
    calls: list[dict] = []
    real = database_url

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr("queryguard.executor.database_url", spy)
    execute("SELECT 1 AS n")
    assert calls, "database_url was never called"
    assert all(call.get("readonly") is True for call in calls), calls


def test_the_session_is_read_only(live_database) -> None:
    result = execute("SELECT current_setting('transaction_read_only') AS ro")
    assert result.ok, result.error_message
    assert result.rows.iloc[0]["ro"] == "on"


# ------------------------------------------------------------- the examples run


@pytest.mark.parametrize("name,sql", EXAMPLE_QUERIES, ids=[n for n, _ in EXAMPLE_QUERIES])
def test_every_few_shot_example_executes(name, sql, live_database) -> None:
    """The guardrail clears them and the database answers them.

    Run through `check()` first, because that is the only path by which SQL is
    supposed to reach the executor, and five of these have no LIMIT of their own.
    """
    guardrail = check(sql)
    assert guardrail.allowed, f"{name} blocked by {guardrail.rule}: {guardrail.reason}"

    result = execute(guardrail.sql_to_execute)
    assert result.ok, f"{name} failed: {result.sqlstate} {result.error_message}"
    assert result.row_count == EXPECTED_ROW_COUNTS[name], name
    assert not result.truncated
    assert result.columns, "column metadata is missing"
    assert len(result.rows) == result.row_count


def test_an_example_reports_its_columns_and_types(live_database) -> None:
    result = execute(check(EXAMPLES[2].sql).sql_to_execute)
    assert result.column_names == ["category", "product_count"]
    assert "int" in dict((c.name, c.dtype) for c in result.columns)["product_count"]


# ----------------------------------------------------------------- the timeout


def test_the_statement_timeout_fires_and_is_reported_not_raised(live_database) -> None:
    """A ten-second sleep under a 500 ms budget, and no exception reaches here."""
    result = execute("SELECT pg_sleep(10)", ExecutorConfig(statement_timeout_ms=500))

    assert result.outcome == OUTCOME_FAILED
    assert result.sqlstate == "57014", f"expected a query cancellation, got {result.sqlstate}"
    assert result.error_class == "QueryCanceled"
    assert "timeout" in result.error_message.lower()
    assert result.rows is None, "a failure must not offer a result set"


def test_the_timeout_does_not_leak_into_the_next_execution(live_database) -> None:
    """set_config(is_local => true) resets when the transaction ends."""
    execute("SELECT pg_sleep(10)", ExecutorConfig(statement_timeout_ms=500))
    result = execute("SELECT pg_sleep(1) AS slept")
    assert result.ok, result.error_message


# --------------------------------------------------------- the row estimate


def test_a_huge_cross_join_is_refused_without_executing(live_database) -> None:
    """225 million estimated rows, refused on the plan alone.

    A row cap alone would not help here: the server would do the whole join and
    then have 1000 rows read out of it.
    """
    sql = "SELECT * FROM order_items AS a CROSS JOIN order_items AS b"
    result = execute(sql)

    assert result.outcome == OUTCOME_REFUSED
    assert result.rows is None, "a refused query must not return rows"
    assert result.execution_ms == 0, "nothing should have been executed"
    assert result.estimated_rows > 100_000
    assert "225,000,000" in result.reason, result.reason
    assert result.plan is not None, "the plan is the evidence for the refusal"


def test_the_estimate_ceiling_is_configurable(live_database) -> None:
    tight = ExecutorConfig(max_estimated_rows=10)
    assert execute("SELECT * FROM orders", tight).outcome == OUTCOME_REFUSED
    assert execute("SELECT * FROM orders", ExecutorConfig()).ok


def test_a_limit_lowers_the_estimate_and_the_query_is_allowed(live_database) -> None:
    """The guardrail's appended LIMIT changes what the planner estimates.

    Documented rather than prevented: with a LIMIT the same cross join returns
    1000 rows and PostgreSQL stops early, so it really is cheap. The refusal is
    about volume, and this is genuinely low volume.
    """
    raw = "SELECT * FROM order_items AS a CROSS JOIN order_items AS b"
    assert execute(raw).outcome == OUTCOME_REFUSED

    capped = check(raw)
    result = execute(capped.sql_to_execute)
    assert result.ok, result.error_message
    assert result.estimated_rows == DEFAULT_MAX_ROWS + 1.0
    assert result.row_count == DEFAULT_MAX_ROWS
    assert result.truncated, "the extra row the guardrail allowed is the overflow signal"


def test_the_estimate_bounds_rows_returned_not_work_done(live_database) -> None:
    """The known gap in the pre-flight, pinned so it cannot regress silently.

    An aggregate over a cross join estimates one row -- one row is what it
    returns -- and the 225-million-row join is performed regardless. Only
    `statement_timeout` bounds this; the row estimate cannot see it.
    """
    result = execute("SELECT count(*) AS n FROM order_items AS a CROSS JOIN order_items AS b")
    assert result.estimated_rows == 1.0, "the estimate is about rows out, not work"
    assert result.outcome != OUTCOME_REFUSED, "the pre-flight cannot catch this shape"


def test_a_timed_out_query_keeps_its_pre_flight_estimate(live_database) -> None:
    """The plan was made before the timeout fired, so the failure still carries it.

    The test above runs this join under the default 5 s timeout, which it takes
    about 5 s to finish -- so it lands on either side. This one forces the
    timeout, so the error path is exercised every run rather than by chance.
    """
    result = execute(
        "SELECT count(*) AS n FROM order_items AS a CROSS JOIN order_items AS b",
        ExecutorConfig(statement_timeout_ms=500),
    )
    assert result.outcome == OUTCOME_FAILED
    assert result.sqlstate == "57014"
    assert result.estimated_rows == 1.0
    assert result.plan is not None
    assert result.rows is None


def test_an_ordinary_aggregate_is_not_refused(live_database) -> None:
    """The ceiling must not reject the queries this project exists to run."""
    result = execute("SELECT count(*) AS n FROM order_items")
    assert result.ok, result.error_message
    assert result.estimated_rows == 1.0


# ------------------------------------------------------- the read-only boundary


def test_a_bypassed_write_is_refused_and_changes_nothing(live_database) -> None:
    """Hand the executor an INSERT directly, with no guardrail in the way.

    The privilege check runs at plan time, so the EXPLAIN pre-flight is what
    refuses this, with 42501 -- the grant boundary, not the transaction one.
    """
    before = _count_categories()
    result = execute("INSERT INTO categories (name) VALUES ('pwned')")

    assert result.outcome == OUTCOME_FAILED
    assert result.sqlstate == "42501", f"expected insufficient privilege, got {result.sqlstate}"
    assert result.error_class == "InsufficientPrivilege"
    assert _count_categories() == before, "the table changed"


def test_the_read_only_transaction_catches_a_write_on_its_own(live_database) -> None:
    """With the EXPLAIN pre-flight off, the inner boundary is the one that fires.

    Two independent layers refuse the same write: privileges at plan time (42501)
    and the read-only transaction at execution time (25006). This test exists so
    that turning one off cannot quietly leave the other untested.
    """
    before = _count_categories()
    result = execute(
        "INSERT INTO categories (name) VALUES ('pwned')",
        ExecutorConfig(explain_first=False),
    )

    assert result.outcome == OUTCOME_FAILED
    assert result.sqlstate == "25006", (
        f"expected a read-only transaction error, got {result.sqlstate}"
    )
    assert result.error_class == "ReadOnlySqlTransaction"
    assert _count_categories() == before, "the table changed"


@pytest.mark.parametrize(
    "sql,sqlstate",
    [
        # DML can be planned, so the plan-time privilege check is what refuses it.
        ("UPDATE categories SET name = 'x'", "42501"),
        ("DELETE FROM categories", "42501"),
        # DDL cannot be planned at all -- `EXPLAIN CREATE TABLE ...` is not valid
        # syntax -- so the pre-flight refuses these before privileges are reached.
        ("CREATE TABLE should_not_exist (a int)", "42601"),
        ("DROP TABLE categories", "42601"),
        ("TRUNCATE categories", "42601"),
    ],
)
def test_every_kind_of_write_is_refused(sql, sqlstate, live_database) -> None:
    """Three mechanisms refuse writes here, and which one fires depends on the shape.

    Asserting the exact SQLSTATE rather than "some error" is the point: a write
    refused for an unexpected reason is a write whose refusal nobody has actually
    verified, which is the distinction `verify_db.py:expect_denied()` draws.
    """
    before = _count_categories()

    result = execute(sql)
    assert result.outcome == OUTCOME_FAILED, sql
    assert result.sqlstate == sqlstate, f"{sql} -> {result.sqlstate} {result.error_class}"

    # With the pre-flight removed, the read-only transaction is the backstop for
    # every one of them, DDL included.
    bypassed = execute(sql, ExecutorConfig(explain_first=False))
    assert bypassed.sqlstate == "25006", f"{sql} -> {bypassed.sqlstate}"

    assert _count_categories() == before, f"{sql} changed the table"


# ------------------------------------------------------------------ truncation


def test_a_five_thousand_row_query_is_truncated_to_the_cap(live_database) -> None:
    """orders holds exactly 5000 rows, and the cap is 1000."""
    result = execute("SELECT * FROM orders")

    assert result.ok, result.error_message
    assert result.row_count == DEFAULT_MAX_ROWS
    assert len(result.rows) == DEFAULT_MAX_ROWS
    assert result.truncated is True


def test_a_result_that_fits_is_not_marked_truncated(live_database) -> None:
    result = execute("SELECT * FROM orders LIMIT 1000", ExecutorConfig(max_rows=1000))
    assert result.row_count == 1000
    assert result.truncated is False, "exactly at the cap is not over it"


def test_the_row_cap_is_configurable(live_database) -> None:
    result = execute("SELECT * FROM orders", ExecutorConfig(max_rows=7))
    assert result.row_count == 7
    assert result.truncated is True


def test_an_empty_result_is_a_success_not_a_failure(live_database) -> None:
    """An empty DataFrame and "the query never ran" must not look alike."""
    result = execute("SELECT * FROM orders WHERE 1 = 0")
    assert result.ok
    assert result.row_count == 0
    assert result.truncated is False
    assert result.rows is not None and result.rows.empty


# ------------------------------------------------------------ structured errors


@pytest.mark.parametrize(
    "sql,sqlstate,error_class",
    [
        ("SELECT * FROM does_not_exist", "42P01", "UndefinedTable"),
        ("SELECT no_such_column FROM orders", "42703", "UndefinedColumn"),
        ("SELECT 1 +", "42601", "SyntaxError"),
        ("SELECT 1/0", "22012", "DivisionByZero"),
    ],
)
def test_database_errors_come_back_structured(sql, sqlstate, error_class, live_database) -> None:
    """No driver exception ever reaches the caller."""
    result = execute(sql)
    assert result.outcome == OUTCOME_FAILED
    assert result.sqlstate == sqlstate
    assert result.error_class == error_class
    assert result.error_message
    assert result.rows is None


# ------------------------------------------------------------------- the log


def _log_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_successful_execution_is_logged_without_the_sql(tmp_path, live_database) -> None:
    sql = "SELECT count(*) AS n FROM orders"
    log = tmp_path / "exec.jsonl"
    execute(sql, log=log)

    entries = _log_lines(log)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["outcome"] == OUTCOME_OK
    assert entry["sql_sha256"] == sql_hash(sql)
    assert entry["row_count"] == 1
    assert "orders" not in json.dumps(entry), "the SQL text leaked into the log"


def test_a_refusal_and_a_failure_are_logged_too(tmp_path, live_database) -> None:
    """A run of refusals is how a bad prompt change first becomes visible."""
    log = tmp_path / "exec.jsonl"
    execute("SELECT * FROM order_items AS a CROSS JOIN order_items AS b", log=log)
    execute("SELECT * FROM does_not_exist", log=log)

    outcomes = [entry["outcome"] for entry in _log_lines(log)]
    assert outcomes == [OUTCOME_REFUSED, OUTCOME_FAILED]


def test_the_log_records_the_sqlstate_of_a_failure(tmp_path, live_database) -> None:
    log = tmp_path / "exec.jsonl"
    execute("SELECT pg_sleep(10)", ExecutorConfig(statement_timeout_ms=500), log=log)
    entry = _log_lines(log)[0]
    assert entry["sqlstate"] == "57014"
    assert entry["statement_timeout_ms"] == 500


def test_the_same_sql_hashes_the_same_and_different_sql_does_not() -> None:
    assert sql_hash("SELECT 1") == sql_hash("SELECT 1")
    assert sql_hash("SELECT 1") != sql_hash("SELECT 2")
