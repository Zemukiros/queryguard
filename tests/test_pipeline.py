"""Tests for the wired pipeline: generate, guard, execute.

Every test injects a fake SDK client, so nothing here constructs a real
anthropic.Anthropic and the suite cannot spend money or need an API key. The
schema is synthetic for the same reason -- `build_system_blocks(None)` would go
and load the real one, which is a dependency the generate step does not need in
order to be tested.

The execute step is real. Faking it would leave the interesting question --
whether the SQL a model writes survives the guardrail *and* the read-only role --
answered by nobody, so the tests that reach that far take `live_database`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import pandas as pd

from queryguard.executor import (
    OUTCOME_FAILED,
    OUTCOME_OK,
    OUTCOME_REFUSED,
    ExecutionResult,
    ExecutorConfig,
)
from queryguard.generate import Ambiguity, ClarificationNeeded, GeneratedSQL, Interpretation
from queryguard.guardrails import DEFAULT_MAX_ROWS, GuardrailConfig
from queryguard.llm.client import LLMClient, reset_request_count
from queryguard.pipeline import PipelineResult, run_question
from queryguard.schema.introspect import ColumnInfo, DatabaseSchema, TableInfo
from queryguard.validation.sanity import CHECK_EMPTY, WARN


@pytest.fixture(autouse=True)
def _isolate_logs_and_counter(tmp_path, monkeypatch):
    """Keep tests out of the real logs, and out of each other's request count."""
    monkeypatch.setenv("QUERYGUARD_LLM_LOG", str(tmp_path / "llm_calls.jsonl"))
    monkeypatch.setenv("QUERYGUARD_EXECUTOR_LOG", str(tmp_path / "executions.jsonl"))
    reset_request_count()
    yield
    reset_request_count()


# ------------------------------------------------------------------ the fakes


class FakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 1000
        self.output_tokens = 200
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class FakeMessage:
    def __init__(self, parsed) -> None:
        self.parsed_output = parsed
        self.usage = FakeUsage()


class FakeMessages:
    def __init__(self, parsed) -> None:
        self._parsed = parsed
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return FakeMessage(self._parsed)


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic. Records calls, never uses the network."""

    def __init__(self, parsed) -> None:
        self.messages = FakeMessages(parsed)

    @property
    def calls(self):
        return self.messages.calls


def _synthetic_schema() -> DatabaseSchema:
    """Enough of a schema to build a prompt. Never compared against the database."""
    return DatabaseSchema(
        tables=[
            TableInfo(
                name="orders",
                row_count=5000,
                columns=[
                    ColumnInfo(name="order_id", sql_type="integer", nullable=False),
                    ColumnInfo(name="status", sql_type="varchar", nullable=False),
                ],
            )
        ],
        extracted_at=datetime.now(timezone.utc),
    )


def _answer(sql: str) -> GeneratedSQL:
    return GeneratedSQL(
        sql=sql,
        explanation="Counts what was asked for.",
        confidence=0.9,
        tables_used=["orders"],
        columns_used=["orders.status"],
        assumptions=[],
        ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]),
    )


def _ambiguous_answer() -> GeneratedSQL:
    return GeneratedSQL(
        sql="",
        explanation="Two readings.",
        confidence=0.4,
        tables_used=["orders"],
        columns_used=[],
        assumptions=[],
        ambiguity=Ambiguity(
            is_ambiguous=True,
            interpretations=[
                Interpretation(
                    label="gross", sql="SELECT sum(total_amount) FROM orders;", explanation="All."
                ),
                Interpretation(
                    label="net",
                    sql="SELECT sum(total_amount) FROM orders WHERE status <> 'cancelled';",
                    explanation="Excludes cancelled.",
                ),
            ],
        ),
    )


def _run(sql_or_answer, **kwargs):
    """Drive the pipeline with a fake that returns exactly this answer."""
    parsed = sql_or_answer if isinstance(sql_or_answer, GeneratedSQL) else _answer(sql_or_answer)
    fake = FakeAnthropic(parsed)
    outcome = run_question(
        "how many orders are cancelled?",
        client=LLMClient(sdk_client=fake),
        schema=_synthetic_schema(),
        **kwargs,
    )
    return outcome, fake


# ---------------------------------------------------------------- the happy path


def test_a_question_becomes_rows(live_database) -> None:
    outcome, fake = _run("SELECT count(*) AS n FROM orders WHERE status = 'cancelled'")

    assert isinstance(outcome, PipelineResult)
    assert outcome.ok, outcome.execution.error_message
    assert outcome.execution.row_count == 1
    assert outcome.execution.rows.iloc[0]["n"] > 0
    assert len(fake.calls) == 1, "exactly one API call per question"


def test_a_missing_limit_is_added_on_the_way_through(live_database) -> None:
    """The SQL that runs is not the SQL the model wrote, and the result says so."""
    model_sql = "SELECT order_id FROM orders"
    outcome, _ = _run(model_sql)

    assert outcome.answer.sql == model_sql
    assert outcome.guardrail.rewritten_sql == f"{model_sql}\nLIMIT {DEFAULT_MAX_ROWS + 1}"
    assert outcome.sql == outcome.guardrail.rewritten_sql
    assert outcome.ok
    assert outcome.execution.row_count == DEFAULT_MAX_ROWS
    assert outcome.execution.truncated, "orders holds 5000 rows; 1000 is not all of them"


def test_an_overflowing_result_is_reported_as_truncated(live_database) -> None:
    """The guardrail lets one row past the cap so the executor can see it.

    Capped at exactly max_rows, the database had no 1001st row to offer, and a
    15k-row answer came back as a complete-looking 1000 with truncated=False.
    """
    fake = FakeAnthropic(_answer("SELECT * FROM order_items;"))
    outcome = run_question(
        "List every order item",
        client=LLMClient(sdk_client=fake),
        schema=_synthetic_schema(),
    )

    assert outcome.ok, outcome.execution.error_message
    assert outcome.sql.endswith(f"LIMIT {DEFAULT_MAX_ROWS + 1}")
    assert outcome.execution.row_count == DEFAULT_MAX_ROWS
    assert len(outcome.execution.rows) == DEFAULT_MAX_ROWS
    assert outcome.execution.truncated is True


def test_a_smaller_guardrail_cap_carries_through_to_the_executor(live_database) -> None:
    """Without an executor config the two caps agree, so the extra row is not data."""
    outcome, _ = _run("SELECT order_id FROM orders", guardrail_config=GuardrailConfig(max_rows=10))

    assert outcome.sql.endswith("LIMIT 11")
    assert outcome.execution.row_count == 10
    assert outcome.execution.truncated


def test_the_pipeline_reports_the_cost_of_the_call(live_database) -> None:
    outcome, _ = _run("SELECT count(*) AS n FROM orders")
    assert outcome.call is not None
    assert outcome.call.cost_usd > 0


# ------------------------------------------------------------------- ambiguity


def test_an_ambiguous_question_returns_clarification_unchanged() -> None:
    """Needs no database: nothing is guarded or executed, by design."""
    outcome, _ = _run(_ambiguous_answer())

    assert isinstance(outcome, ClarificationNeeded)
    assert outcome.question == "how many orders are cancelled?"
    assert [i.label for i in outcome.interpretations] == ["gross", "net"]
    assert not hasattr(outcome, "execution"), "a clarification must not carry a result"


def test_an_ambiguous_question_executes_nothing(monkeypatch) -> None:
    """The executor must not be reached at all when no reading was chosen."""
    called: list[str] = []
    monkeypatch.setattr(
        "queryguard.pipeline.execute", lambda *a, **k: called.append("executed")
    )
    _run(_ambiguous_answer())
    assert called == []


# -------------------------------------------------------------- guardrail stops


def test_a_blocked_query_is_never_executed(monkeypatch) -> None:
    """A guardrail refusal ends the run; the executor is not consulted."""
    called: list[str] = []
    monkeypatch.setattr(
        "queryguard.pipeline.execute", lambda *a, **k: called.append("executed")
    )

    outcome, _ = _run("SELECT order_id FROM orders LIMIT 999999")

    assert isinstance(outcome, PipelineResult)
    assert not outcome.guardrail.allowed
    assert outcome.guardrail.rule == "row_limit"
    assert outcome.execution is None
    assert not outcome.ok
    assert called == [], "a blocked query reached the executor"


def test_the_guardrail_catches_what_phase_one_validation_could_not() -> None:
    """A data-modifying CTE satisfies GeneratedSQL's validator and is still a write.

    `_must_be_a_single_select` checks the leading keyword, and this leads with
    WITH. The guardrail is the layer that reads the whole statement.
    """
    write_disguised_as_a_read = (
        "WITH gone AS (DELETE FROM orders RETURNING *) SELECT count(*) FROM gone"
    )
    # It passes the Phase 1 validator, which is the premise of this test.
    assert _answer(write_disguised_as_a_read).sql == write_disguised_as_a_read

    outcome, _ = _run(write_disguised_as_a_read)
    assert outcome.guardrail.rule == "forbidden_construct"
    assert outcome.execution is None


def test_a_stricter_guardrail_config_is_honoured() -> None:
    outcome, _ = _run(
        "SELECT order_id FROM orders", guardrail_config=GuardrailConfig(auto_limit=False)
    )
    assert outcome.guardrail.rule == "row_limit"
    assert outcome.execution is None


# --------------------------------------------------------- execution not fatal


def test_a_database_error_is_reported_not_raised(live_database) -> None:
    """Valid SQL against a table that does not exist: a result, not a traceback."""
    outcome, _ = _run("SELECT * FROM table_that_does_not_exist")

    assert outcome.guardrail.allowed, "the guardrail has no opinion on missing tables"
    assert outcome.execution.outcome == OUTCOME_FAILED
    assert outcome.execution.sqlstate == "42P01"
    assert not outcome.ok


def test_an_executor_refusal_surfaces_through_the_pipeline(live_database) -> None:
    outcome, _ = _run(
        "SELECT order_id FROM orders LIMIT 100",
        executor_config=ExecutorConfig(max_estimated_rows=10),
    )
    assert outcome.execution.outcome == OUTCOME_REFUSED
    assert not outcome.ok
    assert "the query was not run" in outcome.execution.reason


def test_a_pipeline_run_is_logged_by_both_halves(tmp_path, live_database) -> None:
    """One LLM line and one execution line, in separate logs."""
    _run("SELECT count(*) AS n FROM orders")

    llm_log = tmp_path / "llm_calls.jsonl"
    execution_log = tmp_path / "executions.jsonl"
    assert len(llm_log.read_text().splitlines()) == 1
    assert len(execution_log.read_text().splitlines()) == 1


# ------------------------------------------------------------- sanity flags


def test_a_result_carries_its_sanity_flags(monkeypatch) -> None:
    """No database: the executor is replaced by a result that came back empty."""
    empty = ExecutionResult(
        outcome=OUTCOME_OK,
        sql_sha256="0" * 64,
        rows=pd.DataFrame({"order_id": pd.Series([], dtype="int64")}),
    )
    monkeypatch.setattr("queryguard.pipeline.execute", lambda *a, **k: empty)

    outcome, _ = _run("SELECT order_id FROM orders WHERE status = 'Cancelled'")

    assert outcome.ok, "a flag is advice; the run itself still succeeded"
    [flag] = outcome.sanity
    assert flag.check == CHECK_EMPTY
    assert flag.severity == WARN


def test_a_blocked_query_has_no_sanity_flags() -> None:
    outcome, _ = _run("SELECT order_id FROM orders LIMIT 999999")
    assert outcome.execution is None
    assert outcome.sanity == ()


def test_sanity_flags_come_from_a_real_execution(live_database) -> None:
    """The status filter is miscased, so the database really does return nothing."""
    outcome, _ = _run("SELECT order_id FROM orders WHERE status = 'Cancelled'")

    assert outcome.ok, outcome.execution.error_message
    assert outcome.execution.row_count == 0
    assert [f.check for f in outcome.sanity] == [CHECK_EMPTY]
