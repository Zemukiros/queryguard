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
from decimal import Decimal

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
from queryguard.pipeline import (
    MAX_CALLS_PER_QUESTION,
    PipelineResult,
    QuestionBudget,
    QuestionBudgetExceeded,
    run_answer,
    run_question,
)
from queryguard.schema.introspect import ColumnInfo, DatabaseSchema, TableInfo
from queryguard.validation.backtranslate import (
    VALIDATION_MODEL,
    AlignmentJudgement,
    BackTranslation,
)
from queryguard.validation.sanity import CHECK_EMPTY, WARN


@pytest.fixture(autouse=True)
def _isolate_counter():
    """Keep tests out of each other's request count. Logs are isolated in conftest."""
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
    """Answers each parse() by its output_format, so one fake serves every step.

    A list answers successive calls in order and then repeats its last item --
    the first GeneratedSQL is the generation, the second the second opinion.
    """

    def __init__(self, responses: dict) -> None:
        self._responses = {k: list(v) if isinstance(v, list) else [v] for k, v in responses.items()}
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        queue = self._responses[kwargs["output_format"]]
        parsed = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(parsed, Exception):
            raise parsed
        return FakeMessage(parsed)


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic. Records calls, never uses the network."""

    def __init__(self, parsed, *, second=None, back_translation=None, judgement=None) -> None:
        self.messages = FakeMessages(
            {
                GeneratedSQL: [parsed, second if second is not None else parsed],
                BackTranslation: back_translation
                or BackTranslation(question="How many orders are cancelled?", details=[]),
                AlignmentJudgement: judgement
                or AlignmentJudgement(alignment=0.95, discrepancies=[]),
            }
        )

    @property
    def calls(self):
        return self.messages.calls

    @property
    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]


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


def _clients(fake: FakeAnthropic) -> dict:
    return {
        "client": LLMClient(sdk_client=fake),
        "validation_client": LLMClient(sdk_client=fake, model=VALIDATION_MODEL),
    }


def _run(sql_or_answer, *, fake_kwargs=None, **kwargs):
    """Drive the pipeline with a fake that returns exactly this answer."""
    parsed = sql_or_answer if isinstance(sql_or_answer, GeneratedSQL) else _answer(sql_or_answer)
    fake = FakeAnthropic(parsed, **(fake_kwargs or {}))
    outcome = run_question(
        "how many orders are cancelled?",
        schema=_synthetic_schema(),
        **_clients(fake),
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
    # count(*) is an aggregate, so every validation step runs: the budget is spent.
    assert fake.models == [
        "claude-sonnet-5", VALIDATION_MODEL, VALIDATION_MODEL, "claude-sonnet-5"
    ]


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
    outcome = run_question("List every order item", schema=_synthetic_schema(), **_clients(fake))

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
    """Four LLM lines and two execution lines (the second opinion runs too)."""
    _run("SELECT count(*) AS n FROM orders")

    llm_log = tmp_path / "llm_calls.jsonl"
    execution_log = tmp_path / "executions.jsonl"
    assert len(llm_log.read_text().splitlines()) == 4
    assert len(execution_log.read_text().splitlines()) == 2


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


# ------------------------------------------------- hallucination and confidence


def _executor_returning(frames: dict[str, pd.DataFrame]):
    """An execute() stand-in keyed by a substring of the SQL. No database."""

    def fake_execute(sql, config=None):
        for needle, frame in frames.items():
            if needle in sql:
                return ExecutionResult(
                    outcome=OUTCOME_OK, sql_sha256="0" * 64, rows=frame, row_count=len(frame)
                )
        raise AssertionError(f"unexpected SQL: {sql}")

    return fake_execute


def _patch_execute(monkeypatch, frames) -> None:
    fake = _executor_returning(frames)
    monkeypatch.setattr("queryguard.pipeline.execute", fake)
    monkeypatch.setattr("queryguard.validation.agreement.execute", fake)


def _features_log(tmp_path) -> list[dict]:
    import json

    path = tmp_path / "confidence_features.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_an_injected_trivial_wrong_answer_is_caught_by_alignment(monkeypatch, tmp_path) -> None:
    """run_answer skips generation, so a known-wrong query gets the same checks."""
    wrong = _answer(
        "SELECT c.customer_id, c.lifetime_value FROM customers AS c "
        "ORDER BY c.lifetime_value DESC LIMIT 5"
    )
    right = _answer(
        "WITH s AS (SELECT o.customer_id, sum(o.total_amount) AS spent FROM orders AS o "
        "WHERE o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01' "
        "GROUP BY o.customer_id) SELECT s.customer_id, s.spent FROM s ORDER BY s.spent DESC LIMIT 5"
    )
    _patch_execute(monkeypatch, {
        "lifetime_value": pd.DataFrame({"customer_id": [301, 77], "lifetime_value": [Decimal("82098.49"), Decimal("80110.00")]}),
        "WITH s AS": pd.DataFrame({"customer_id": [12, 301], "spent": [Decimal("31002.10"), Decimal("29877.45")]}),
    })
    fake = FakeAnthropic(
        right,
        back_translation=BackTranslation(question="Which 5 customers have the highest lifetime value?", details=[]),
        judgement=AlignmentJudgement(alignment=0.25, discrepancies=["original asks for 2025 spend; query uses lifetime_value"]),
    )

    outcome = run_answer(
        "Which 5 customers spent the most in 2025?", wrong, schema=_synthetic_schema(), **_clients(fake)
    )

    # The wrong query has no join, aggregate, CTE or subquery, so it is trivial
    # by definition and gets no second opinion: alignment alone has to catch it.
    assert outcome.agreement is None
    assert outcome.alignment == 0.25
    assert outcome.discrepancies == ("original asks for 2025 spend; query uses lifetime_value",)
    assert outcome.back_translation == "Which 5 customers have the highest lifetime value?"
    assert outcome.confidence < 0.3
    assert outcome.call is None, "no generation call was made"
    assert len(fake.calls) == 2, "trivial SQL: back-translate and judge only"


def test_a_non_trivial_wrong_answer_also_meets_disagreement(monkeypatch) -> None:
    """Miscased status: count(*) is an aggregate, so the second opinion runs."""
    wrong = _answer("SELECT count(*) AS n FROM orders AS o WHERE o.status = 'Cancelled'")
    right = _answer("SELECT count(o.order_id) FROM orders AS o WHERE o.status = 'cancelled'")
    _patch_execute(monkeypatch, {
        "'Cancelled'": pd.DataFrame({"n": pd.Series([0], dtype="int64")}),
        "'cancelled'": pd.DataFrame({"count": pd.Series([812], dtype="int64")}),
    })
    # run_answer makes no generation call, so the only GeneratedSQL the fake
    # hands out is the second opinion.
    fake = FakeAnthropic(right)

    outcome = run_answer("How many orders were cancelled?", wrong, schema=_synthetic_schema(), **_clients(fake))

    assert outcome.agreement.outcome == "disagree"
    assert "0 vs 812" in outcome.agreement.explanation
    assert outcome.features.agreement == "disagree"
    assert outcome.confidence < 0.5
    assert len(outcome.validation_calls) == 3


def test_a_clean_run_scores_high_and_uses_exactly_the_budget(monkeypatch) -> None:
    sql = "SELECT count(*) AS n FROM orders AS o WHERE o.status = 'cancelled'"
    _patch_execute(monkeypatch, {"count": pd.DataFrame({"n": pd.Series([812], dtype="int64")})})
    outcome, fake = _run(sql)

    assert outcome.agreement.outcome == "agree"
    assert outcome.alignment == 0.95
    assert outcome.confidence > 0.85
    assert len(fake.calls) == MAX_CALLS_PER_QUESTION
    assert outcome.cost_usd == pytest.approx(outcome.call.cost_usd + sum(c.cost_usd for c in outcome.validation_calls))


def test_a_trivial_query_gets_no_second_opinion(monkeypatch) -> None:
    _patch_execute(monkeypatch, {"email": pd.DataFrame({"email": ["a@b.c"]})})
    outcome, fake = _run("SELECT c.email FROM customers AS c WHERE c.customer_id = 42")
    assert outcome.agreement is None
    assert outcome.features.agreement == "not_run"
    assert len(fake.calls) == 3


def test_validation_can_be_switched_off(monkeypatch, tmp_path) -> None:
    _patch_execute(monkeypatch, {"count": pd.DataFrame({"n": pd.Series([812], dtype="int64")})})
    outcome, fake = _run("SELECT count(*) AS n FROM orders", validate=False)
    assert len(fake.calls) == 1
    assert outcome.alignment is None and outcome.agreement is None
    assert outcome.features.alignment is None
    assert len(_features_log(tmp_path)) == 1, "an unvalidated run is still a training row"


def test_a_blocked_query_scores_zero_spends_nothing_more_and_is_logged(tmp_path) -> None:
    outcome, fake = _run("SELECT order_id FROM orders LIMIT 999999")
    assert outcome.confidence == 0.0
    assert outcome.confidence_breakdown == {}
    assert not outcome.features.executed
    assert len(fake.calls) == 1
    [row] = _features_log(tmp_path)
    assert row["confidence"] == 0.0 and row["features"]["executed"] is False


def test_a_failed_query_is_not_validated(monkeypatch) -> None:
    failed = ExecutionResult(outcome=OUTCOME_FAILED, sql_sha256="0" * 64, sqlstate="42P01")
    monkeypatch.setattr("queryguard.pipeline.execute", lambda *a, **k: failed)
    outcome, fake = _run("SELECT count(*) FROM nope")
    assert len(fake.calls) == 1
    assert outcome.confidence == 0.0


def test_a_failing_validation_call_costs_its_signal_not_the_answer(monkeypatch) -> None:
    _patch_execute(monkeypatch, {"count": pd.DataFrame({"n": pd.Series([812], dtype="int64")})})
    outcome, _ = _run(
        "SELECT count(*) AS n FROM orders",
        fake_kwargs={"back_translation": ValueError("malformed structured output")},
    )
    assert outcome.ok
    assert outcome.alignment is None
    assert outcome.validation_errors == ("alignment: ValueError: malformed structured output",)
    assert outcome.agreement.outcome == "agree", "the other step still ran"
    assert outcome.features.alignment is None


def test_the_per_question_budget_refuses_the_call_past_its_limit() -> None:
    budget = QuestionBudget(limit=2)
    budget.spend("generate")
    budget.spend("back_translate")
    with pytest.raises(QuestionBudgetExceeded, match="refused judge"):
        budget.spend("judge")


def test_the_pipeline_enforces_the_budget_before_calling(monkeypatch) -> None:
    """With one call left, the judge and second opinion are refused, not made."""
    _patch_execute(monkeypatch, {"count": pd.DataFrame({"n": pd.Series([812], dtype="int64")})})
    fake = FakeAnthropic(_answer("SELECT count(*) AS n FROM orders"))
    budget = QuestionBudget(limit=1)

    outcome = run_answer(
        "how many orders?", _answer("SELECT count(*) AS n FROM orders"),
        schema=_synthetic_schema(), budget=budget, **_clients(fake),
    )

    assert budget.spent == ["back_translate"]
    assert len(fake.calls) == 1, "nothing past the budget reached the client"
    assert outcome.alignment is None and outcome.agreement is None
    assert [e.split(":")[0] for e in outcome.validation_errors] == ["alignment", "agreement"]


def test_the_budget_is_separate_from_the_global_cap(monkeypatch) -> None:
    """The process cap still applies inside a question's budget."""
    monkeypatch.setenv("QUERYGUARD_MAX_REQUESTS", "2")
    _patch_execute(monkeypatch, {"count": pd.DataFrame({"n": pd.Series([812], dtype="int64")})})
    outcome, fake = _run("SELECT count(*) AS n FROM orders")
    assert len(fake.calls) == 2
    assert any("RequestCapExceeded" in e for e in outcome.validation_errors)
