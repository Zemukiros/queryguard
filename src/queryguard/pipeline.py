"""Question in, answered rows out: generate, guard, execute.

The three phases stay separate types rather than collapsing into one, because
each can end the run for a different reason and a caller needs to tell those
apart:

- The question had more than one defensible reading. Phase 1's
  ClarificationNeeded is returned *unchanged* -- no SQL was chosen, so there is
  nothing to guard or execute, and wrapping it would invite a caller to go
  looking for a `.sql` that does not exist.
- The model produced SQL the guardrail refused. There is a query to show and a
  named rule to explain it, but nothing ran.
- The query ran, or the database refused it.

A query that ran also carries sanity flags: shapes in its rows -- an empty
result, a column that is mostly NULL, a sum no un-joined table could reach --
that suggest the SQL answered a different question from the one asked. They are
advice, not a gate, so the rows are returned either way.

A PipelineResult therefore carries the guardrail outcome even on success: the
SQL that executed is frequently not the SQL the model wrote -- a missing LIMIT
is added on the way through -- and a result that showed only one of the two
would misreport what happened.

Usage:  uv run python -m queryguard.pipeline "how many orders last quarter?"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace
from typing import Any

import anthropic

from queryguard.executor import ExecutionResult, ExecutorConfig, execute
from queryguard.generate import ClarificationNeeded, GeneratedSQL, generate_sql_with_stats
from queryguard.guardrails import GuardrailConfig, GuardrailResult, check
from queryguard.llm.client import CallResult, LLMClient, RequestCapExceeded
from queryguard.schema.introspect import load_schema
from queryguard.validation.agreement import AgreementResult, check_agreement, is_non_trivial
from queryguard.validation.backtranslate import VALIDATION_MODEL, back_translate, judge_alignment
from queryguard.validation.confidence import Features, build_features, log_features, score
from queryguard.validation.sanity import SanityFlag, check_result


# Generate, back-translate, judge, second SQL. One call per step, so a
# question can only exceed this through a bug -- which is the point of the
# check. Separate from the process-wide cap in llm/client.py, which bounds a
# runaway loop across questions rather than the cost of any one of them.
MAX_CALLS_PER_QUESTION = 4


class QuestionBudgetExceeded(RuntimeError):
    """A single question tried to make more API calls than it is allowed."""


class QuestionBudget:
    """Counts API calls for one question and refuses the one past the limit."""

    def __init__(self, limit: int = MAX_CALLS_PER_QUESTION) -> None:
        self.limit = limit
        self.spent: list[str] = []

    def spend(self, step: str) -> None:
        """Call before the API request, so a refusal costs nothing."""
        if len(self.spent) >= self.limit:
            raise QuestionBudgetExceeded(
                f"per-question budget of {self.limit} calls reached "
                f"({', '.join(self.spent)}); refused {step}"
            )
        self.spent.append(step)


@dataclass(frozen=True)
class PipelineResult:
    """A question that produced one query, and what happened to it."""

    question: str
    answer: GeneratedSQL
    guardrail: GuardrailResult
    execution: ExecutionResult | None = None
    call: CallResult | None = None
    sanity: tuple[SanityFlag, ...] = ()

    # Hallucination detection. None / empty when the step did not run: the
    # query was blocked or failed, validation was switched off, or a call
    # failed (see validation_errors).
    back_translation: str | None = None
    alignment: float | None = None
    discrepancies: tuple[str, ...] = ()
    agreement: AgreementResult | None = None
    validation_calls: tuple[CallResult, ...] = ()
    validation_errors: tuple[str, ...] = ()

    # Confidence. v0 weights are hand-set and temporary: see confidence.py.
    features: Features | None = None
    confidence: float | None = None
    confidence_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True only when the query was cleared and ran."""
        return self.execution is not None and self.execution.ok

    @property
    def sql(self) -> str:
        """The SQL that ran, or would have -- the rewritten form when there is one."""
        return self.guardrail.rewritten_sql or self.answer.sql

    @property
    def cost_usd(self) -> float:
        """Every API call this question made, generation included."""
        calls = ([self.call] if self.call else []) + list(self.validation_calls)
        return sum(c.cost_usd for c in calls)


# A validation step that fails must not take the executed answer down with it.
# These are the failures a step can have that are not bugs in this code.
_VALIDATION_FAILURES = (
    RequestCapExceeded,
    QuestionBudgetExceeded,
    anthropic.APIError,
    ValueError,  # includes pydantic's ValidationError on a malformed response
)


def run_question(
    question: str,
    *,
    client: LLMClient | None = None,
    validation_client: LLMClient | None = None,
    schema: Any = None,
    guardrail_config: GuardrailConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    validate: bool = True,
) -> PipelineResult | ClarificationNeeded:
    """Generate SQL for a question, guard it, run it read-only, and check it.

    `client` (Sonnet: generation and the second query) and `validation_client`
    (Haiku: back-translation and judging) exist so tests can inject fakes and
    never reach the API.
    """
    # Resolved once here because several steps need it: the prompts are built
    # from it and the sanity checks read its profile.
    if schema is None:
        schema = load_schema()
    client = client or LLMClient()
    budget = QuestionBudget()

    budget.spend("generate")
    answer, call = generate_sql_with_stats(question, client=client, schema=schema)

    if isinstance(answer, ClarificationNeeded):
        return answer

    return run_answer(
        question,
        answer,
        call=call,
        client=client,
        validation_client=validation_client,
        schema=schema,
        guardrail_config=guardrail_config,
        executor_config=executor_config,
        validate=validate,
        budget=budget,
    )


def run_answer(
    question: str,
    answer: GeneratedSQL,
    *,
    call: CallResult | None = None,
    client: LLMClient | None = None,
    validation_client: LLMClient | None = None,
    schema: Any = None,
    guardrail_config: GuardrailConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    validate: bool = True,
    budget: QuestionBudget | None = None,
) -> PipelineResult:
    """Guard, execute and validate an answer that already exists.

    The second half of run_question, callable on its own so a known answer --
    including a deliberately wrong one, to test the detectors -- goes through
    exactly the same checks as a generated one.
    """
    if schema is None:
        schema = load_schema()
    budget = budget or QuestionBudget()

    guardrail_config = guardrail_config or GuardrailConfig()
    guardrail = check(answer.sql, guardrail_config)
    if not guardrail.allowed:
        # Deliberately no execution attempt. The guardrail is the gate, not a
        # warning, so a refusal ends the run here -- and with nothing executed
        # there is nothing to validate.
        return _scored(
            PipelineResult(question=question, answer=answer, guardrail=guardrail, call=call)
        )

    # The guardrail caps at max_rows + 1 so the executor can see overflow; the
    # two caps have to agree or that extra row comes back as data.
    if executor_config is None:
        executor_config = ExecutorConfig(max_rows=guardrail_config.max_rows)
    execution = execute(guardrail.sql_to_execute, executor_config)
    sanity = tuple(check_result(question, guardrail.sql_to_execute, execution, schema))
    result = PipelineResult(
        question=question,
        answer=answer,
        guardrail=guardrail,
        execution=execution,
        call=call,
        sanity=sanity,
    )
    if not (validate and execution.ok):
        return _scored(result)

    return _scored(
        _validate(
            result,
            client=client or LLMClient(),
            validation_client=validation_client or LLMClient(model=VALIDATION_MODEL),
            schema=schema,
            guardrail_config=guardrail_config,
            executor_config=executor_config,
            budget=budget,
        )
    )


def _validate(
    result: PipelineResult,
    *,
    client: LLMClient,
    validation_client: LLMClient,
    schema: Any,
    guardrail_config: GuardrailConfig,
    executor_config: ExecutorConfig,
    budget: QuestionBudget,
) -> PipelineResult:
    """Back-translate, judge, and (for non-trivial SQL) get a second opinion.

    Each step is independent of the others' success, so one failing call costs
    that signal and nothing else.
    """
    calls: list[CallResult] = []
    errors: list[str] = []
    back_translation: str | None = None
    alignment: float | None = None
    discrepancies: tuple[str, ...] = ()
    agreement: AgreementResult | None = None

    # The model's SQL, not the executed form: the guardrail's LIMIT 1001 is a
    # safety cap, and back-translating it would describe "the first 1001 ...".
    sql = result.answer.sql

    try:
        budget.spend("back_translate")
        translated, bt_call = back_translate(sql, client=validation_client, schema=schema)
        calls.append(bt_call)
        back_translation = translated.question

        budget.spend("judge")
        judgement, judge_call = judge_alignment(
            result.question, translated, client=validation_client
        )
        calls.append(judge_call)
        alignment = judgement.alignment
        discrepancies = tuple(judgement.discrepancies)
    except _VALIDATION_FAILURES as exc:
        errors.append(f"alignment: {type(exc).__name__}: {exc}")

    if is_non_trivial(sql):
        try:
            budget.spend("second_sql")
            agreement = check_agreement(
                result.question,
                sql,
                result.execution,
                client=client,
                schema=schema,
                guardrail_config=guardrail_config,
                executor_config=executor_config,
            )
            if agreement.call is not None:
                calls.append(agreement.call)
        except _VALIDATION_FAILURES as exc:
            errors.append(f"agreement: {type(exc).__name__}: {exc}")

    return replace(
        result,
        back_translation=back_translation,
        alignment=alignment,
        discrepancies=discrepancies,
        agreement=agreement,
        validation_calls=tuple(calls),
        validation_errors=tuple(errors),
    )


def _scored(result: PipelineResult) -> PipelineResult:
    """Attach features and confidence, and append the training row."""
    execution = result.execution
    features = build_features(
        executed=result.ok,
        self_confidence=result.answer.confidence,
        alignment=result.alignment,
        discrepancies=result.discrepancies,
        sanity=result.sanity,
        agreement=result.agreement.outcome if result.agreement else None,
        guardrail_rewrote=result.guardrail.rewritten_sql is not None,
        row_count=execution.row_count if execution else 0,
        truncated=execution.truncated if execution else False,
    )
    confidence, breakdown = score(features)
    log_features(
        question=result.question,
        sql=result.answer.sql,
        features=features,
        confidence=confidence,
        breakdown=breakdown,
    )
    return replace(
        result, features=features, confidence=confidence, confidence_breakdown=breakdown
    )


# ---------------------------------------------------------------------- cli


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CAP_EXCEEDED = 2
EXIT_CLARIFICATION_NEEDED = 3
EXIT_BLOCKED = 4


def _render(result: PipelineResult) -> str:
    lines = [result.sql, ""]

    if not result.guardrail.allowed:
        lines.append(f"BLOCKED by {result.guardrail.rule}: {result.guardrail.reason}")
        return "\n".join(lines)

    execution = result.execution
    assert execution is not None  # allowed implies an execution attempt

    if execution.outcome == "refused":
        lines.append(f"REFUSED: {execution.reason}")
    elif not execution.ok:
        lines.append(
            f"FAILED [{execution.sqlstate} {execution.error_class}]: {execution.error_message}"
        )
    else:
        lines.append(execution.rows.to_string(index=False) if execution.row_count else "(no rows)")
        note = f"\n{execution.row_count} rows in {execution.execution_ms} ms"
        if execution.truncated:
            note += " (truncated)"
        if result.guardrail.rewritten_sql:
            note += " · LIMIT added by the guardrail"
        lines.append(note)
        for flag in result.sanity:
            lines.append(f"{flag.severity.upper()} [{flag.check}] {flag.explanation}")
        if result.back_translation is not None:
            lines.append(f"\nSQL answers: {result.back_translation}")
        if result.alignment is not None:
            lines.append(f"alignment {result.alignment:.2f}")
        for discrepancy in result.discrepancies:
            lines.append(f"  - {discrepancy}")
        if result.agreement is not None:
            lines.append(f"second query: {result.agreement.outcome} ({result.agreement.explanation})")
        for error in result.validation_errors:
            lines.append(f"validation step failed: {error}")
        if result.confidence is not None:
            lines.append(
                f"confidence {result.confidence:.2f} (v0 hand-set weights, uncalibrated)"
                f" · ${result.cost_usd:.4f} across {1 + len(result.validation_calls)} calls"
            )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m queryguard.pipeline",
        description="Ask a question in English; get rows from the read-only role.",
        epilog=(
            "exit codes: 0 ok · 1 error · 2 request cap reached · "
            "3 clarification needed · 4 blocked by a guardrail"
        ),
    )
    parser.add_argument("question", help="the question to answer")
    args = parser.parse_args(argv)

    try:
        outcome = run_question(args.question)
    except RequestCapExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CAP_EXCEEDED
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if isinstance(outcome, ClarificationNeeded):
        print(f"That question has {len(outcome.interpretations)} defensible readings:\n")
        for interpretation in outcome.interpretations:
            print(f"[{interpretation.label}] {interpretation.explanation}\n{interpretation.sql}\n")
        return EXIT_CLARIFICATION_NEEDED

    print(_render(outcome))
    if not outcome.guardrail.allowed:
        return EXIT_BLOCKED
    return EXIT_OK if outcome.ok else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
