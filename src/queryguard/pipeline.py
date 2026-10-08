"""Question in, answered rows out: generate, guard, execute.

The three phases stay separate types rather than collapsing into one, because
each can end the run for a different reason and a caller needs to tell those
apart:

- The question had more than one defensible reading. Phase 1's
  ClarificationNeeded is returned *unchanged* -- no SQL was chosen, so there is
  nothing to guard or execute, and wrapping it would invite a caller to go
  looking for a `.sql` that does not exist.
- Nothing in the schema can answer it. CannotAnswer is returned unchanged, for
  the same reason.
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

There is one code path. `stream_question` is an async generator that yields a
typed StageEvent as each stage finishes (sequence and payloads: events.py), and
the synchronous `run_question` / `run_answer` are thin consumers that drain it
and return the outcome from the `done` event -- or re-raise what an `error`
event carries. The blocking SDK and database calls run via asyncio.to_thread,
so one slow question does not stall an event loop serving others.

Usage:  uv run python -m queryguard.pipeline "how many orders last quarter?"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any


from queryguard.events import (
    AgreementPayload,
    BacktranslatePayload,
    ClarificationPayload,
    ColumnModel,
    ConfidencePayload,
    Contribution,
    ErrorPayload,
    ExecutingPayload,
    GeneratingPayload,
    GuardrailsPayload,
    QueryResult,
    SanityFlagModel,
    SanityPayload,
    Stage,
    StageEvent,
    json_cell,
)
from queryguard.executor import OUTCOME_REFUSED, ExecutionResult, ExecutorConfig, execute
from queryguard.generate import (
    CannotAnswer,
    ClarificationNeeded,
    GeneratedSQL,
    generate_sql_with_stats,
)
from queryguard.guardrails import GuardrailConfig, GuardrailResult, check
from queryguard.llm.client import (
    MAX_CALLS_PER_QUESTION,
    CallResult,
    LLMClient,
    RequestCapExceeded,
    sdk_error_types,
)
from queryguard.schema.introspect import load_schema
from queryguard.validation.agreement import AgreementResult, check_agreement, is_non_trivial
from queryguard.validation.backtranslate import VALIDATION_MODEL, back_translate, judge_alignment
from queryguard.validation.confidence import (
    SCORER_VERSION,
    Features,
    band,
    build_features,
    contributions,
    log_features,
    score,
)
from queryguard.validation.sanity import SanityFlag, check_result


# MAX_CALLS_PER_QUESTION (llm/client.py): generate, back-translate, judge,
# second SQL. One call per step, so a question can only exceed it through a
# bug -- which is the point of the check. Defined beside the other call caps so
# the API can read it without importing the pipeline.


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

    # Confidence: see confidence.py for where the weights come from.
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
# These are the failures a step can have that are not bugs in this code. A
# function, not a tuple: `except` evaluates it only when something is raised,
# so the SDK's error type is looked up without importing the SDK (see
# llm.client.sdk_error_types).
def _validation_failures() -> tuple[type[BaseException], ...]:
    return (
        RequestCapExceeded,
        QuestionBudgetExceeded,
        *sdk_error_types(),
        ValueError,  # includes pydantic's ValidationError on a malformed response
    )


async def stream_question(
    question: str,
    *,
    client: LLMClient | None = None,
    validation_client: LLMClient | None = None,
    schema: Any = None,
    guardrail_config: GuardrailConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    validate: bool = True,
    query_id: str | None = None,
) -> AsyncIterator[StageEvent]:
    """Generate SQL for a question, guard it, run it read-only, check it: one event per stage.

    `client` (Sonnet: generation and the second query) and `validation_client`
    (Haiku: back-translation and judging) exist so tests can inject fakes and
    never reach the API. The event sequence is documented in events.py.
    """
    run = _Run(query_id)
    stages = _question_stages(
        run, question, client=client, validation_client=validation_client, schema=schema,
        guardrail_config=guardrail_config, executor_config=executor_config, validate=validate,
    )
    async for event in _guarded(run, stages):
        yield event


async def stream_answer(
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
    query_id: str | None = None,
) -> AsyncIterator[StageEvent]:
    """Guard, execute and validate an answer that already exists: the stream from `guardrails` on.

    The second half of stream_question, callable on its own so a known answer --
    including a deliberately wrong one, to test the detectors -- goes through
    exactly the same checks as a generated one.
    """
    run = _Run(query_id)
    stages = _answer_stages(
        run, question, answer, call=call, client=client, validation_client=validation_client,
        schema=schema, guardrail_config=guardrail_config, executor_config=executor_config,
        validate=validate, budget=budget,
    )
    async for event in _guarded(run, stages):
        yield event


def run_question(
    question: str,
    *,
    client: LLMClient | None = None,
    validation_client: LLMClient | None = None,
    schema: Any = None,
    guardrail_config: GuardrailConfig | None = None,
    executor_config: ExecutorConfig | None = None,
    validate: bool = True,
) -> PipelineResult | ClarificationNeeded | CannotAnswer:
    """stream_question, drained: the final outcome, or the exception a stage raised.

    Uses asyncio.run, so call it from synchronous code only; async callers
    iterate stream_question themselves.
    """
    return _drain(
        stream_question(
            question, client=client, validation_client=validation_client, schema=schema,
            guardrail_config=guardrail_config, executor_config=executor_config, validate=validate,
        )
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
    """stream_answer, drained. Same constraints as run_question."""
    return _drain(
        stream_answer(
            question, answer, call=call, client=client, validation_client=validation_client,
            schema=schema, guardrail_config=guardrail_config, executor_config=executor_config,
            validate=validate, budget=budget,
        )
    )


def _drain(stream: AsyncIterator[StageEvent]) -> Any:
    async def consume() -> Any:
        async for event in stream:
            if event.exception is not None:
                raise event.exception
            if event.stage == "done":
                return event.result
        raise RuntimeError("pipeline stream ended without a done event")

    return asyncio.run(consume())


# ------------------------------------------------------------------ the stages


class _Run:
    """One question's clock, id, and the stage currently in progress."""

    def __init__(self, query_id: str | None) -> None:
        self.query_id = query_id or uuid.uuid4().hex
        self.started = self._mark = time.perf_counter()
        self.stage: Stage = "generating"

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def event(self, payload: Any) -> StageEvent:
        now = time.perf_counter()
        event = StageEvent(
            elapsed_ms=int((now - self.started) * 1000),
            duration_ms=int((now - self._mark) * 1000),
            payload=payload,
        )
        self._mark = now
        return event

    def done(self, outcome: Any, call: CallResult | None = None) -> StageEvent:
        self.stage = "done"
        event = self.event(
            to_query_result(outcome, query_id=self.query_id, elapsed_ms=self.elapsed_ms(), call=call)
        )
        event._outcome = outcome
        return event


async def _guarded(run: _Run, stages: AsyncIterator[StageEvent]) -> AsyncIterator[StageEvent]:
    """Turn an exception in any stage into a final `error` event that carries it."""
    try:
        async for event in stages:
            yield event
    except Exception as exc:  # noqa: BLE001 - reported as an event; _drain re-raises it
        event = run.event(
            ErrorPayload(failed_stage=run.stage, error_type=type(exc).__name__, message=str(exc))
        )
        event._exception = exc
        yield event


async def _question_stages(
    run: _Run,
    question: str,
    *,
    client: LLMClient | None,
    validation_client: LLMClient | None,
    schema: Any,
    guardrail_config: GuardrailConfig | None,
    executor_config: ExecutorConfig | None,
    validate: bool,
) -> AsyncIterator[StageEvent]:
    # Resolved once here because several steps need it: the prompts are built
    # from it and the sanity checks read its profile.
    if schema is None:
        schema = await asyncio.to_thread(load_schema)
    client = client or LLMClient()
    budget = QuestionBudget()

    budget.spend("generate")
    answer, call = await asyncio.to_thread(
        generate_sql_with_stats, question, client=client, schema=schema
    )
    yield run.event(_generating_payload(answer, call))

    if isinstance(answer, ClarificationNeeded):
        run.stage = "clarification"
        yield run.event(ClarificationPayload(interpretations=answer.interpretations))
    if isinstance(answer, (ClarificationNeeded, CannotAnswer)):
        yield run.done(answer, call)
        return

    async for event in _answer_stages(
        run, question, answer, call=call, client=client, validation_client=validation_client,
        schema=schema, guardrail_config=guardrail_config, executor_config=executor_config,
        validate=validate, budget=budget,
    ):
        yield event


async def _answer_stages(
    run: _Run,
    question: str,
    answer: GeneratedSQL,
    *,
    call: CallResult | None,
    client: LLMClient | None,
    validation_client: LLMClient | None,
    schema: Any,
    guardrail_config: GuardrailConfig | None,
    executor_config: ExecutorConfig | None,
    validate: bool,
    budget: QuestionBudget | None,
) -> AsyncIterator[StageEvent]:
    if schema is None:
        schema = await asyncio.to_thread(load_schema)
    budget = budget or QuestionBudget()
    guardrail_config = guardrail_config or GuardrailConfig()

    run.stage = "guardrails"
    guardrail = check(answer.sql, guardrail_config)  # pure parsing; no I/O to offload
    yield run.event(
        GuardrailsPayload(
            allowed=guardrail.allowed, rule=guardrail.rule, reason=guardrail.reason,
            rewritten_sql=guardrail.rewritten_sql,
            sql_to_execute=guardrail.sql_to_execute if guardrail.allowed else None,
        )
    )
    result = PipelineResult(question=question, answer=answer, guardrail=guardrail, call=call)

    # A refusal ends the run here: the guardrail is the gate, not a warning, and
    # with nothing executed there is nothing to validate.
    if guardrail.allowed:
        # The guardrail caps at max_rows + 1 so the executor can see overflow;
        # the two caps have to agree or that extra row comes back as data.
        if executor_config is None:
            executor_config = ExecutorConfig(max_rows=guardrail_config.max_rows)

        run.stage = "executing"
        execution = await asyncio.to_thread(execute, guardrail.sql_to_execute, executor_config)
        yield run.event(_executing_payload(execution))

        run.stage = "sanity"
        sanity = tuple(
            await asyncio.to_thread(check_result, question, guardrail.sql_to_execute, execution, schema)
        )
        yield run.event(SanityPayload(flags=[_flag_model(f) for f in sanity]))
        result = replace(result, execution=execution, sanity=sanity)

        if validate and execution.ok:
            async for event, result in _validation_stages(
                run, result,
                client=client or LLMClient(),
                validation_client=validation_client or LLMClient(model=VALIDATION_MODEL),
                schema=schema, guardrail_config=guardrail_config,
                executor_config=executor_config, budget=budget,
            ):
                yield event

    run.stage = "confidence"
    result = await asyncio.to_thread(_scored, result)
    terms = _contributions(result)
    yield run.event(
        ConfidencePayload(
            confidence=result.confidence, band=band(result.confidence),
            logit=sum(t.contribution for t in terms) if terms else None, contributions=terms,
            breakdown=result.confidence_breakdown, scorer_version=SCORER_VERSION,
        )
    )
    yield run.done(result)


async def _validation_stages(
    run: _Run,
    result: PipelineResult,
    *,
    client: LLMClient,
    validation_client: LLMClient,
    schema: Any,
    guardrail_config: GuardrailConfig,
    executor_config: ExecutorConfig,
    budget: QuestionBudget,
) -> AsyncIterator[tuple[StageEvent, PipelineResult]]:
    """Back-translate, judge, and (for non-trivial SQL) get a second opinion.

    Yields each event with the result as updated so far. Each step is
    independent of the others' success, so one failing call costs that signal
    and nothing else.
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

    run.stage = "backtranslate"
    error = None
    try:
        budget.spend("back_translate")
        translated, bt_call = await asyncio.to_thread(
            back_translate, sql, client=validation_client, schema=schema
        )
        calls.append(bt_call)
        back_translation = translated.question

        budget.spend("judge")
        judgement, judge_call = await asyncio.to_thread(
            judge_alignment, result.question, translated, client=validation_client
        )
        calls.append(judge_call)
        alignment = judgement.alignment
        discrepancies = tuple(judgement.discrepancies)
    except _validation_failures() as exc:
        error = f"alignment: {type(exc).__name__}: {exc}"
        errors.append(error)
    result = replace(
        result, back_translation=back_translation, alignment=alignment,
        discrepancies=discrepancies, validation_calls=tuple(calls), validation_errors=tuple(errors),
    )
    yield run.event(
        BacktranslatePayload(
            back_translation=back_translation, alignment=alignment,
            discrepancies=list(discrepancies), error=error,
        )
    ), result

    run.stage = "agreement"
    if not is_non_trivial(sql):
        payload = AgreementPayload(
            ran=False, skipped_reason="no join, aggregate, CTE, subquery or top-N: nothing for a second query to disagree on"
        )
    else:
        try:
            budget.spend("second_sql")
            agreement = await asyncio.to_thread(
                check_agreement, result.question, sql, result.execution, client=client,
                schema=schema, guardrail_config=guardrail_config, executor_config=executor_config,
            )
            if agreement.call is not None:
                calls.append(agreement.call)
            payload = AgreementPayload(
                ran=True, outcome=agreement.outcome, explanation=agreement.explanation,
                second_sql=agreement.second_sql,
            )
        except _validation_failures() as exc:
            error = f"agreement: {type(exc).__name__}: {exc}"
            errors.append(error)
            payload = AgreementPayload(ran=True, error=error)
    result = replace(
        result, agreement=agreement, validation_calls=tuple(calls), validation_errors=tuple(errors)
    )
    yield run.event(payload), result


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


# --------------------------------------------------------------- serialising


def _generating_payload(
    answer: GeneratedSQL | ClarificationNeeded | CannotAnswer, call: CallResult
) -> GeneratingPayload:
    # The parsed model output holds explanation and self-confidence for every
    # kind; the typed outcome only says which kind it was.
    parsed: GeneratedSQL = call.parsed
    kind = (
        "clarification" if isinstance(answer, ClarificationNeeded)
        else "cannot_answer" if isinstance(answer, CannotAnswer)
        else "sql"
    )
    return GeneratingPayload(
        kind=kind, sql=parsed.sql or None, explanation=parsed.explanation,
        self_confidence=parsed.confidence, assumptions=parsed.assumptions,
        tables_used=parsed.tables_used, cost_usd=call.cost_usd, latency_ms=call.latency_ms,
    )


def _executing_payload(execution: ExecutionResult) -> ExecutingPayload:
    return ExecutingPayload(
        outcome=execution.outcome, row_count=execution.row_count, truncated=execution.truncated,
        execution_ms=execution.execution_ms,
        columns=[ColumnModel(name=c.name, dtype=c.dtype) for c in execution.columns],
        estimated_rows=execution.estimated_rows, reason=execution.reason,
        error_class=execution.error_class, error_message=execution.error_message,
        sqlstate=execution.sqlstate,
    )


def _contributions(result: PipelineResult) -> list[Contribution]:
    return [Contribution(**term) for term in contributions(result.features)] if result.features else []


def _flag_model(flag: SanityFlag) -> SanityFlagModel:
    return SanityFlagModel(
        check=flag.check, severity=flag.severity, explanation=flag.explanation, column=flag.column
    )


def to_query_result(
    outcome: PipelineResult | ClarificationNeeded | CannotAnswer,
    *,
    query_id: str,
    elapsed_ms: int,
    call: CallResult | None = None,
) -> QueryResult:
    """The JSON-safe summary of any outcome: the `done` payload."""
    if isinstance(outcome, ClarificationNeeded | CannotAnswer):
        parsed = call.parsed if call else None
        return QueryResult(
            query_id=query_id, question=outcome.question,
            outcome="clarification" if isinstance(outcome, ClarificationNeeded) else "cannot_answer",
            explanation=parsed.explanation if parsed else None,
            assumptions=parsed.assumptions if parsed else [],
            interpretations=outcome.interpretations if isinstance(outcome, ClarificationNeeded) else [],
            cannot_answer_reason=outcome.explanation if isinstance(outcome, CannotAnswer) else None,
            n_calls=1 if call else 0, cost_usd=call.cost_usd if call else 0.0, elapsed_ms=elapsed_ms,
        )

    execution = outcome.execution
    if not outcome.guardrail.allowed:
        kind = "blocked"
    elif execution.ok:
        kind = "answered"
    else:
        kind = "refused" if execution.outcome == OUTCOME_REFUSED else "failed"

    rows: list[list[Any]] = []
    if execution is not None and execution.ok:
        rows = [[json_cell(v) for v in row] for row in execution.rows.itertuples(index=False, name=None)]
    agreement = outcome.agreement
    terms = _contributions(outcome)
    return QueryResult(
        query_id=query_id, question=outcome.question, outcome=kind,
        sql=outcome.answer.sql, executed_sql=outcome.sql if outcome.guardrail.allowed else None,
        explanation=outcome.answer.explanation, assumptions=outcome.answer.assumptions,
        columns=execution.column_names if execution else [], rows=rows,
        row_count=execution.row_count if execution else 0,
        truncated=execution.truncated if execution else False,
        execution_ms=execution.execution_ms if execution else None,
        guardrail_rule=outcome.guardrail.rule, guardrail_reason=outcome.guardrail.reason,
        execution_error=(execution.reason or execution.error_message) if execution and not execution.ok else None,
        sanity=[_flag_model(f) for f in outcome.sanity],
        back_translation=outcome.back_translation, alignment=outcome.alignment,
        discrepancies=list(outcome.discrepancies),
        agreement=agreement.outcome if agreement else None,
        agreement_explanation=agreement.explanation if agreement else None,
        second_sql=agreement.second_sql if agreement else None,
        validation_errors=list(outcome.validation_errors),
        confidence=outcome.confidence,
        confidence_band=band(outcome.confidence) if outcome.confidence is not None else None,
        confidence_logit=sum(t.contribution for t in terms) if terms else None,
        contributions=terms, confidence_breakdown=outcome.confidence_breakdown,
        scorer_version=SCORER_VERSION if outcome.confidence is not None else None,
        n_calls=(1 if outcome.call else 0) + len(outcome.validation_calls),
        cost_usd=outcome.cost_usd, elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------- cli


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CAP_EXCEEDED = 2
EXIT_CLARIFICATION_NEEDED = 3
EXIT_BLOCKED = 4
EXIT_CANNOT_ANSWER = 5


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
                f"confidence {result.confidence:.2f} ({SCORER_VERSION})"
                f" · ${result.cost_usd:.4f} across {1 + len(result.validation_calls)} calls"
            )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m queryguard.pipeline",
        description="Ask a question in English; get rows from the read-only role.",
        epilog=(
            "exit codes: 0 ok · 1 error · 2 request cap reached · "
            "3 clarification needed · 4 blocked by a guardrail · "
            "5 cannot be answered from this schema"
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
    if isinstance(outcome, CannotAnswer):
        print(f"Cannot answer from this schema: {outcome.explanation}")
        return EXIT_CANNOT_ANSWER

    print(_render(outcome))
    if not outcome.guardrail.allowed:
        return EXIT_BLOCKED
    return EXIT_OK if outcome.ok else EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
