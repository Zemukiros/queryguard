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

A PipelineResult therefore carries the guardrail outcome even on success: the
SQL that executed is frequently not the SQL the model wrote -- a missing LIMIT
is added on the way through -- and a result that showed only one of the two
would misreport what happened.

Usage:  uv run python -m queryguard.pipeline "how many orders last quarter?"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from queryguard.executor import ExecutionResult, ExecutorConfig, execute
from queryguard.generate import ClarificationNeeded, GeneratedSQL, generate_sql_with_stats
from queryguard.guardrails import GuardrailConfig, GuardrailResult, check
from queryguard.llm.client import CallResult, LLMClient, RequestCapExceeded


@dataclass(frozen=True)
class PipelineResult:
    """A question that produced one query, and what happened to it."""

    question: str
    answer: GeneratedSQL
    guardrail: GuardrailResult
    execution: ExecutionResult | None = None
    call: CallResult | None = None

    @property
    def ok(self) -> bool:
        """True only when the query was cleared and ran."""
        return self.execution is not None and self.execution.ok

    @property
    def sql(self) -> str:
        """The SQL that ran, or would have -- the rewritten form when there is one."""
        return self.guardrail.rewritten_sql or self.answer.sql


def run_question(
    question: str,
    *,
    client: LLMClient | None = None,
    schema: Any = None,
    guardrail_config: GuardrailConfig | None = None,
    executor_config: ExecutorConfig | None = None,
) -> PipelineResult | ClarificationNeeded:
    """Generate SQL for a question, guard it, and run it read-only.

    `client` exists so tests can inject a fake and never reach the API.
    """
    answer, call = generate_sql_with_stats(question, client=client, schema=schema)

    if isinstance(answer, ClarificationNeeded):
        return answer

    guardrail = check(answer.sql, guardrail_config)
    if not guardrail.allowed:
        # Deliberately no execution attempt. The guardrail is the gate, not a
        # warning, so a refusal ends the run here.
        return PipelineResult(
            question=question, answer=answer, guardrail=guardrail, call=call
        )

    execution = execute(guardrail.sql_to_execute, executor_config)
    return PipelineResult(
        question=question,
        answer=answer,
        guardrail=guardrail,
        execution=execution,
        call=call,
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
