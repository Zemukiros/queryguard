"""Turn a natural-language question into a typed SQL query.

Generation only. Nothing in this module executes the SQL it produces -- running
it against the read-only role is Phase 2's job, and keeping the two apart means
a generation bug can never touch the database.

Usage:  uv run python -m queryguard.generate "How many orders were cancelled?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from queryguard.llm.client import (
    DEFAULT_MODEL,
    CallResult,
    LLMClient,
    RequestCapExceeded,
    running_total_usd,
)
from queryguard.llm.prompt import build_system_blocks, build_user_message

# Leading line- and block-comments, so a commented preamble cannot disguise a
# non-SELECT statement.
_LEADING_COMMENTS = re.compile(r"^(?:\s*(?:--[^\n]*\n|/\*.*?\*/))*\s*", re.DOTALL)

# How many readings a clarification may offer. Two is the point; three is the
# ceiling, because a list long enough to need scrolling is not a question the
# caller can actually answer.
MIN_INTERPRETATIONS = 2
MAX_INTERPRETATIONS = 3


def _must_be_a_single_select(value: str, *, field: str) -> str:
    """Reject anything that is not a single SELECT / WITH ... SELECT.

    Shared by the top-level `sql` and by every interpretation's `sql`: an
    alternative reading is a query the caller may well choose to run, so it is
    held to exactly the same standard as a lone answer.

    This checks the leading keyword after stripping comments, so it does not
    catch a second statement smuggled in after a `;`. That gap predates this
    module's ambiguity support and closes in Phase 2, where a sqlparse layer
    guards the code path that actually executes SQL. Nothing here executes.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field} is empty")
    body = _LEADING_COMMENTS.sub("", stripped)
    if not re.match(r"(?i)(select|with)\b", body):
        raise ValueError(f"{field} must start with SELECT or WITH, got: {body[:40]!r}")
    return stripped


class Interpretation(BaseModel):
    """One defensible reading of an ambiguous question, with runnable SQL."""

    label: str = Field(description="Short slug naming this reading, e.g. gross_revenue.")
    sql: str = Field(description="A single PostgreSQL SELECT answering this reading.")
    explanation: str = Field(description="What this reading counts, and what it leaves out.")

    @field_validator("sql")
    @classmethod
    def _check_sql(cls, value: str) -> str:
        return _must_be_a_single_select(value, field="interpretation sql")


class Ambiguity(BaseModel):
    """Whether the question admits more than one defensible answer."""

    is_ambiguous: bool = Field(
        description="True when a term in the question has more than one defensible meaning."
    )
    interpretations: list[Interpretation] = Field(
        description=(
            f"When is_ambiguous is true, {MIN_INTERPRETATIONS}-{MAX_INTERPRETATIONS} "
            "competing readings. Empty otherwise."
        )
    )

    @model_validator(mode="after")
    def _check_interpretation_count(self) -> Ambiguity:
        if not self.is_ambiguous:
            # Interpretations alongside is_ambiguous=false are ignored, not
            # rejected. Raising here would turn a harmless surplus into a hard
            # failure of a call that already cost money and produced usable SQL.
            return self
        if not MIN_INTERPRETATIONS <= len(self.interpretations) <= MAX_INTERPRETATIONS:
            raise ValueError(
                f"an ambiguous answer needs {MIN_INTERPRETATIONS}-{MAX_INTERPRETATIONS} "
                f"interpretations, got {len(self.interpretations)}"
            )
        return self


class GeneratedSQL(BaseModel):
    """The model's answer, as a typed object rather than prose."""

    sql: str = Field(
        description="A single PostgreSQL SELECT or WITH ... SELECT. Empty when ambiguous."
    )
    explanation: str = Field(description="One or two sentences on how it answers the question.")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 self-estimate.")
    tables_used: list[str] = Field(description="Tables referenced by the query.")
    columns_used: list[str] = Field(description="Columns referenced, as table.column.")
    assumptions: list[str] = Field(description="Judgement calls made; empty if none.")
    ambiguity: Ambiguity = Field(description="Whether the question has one defensible reading.")

    @field_validator("sql")
    @classmethod
    def _strip_sql(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _sql_required_unless_ambiguous(self) -> GeneratedSQL:
        """`sql` is mandatory only when a single reading was actually chosen.

        Whether an empty `sql` is a bug or the correct answer depends on
        `ambiguity`, and a field validator cannot see a sibling field -- hence
        the check lives here rather than on `sql` itself.
        """
        if self.ambiguity.is_ambiguous:
            return self
        _must_be_a_single_select(self.sql, field="sql")
        return self


@dataclass(frozen=True)
class ClarificationNeeded:
    """Returned instead of SQL when the question has no single defensible answer.

    A distinct type rather than a flag on GeneratedSQL: the caller cannot
    accidentally read `.sql` off this and run a query nobody chose.
    """

    question: str
    interpretations: list[Interpretation]


def generate_sql_with_stats(
    question: str, *, client: LLMClient | None = None, schema: Any = None
) -> tuple[GeneratedSQL | ClarificationNeeded, CallResult]:
    """generate_sql, plus the call's usage and cost for the CLI to report."""
    llm = client or LLMClient()
    result = llm.complete(
        build_system_blocks(schema),
        build_user_message(question),
        output_format=GeneratedSQL,
        # Adaptive thinking with medium effort: enough reasoning to get joins
        # and status filters right without paying high-effort spend on a
        # one-shot generation. budget_tokens and temperature are rejected
        # outright by this model -- do not add them.
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
    )

    answer: GeneratedSQL = result.parsed
    if answer.ambiguity.is_ambiguous:
        return (
            ClarificationNeeded(
                question=question,
                interpretations=list(answer.ambiguity.interpretations),
            ),
            result,
        )
    return answer, result


def generate_sql(question: str) -> GeneratedSQL | ClarificationNeeded:
    """Generate SQL for a question. Does not execute it.

    Returns ClarificationNeeded when the question has more than one defensible
    reading, so a caller that assumes `.sql` will fail loudly rather than run a
    query that answers a question nobody asked.
    """
    return generate_sql_with_stats(question)[0]


def _footer(result: CallResult) -> list[str]:
    """Cost and usage lines, identical for either kind of answer."""
    usage = result.usage
    return [
        "",
        f"Cost        : ${result.cost_usd:.6f}  ({result.latency_ms} ms)",
        f"Tokens      : in {getattr(usage, 'input_tokens', 0)}"
        f" / out {getattr(usage, 'output_tokens', 0)}"
        f" / cache write {getattr(usage, 'cache_creation_input_tokens', 0) or 0}"
        f" / cache read {getattr(usage, 'cache_read_input_tokens', 0) or 0}",
        f"Running total: ${running_total_usd():.6f}",
    ]


def _render(question: str, answer: GeneratedSQL, result: CallResult) -> str:
    lines = [
        f"Q: {question}",
        "",
        answer.sql,
        "",
        f"Explanation : {answer.explanation}",
        f"Confidence  : {answer.confidence:.2f}",
        f"Tables      : {', '.join(answer.tables_used) or '(none)'}",
    ]
    if answer.assumptions:
        lines.append("Assumptions :")
        lines.extend(f"  - {item}" for item in answer.assumptions)
    else:
        lines.append("Assumptions : (none)")

    return "\n".join(lines + _footer(result))


def _render_clarification(answer: ClarificationNeeded, result: CallResult) -> str:
    """Print every reading side by side, so the caller can pick one.

    No query is singled out as the default. Presenting one first-among-equals
    would reintroduce exactly the silent guess this feature exists to prevent.
    """
    count = len(answer.interpretations)
    lines = [
        f"Q: {answer.question}",
        "",
        f"CLARIFICATION NEEDED - {count} defensible readings:",
    ]
    for index, option in enumerate(answer.interpretations, start=1):
        lines += ["", f"  [{index}] {option.label}", f"      {option.explanation}", ""]
        lines.extend(f"      {line}" for line in option.sql.splitlines())

    return "\n".join(lines + _footer(result))


def _as_json(question: str, answer: GeneratedSQL | ClarificationNeeded, result: CallResult) -> str:
    """Machine-readable form, tagged so a consumer can branch without parsing prose."""
    usage = result.usage
    payload: dict[str, Any] = {
        "question": question,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        },
    }
    if isinstance(answer, ClarificationNeeded):
        payload["kind"] = "clarification"
        payload["interpretations"] = [option.model_dump() for option in answer.interpretations]
    else:
        payload["kind"] = "sql"
        payload |= answer.model_dump()
    return json.dumps(payload, indent=2, sort_keys=True)


# Exit codes. A clarification is a success, not an error -- but it carries its
# own code so a script can branch on it without parsing stdout.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CAP_EXCEEDED = 2
EXIT_CLARIFICATION_NEEDED = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m queryguard.generate",
        description="Generate (but do not run) SQL for a natural-language question.",
        epilog=(
            f"exit codes: {EXIT_OK} a single query, "
            f"{EXIT_CLARIFICATION_NEEDED} clarification needed, "
            f"{EXIT_ERROR} error, {EXIT_CAP_EXCEEDED} request cap exceeded"
        ),
    )
    parser.add_argument("question", help="the question to translate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="text (default) for reading, json for piping",
    )
    args = parser.parse_args(argv)

    try:
        answer, result = generate_sql_with_stats(
            args.question, client=LLMClient(model=args.model)
        )
    except RequestCapExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_CAP_EXCEEDED
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.output_format == "json":
        print(_as_json(args.question, answer, result))
    elif isinstance(answer, ClarificationNeeded):
        print(_render_clarification(answer, result))
    else:
        print(_render(args.question, answer, result))

    return (
        EXIT_CLARIFICATION_NEEDED
        if isinstance(answer, ClarificationNeeded)
        else EXIT_OK
    )


if __name__ == "__main__":
    sys.exit(main())
