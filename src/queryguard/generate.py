"""Turn a natural-language question into a typed SQL query.

Generation only. Nothing in this module executes the SQL it produces -- running
it against the read-only role is Phase 2's job, and keeping the two apart means
a generation bug can never touch the database.

Usage:  uv run python -m queryguard.generate "How many orders were cancelled?"
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from pydantic import BaseModel, Field, field_validator

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


class GeneratedSQL(BaseModel):
    """The model's answer, as a typed object rather than prose."""

    sql: str = Field(description="A single PostgreSQL SELECT or WITH ... SELECT.")
    explanation: str = Field(description="One or two sentences on how it answers the question.")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 self-estimate.")
    tables_used: list[str] = Field(description="Tables referenced by the query.")
    columns_used: list[str] = Field(description="Columns referenced, as table.column.")
    assumptions: list[str] = Field(description="Judgement calls made; empty if none.")

    @field_validator("sql")
    @classmethod
    def _must_be_a_single_select(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("sql is empty")
        body = _LEADING_COMMENTS.sub("", stripped)
        if not re.match(r"(?i)(select|with)\b", body):
            raise ValueError(f"sql must start with SELECT or WITH, got: {body[:40]!r}")
        return stripped


def generate_sql_with_stats(
    question: str, *, client: LLMClient | None = None, schema: Any = None
) -> tuple[GeneratedSQL, CallResult]:
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
    return result.parsed, result


def generate_sql(question: str) -> GeneratedSQL:
    """Generate SQL for a question. Does not execute it."""
    return generate_sql_with_stats(question)[0]


def _render(question: str, answer: GeneratedSQL, result: CallResult) -> str:
    usage = result.usage
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

    lines += [
        "",
        f"Cost        : ${result.cost_usd:.6f}  ({result.latency_ms} ms)",
        f"Tokens      : in {getattr(usage, 'input_tokens', 0)}"
        f" / out {getattr(usage, 'output_tokens', 0)}"
        f" / cache write {getattr(usage, 'cache_creation_input_tokens', 0) or 0}"
        f" / cache read {getattr(usage, 'cache_read_input_tokens', 0) or 0}",
        f"Running total: ${running_total_usd():.6f}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m queryguard.generate",
        description="Generate (but do not run) SQL for a natural-language question.",
    )
    parser.add_argument("question", help="the question to translate")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    args = parser.parse_args(argv)

    try:
        answer, result = generate_sql_with_stats(
            args.question, client=LLMClient(model=args.model)
        )
    except RequestCapExceeded as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(_render(args.question, answer, result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
