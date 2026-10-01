"""Two independent queries for one question: do they return the same answer?

A wrong query and a right one rarely agree. If a second query, written with a
deliberately different structure, returns the same rows, both are probably
right; if it does not, at least one is wrong -- and the diff says where to look.

Only non-trivial queries get a second opinion. `SELECT email FROM customers
WHERE customer_id = 42` has nowhere to hide a mistake that a rewrite would
catch, and the second call costs a Sonnet generation. Anything with a join, an
aggregate, a CTE or a subquery qualifies, and so does a top-N (`ORDER BY ...
LIMIT`): the first live run ranked customers by `lifetime_value` for a question
about 2025 spend, a single-table query with no join or aggregate in sight, and
ranking by the wrong column is exactly the mistake a rewrite exposes.

Comparison rules, in the order they apply:

- Shape first. A different number of columns is *incomparable*: the two queries
  answer at different granularities, or one adds a label column, and there is
  no principled way to line them up. A different number of rows is *disagree*.
- Column names are ignored -- `revenue` and `total_spent` are the same column if
  they hold the same values. Columns are matched by position, or, failing that,
  by content, so `SELECT name, total` agrees with `SELECT total, name`.
- Row order is ignored unless the question implies one ("top 5", "most",
  "latest"); then rows are compared in order.
- Numbers compare with a 1e-6 relative tolerance. Beyond that, a value that
  equals the other rounded to its own displayed precision -- `round(avg(x), 2)`
  against an unrounded `avg(x)` -- also counts as equal, and the explanation
  says so. Without that rule every rounded aggregate would read as disagreement.
- Decimal, int and float compare as numbers; date, timestamp and timestamptz
  compare as UTC instants; NULL equals NULL.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Function, Parenthesis

from queryguard.executor import ExecutionResult, ExecutorConfig, execute
from queryguard.generate import GeneratedSQL
from queryguard.guardrails import GuardrailConfig, check
from queryguard.llm.client import CallResult, LLMClient
from queryguard.llm.prompt import build_system_blocks
from queryguard.schema.introspect import DatabaseSchema

AGREE = "agree"
DISAGREE = "disagree"
INCOMPARABLE = "incomparable"

REL_TOL = 1e-6
ABS_TOL = 1e-9

_AGGREGATES = frozenset({"count", "sum", "avg", "min", "max", "array_agg", "string_agg", "bool_and", "bool_or"})
_ORDERING = re.compile(
    r"\b(top|most|least|highest|lowest|largest|smallest|biggest|best|worst|first|"
    r"last|latest|earliest|newest|oldest|rank|ranked|ranking|sorted|ascending|"
    r"descending|order(?:ed)? by)\b",
    re.IGNORECASE,
)

SECOND_OPINION_INSTRUCTION = """\
Write your own query for the question above. A query written for it earlier is
shown below for structural contrast only: it may be wrong, so do not copy its
logic, and use a structurally different approach -- for example a CTE where it
uses a subquery, a subquery or EXISTS where it joins, a different join order,
or a window function where it aggregates -- while answering the question as
asked. Do not set `ambiguity.is_ambiguous`: if the question admits more than
one reading, choose the one its wording most supports and record it in
`assumptions`.

Earlier query:
"""


@dataclass(frozen=True)
class AgreementResult:
    outcome: str
    explanation: str
    second_sql: str | None = None
    second_execution: ExecutionResult | None = None
    call: CallResult | None = None


# ------------------------------------------------------------- which queries


def _has_subquery(node) -> bool:
    for tok in getattr(node, "tokens", []):
        if isinstance(tok, Parenthesis) and any(
            t.ttype is T.DML and t.value.upper() == "SELECT" for t in tok.flatten()
        ):
            return True
        if _has_subquery(tok):
            return True
    return False


def _has_aggregate(node) -> bool:
    for tok in getattr(node, "tokens", []):
        if isinstance(tok, Function) and (tok.get_name() or "").lower() in _AGGREGATES:
            return True
        if _has_aggregate(tok):
            return True
    return False


def is_non_trivial(sql: str) -> bool:
    """A join, an aggregate, a CTE or a subquery: somewhere a mistake can hide."""
    statement = sqlparse.parse(sql)[0]
    keywords = {t.normalized for t in statement.flatten() if t.ttype in T.Keyword}
    if any("JOIN" in k for k in keywords) or "GROUP BY" in keywords:
        return True
    if "ORDER BY" in keywords and "LIMIT" in keywords:
        return True  # a top-N: which rows make the cut depends on what is ranked
    if any(t.ttype is T.Keyword.CTE for t in statement.flatten()):
        return True
    return _has_aggregate(statement) or _has_subquery(statement)


def implies_ordering(question: str) -> bool:
    return bool(_ORDERING.search(question))


# ------------------------------------------------------------- comparing rows


def _norm(value: Any) -> tuple[str, Any]:
    """(kind, comparable value) for one cell."""
    if value is None or value is pd.NaT:
        return ("null", None)
    if isinstance(value, float) and math.isnan(value):
        return ("null", None)
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, float, Decimal, np.integer, np.floating)):
        return ("num", value)
    if isinstance(value, (datetime, date)):
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return ("time", stamp)
    return ("text", str(value))


def _decimals(value: Any) -> int:
    """Displayed fractional digits: Decimal('3.10') -> 2, 3.25 -> 2, 7 -> 0."""
    if isinstance(value, Decimal):
        return max(0, -value.as_tuple().exponent)
    if isinstance(value, (int, np.integer)):
        return 0
    text = repr(float(value))
    if "e" in text or "E" in text:
        return 15
    return len(text.split(".")[1].rstrip("0")) if "." in text else 0


def _cells_equal(a: tuple[str, Any], b: tuple[str, Any]) -> tuple[bool, bool]:
    """(equal, equal only after rounding)."""
    kind_a, va = a
    kind_b, vb = b
    if kind_a != kind_b:
        return False, False
    if kind_a == "null":
        return True, False
    if kind_a != "num":
        return va == vb, False

    fa, fb = float(va), float(vb)
    if math.isclose(fa, fb, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True, False
    # round(avg(x), 2) vs avg(x). Only for values with a fractional part, so a
    # count of 3 never "rounds" to equal an average of 3.4.
    places = min(_decimals(va), _decimals(vb))
    if places >= 1 and math.isclose(round(fa, places), round(fb, places), rel_tol=REL_TOL, abs_tol=ABS_TOL):
        return True, True
    return False, False


def _sort_key(row: tuple[tuple[str, Any], ...]) -> tuple:
    order = {"null": 0, "bool": 1, "num": 2, "time": 3, "text": 4}
    key = []
    for kind, value in row:
        if kind == "num":
            value = float(value)
        elif kind == "time":
            value = value.value
        elif kind == "null":
            value = 0
        key.append((order[kind], value))
    return tuple(key)


def _column(frame: pd.DataFrame, index: int) -> list[tuple[str, Any]]:
    return [_norm(v) for v in frame.iloc[:, index].tolist()]


def _columns_match(a: list, b: list, ordered: bool) -> bool:
    if not ordered:
        a = sorted(a, key=lambda c: _sort_key((c,)))
        b = sorted(b, key=lambda c: _sort_key((c,)))
    return all(_cells_equal(x, y)[0] for x, y in zip(a, b))


def _align_columns(first: pd.DataFrame, second: pd.DataFrame, ordered: bool) -> list[int]:
    """Positions in `second` for each column of `first`: by position, else by content."""
    width = first.shape[1]
    identity = list(range(width))
    if all(_columns_match(_column(first, i), _column(second, i), ordered) for i in identity):
        return identity
    unused = set(identity)
    mapping: list[int] = []
    for i in identity:
        left = _column(first, i)
        match = next((j for j in sorted(unused) if _columns_match(left, _column(second, j), ordered)), None)
        if match is None:
            return identity  # no clean permutation: compare positionally and report
        mapping.append(match)
        unused.discard(match)
    return mapping


def _show(cell: tuple[str, Any]) -> str:
    kind, value = cell
    if kind == "null":
        return "NULL"
    if kind == "time":
        return value.isoformat()
    if kind == "num":
        return f"{float(value):,.6g}" if abs(float(value)) < 1e6 else f"{float(value):,.2f}"
    return repr(value)


def compare_results(first: pd.DataFrame, second: pd.DataFrame, *, ordered: bool) -> tuple[str, str]:
    """(outcome, explanation) for two result sets. Column names are never consulted."""
    if first.shape[1] != second.shape[1]:
        return INCOMPARABLE, (
            f"the queries return different columns ({first.shape[1]} vs {second.shape[1]}), "
            "so their rows cannot be lined up"
        )
    if len(first) != len(second):
        return DISAGREE, f"the queries return different row counts ({len(first)} vs {len(second)})"

    mapping = _align_columns(first, second, ordered)
    reordered = mapping != list(range(first.shape[1]))
    second = second.iloc[:, mapping]

    rows_a = [tuple(_norm(v) for v in row) for row in first.itertuples(index=False, name=None)]
    rows_b = [tuple(_norm(v) for v in row) for row in second.itertuples(index=False, name=None)]
    if not ordered:
        rows_a.sort(key=_sort_key)
        rows_b.sort(key=_sort_key)

    differing: list[tuple[int, int]] = []
    rounded = 0
    for r, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        for c, (cell_a, cell_b) in enumerate(zip(row_a, row_b)):
            equal, by_rounding = _cells_equal(cell_a, cell_b)
            if not equal:
                differing.append((r, c))
            rounded += by_rounding

    order_note = "in order" if ordered else "ignoring row order"
    if differing:
        r, c = differing[0]
        rows = len({r for r, _ in differing})
        where = f"row {r + 1}" + ("" if ordered else " after sorting")
        return DISAGREE, (
            f"{rows} of {len(rows_a)} rows differ ({order_note}); first difference at "
            f"{where}, column {c + 1}: {_show(rows_a[r][c])} vs {_show(rows_b[r][c])}"
        )

    notes = []
    if reordered:
        notes.append("columns matched by content, not position")
    if rounded:
        notes.append(f"{rounded} value(s) equal only after rounding to the shorter precision")
    suffix = f"; {'; '.join(notes)}" if notes else ""
    return AGREE, f"{len(rows_a)} rows x {first.shape[1]} columns match ({order_note}){suffix}"


# ------------------------------------------------------------ the second query


def second_opinion(
    question: str,
    first_sql: str,
    *,
    client: LLMClient,
    schema: DatabaseSchema,
) -> tuple[GeneratedSQL, CallResult]:
    """One Sonnet call, on the same cached prefix as generation."""
    result = client.complete(
        build_system_blocks(schema),
        f"Q: {question}\n\n{SECOND_OPINION_INSTRUCTION}{first_sql}",
        output_format=GeneratedSQL,
        # Same settings as generate.py: the second query should be as careful
        # as the first, or a disagreement says more about effort than SQL.
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
    )
    return result.parsed, result


def check_agreement(
    question: str,
    first_sql: str,
    first_execution: ExecutionResult,
    *,
    client: LLMClient,
    schema: DatabaseSchema,
    guardrail_config: GuardrailConfig | None = None,
    executor_config: ExecutorConfig | None = None,
) -> AgreementResult:
    """Generate, guard and run a second query, then compare the two results.

    The caller decides whether the query is non-trivial and whether the budget
    allows the call; this function always makes exactly one API call.
    """
    answer, call = second_opinion(question, first_sql, client=client, schema=schema)

    if answer.ambiguity.is_ambiguous or not answer.sql:
        return AgreementResult(INCOMPARABLE, "the second query declined to pick a reading", call=call)

    guardrail = check(answer.sql, guardrail_config)
    if not guardrail.allowed:
        return AgreementResult(
            INCOMPARABLE,
            f"the second query was blocked by {guardrail.rule}: {guardrail.reason}",
            second_sql=answer.sql,
            call=call,
        )

    execution = execute(guardrail.sql_to_execute, executor_config)
    if not execution.ok:
        detail = execution.reason or f"{execution.sqlstate} {execution.error_message}"
        return AgreementResult(
            INCOMPARABLE,
            f"the second query did not run ({execution.outcome}: {detail})",
            second_sql=answer.sql,
            second_execution=execution,
            call=call,
        )
    if first_execution.truncated or execution.truncated:
        return AgreementResult(
            INCOMPARABLE,
            "a result was truncated at the row cap, so the full sets cannot be compared",
            second_sql=answer.sql,
            second_execution=execution,
            call=call,
        )

    outcome, explanation = compare_results(
        first_execution.rows, execution.rows, ordered=implies_ordering(question)
    )
    return AgreementResult(outcome, explanation, answer.sql, execution, call)
