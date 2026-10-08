"""Does the answer look like an answer? Sanity checks on an executed result.

The guardrail decides whether SQL may run and the executor decides what it may
cost. Neither says anything about whether the rows are *right*, and the most
common way generated SQL goes wrong is not an error at all: it is a query that
runs cleanly and returns a plausible-looking table built on a bad JOIN or a
filter that matched nothing. These checks look at the rows the way a reviewer
would and flag shapes that are rarely correct:

- empty_result          nothing came back for a question that asks for data --
                        including one row of count = 0 or NULL/zero sums, which
                        is how an aggregate reports that it matched nothing
- null_heavy_column     more than half NULL: usually an outer join that missed,
                        or a DESC sort putting NULLs first (Postgres's default)
- constant_column       one value in every row of more than five -- the Phase 0
                        seed bug looked exactly like this, a volatile expression
                        evaluated once and copied into every row
- date_out_of_span      dates before or after anything in the data
- negative_values       a negative amount, count or quantity
- implausible_aggregate sum() beyond the column's profiled max times the
                        table's row count, which only a fan-out join can reach
- duplicate_rows        exact duplicate rows despite a primary key column and
                        no DISTINCT, the other signature of a fan-out join
- revenue_status        a revenue or spend total that sums orders.total_amount
                        without a status filter excluding pending and
                        cancelled orders, which the metric glossary says are
                        not revenue

The last one reads the SQL, not the rows. Every validator that compares the
SQL with the *question* passed refund_04 in live-2026-10-01 (gross revenue
that included unpaid orders): the question never says "exclude pending", so
a query that forgets to is a faithful translation of it. Only a stated
business rule catches that, so this check applies the glossary's revenue
definition directly.

Every range comes from the schema profile in `schema_cache.json`, never from a
number written here, so the checks follow the data when it is re-seeded. The
profile is a snapshot: a result that disagrees with it may mean the data moved
since `extracted_at`, which is why most of these warn rather than fail. `fail`
is kept for results the profiled data cannot produce at all.

A flag is advice, not a gate: the query has already run and its rows are still
returned. Pure Python over a DataFrame -- no database, no API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Iterator, Literal

import pandas as pd
import sqlparse
from sqlparse import tokens as T
from sqlparse.sql import Function, Having, Identifier, IdentifierList, Statement, TokenList, Where

from queryguard.executor import ExecutionResult
from queryguard.schema.introspect import ColumnInfo, DatabaseSchema, TableInfo

Severity = Literal["info", "warn", "fail"]
INFO: Severity = "info"
WARN: Severity = "warn"
FAIL: Severity = "fail"
_SEVERITY_ORDER = {FAIL: 0, WARN: 1, INFO: 2}

CHECK_EMPTY = "empty_result"
CHECK_NULL_HEAVY = "null_heavy_column"
CHECK_CONSTANT = "constant_column"
CHECK_DATE_SPAN = "date_out_of_span"
CHECK_NEGATIVE = "negative_values"
CHECK_AGGREGATE = "implausible_aggregate"
CHECK_DUPLICATES = "duplicate_rows"
CHECK_REVENUE_STATUS = "revenue_status"

# Above this fraction a column counts as NULL-heavy. Below the row minimum one
# NULL is already a large fraction and means nothing.
NULL_HEAVY_FRACTION = 0.5
NULL_HEAVY_MIN_ROWS = 4

# "More than five rows". Five identical values can be coincidence; a thousand
# cannot.
CONSTANT_MIN_ROWS = 6

# round(avg(x), 2) can land a hair above max(x).
_AGGREGATE_TOLERANCE = 0.01

_AGGREGATES = frozenset({"sum", "count", "avg", "min", "max"})

# Words in a column name that make it an amount, count or quantity.
_QUANTITY_WORDS = frozenset(
    {"amount", "count", "quantity", "qty", "total", "revenue", "sales", "price",
     "cost", "spend", "units", "orders", "items", "n"}
)
# Words that make a negative value legitimate.
_SIGNED_WORDS = frozenset(
    {"net", "delta", "diff", "difference", "change", "growth", "balance", "profit",
     "margin", "variance", "adjustment"}
)

# A question phrased as "is there any ..." is answered by an empty result.
_EXISTENCE = re.compile(
    r"^\s*(is|are|was|were|do|does|did|has|have|can)\b"
    r"|\b(any|ever|none|no one|nobody|never|without)\b",
    re.IGNORECASE,
)

_TABLE_REF = re.compile(
    r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)(?:\s+(?:AS\s+)?([A-Za-z_]\w*))?", re.IGNORECASE
)
_NOT_AN_ALIAS = frozenset(
    {"where", "join", "on", "left", "right", "inner", "outer", "full", "cross", "natural",
     "using", "group", "order", "limit", "having", "union", "intersect", "except",
     "window", "lateral", "fetch", "offset"}
)
_DATE_TRUNC = re.compile(r"date_trunc\s*\(\s*'(\w+)'", re.IGNORECASE)
_TRUNC_ORDER = ("day", "week", "month", "quarter", "year")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# The metric glossary (db/init/04_comments.sql on orders.total_amount, and the
# system prompt): revenue sums total_amount over these statuses only.
REVENUE_STATUSES = ("paid", "shipped", "delivered", "refunded")
NOT_REVENUE_STATUSES = ("pending", "cancelled")
_REVENUE_RULE = (
    "revenue = sum(orders.total_amount) for status IN ('paid', 'shipped', "
    "'delivered', 'refunded'); pending and cancelled orders are not revenue"
)
# Words in the question or an output name that make a total a revenue/spend figure.
_REVENUE_WORDS = re.compile(
    r"\b(revenue|sales|spend|spent|spending|earned|earnings|income|turnover)\b", re.IGNORECASE
)
_STATUS_PREDICATE = re.compile(
    r"(?:\b\w+\.)?\bstatus\s*(=|<>|!=|\bNOT\s+IN\b|\bIN\b)\s*(\([^)]*\)|'(?:[^']|'')*')",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"'((?:[^']|'')*)'")


@dataclass(frozen=True)
class SanityFlag:
    """One suspicious shape in a result, and why it is suspicious."""

    check: str
    severity: Severity
    explanation: str
    column: str | None = None


# ------------------------------------------------------------- reading the SQL


@dataclass(frozen=True)
class _Output:
    """What one select-list item is made of, as far as it can be told."""

    aggregate: str | None = None  # sum/count/avg/min/max, when that is the whole item
    qualifier: str | None = None  # the table alias in `o.total_amount`
    column: str | None = None  # the source column, when the item reads exactly one


def _is_operator(token) -> bool:
    return token.ttype is not None and token.ttype in T.Operator


def _single_column(node) -> tuple[str | None, str | None]:
    """(qualifier, column) when `node` is a bare column reference, a cast allowed."""
    if not isinstance(node, Identifier):
        return None, None
    if any(isinstance(t, Function) for t in node.get_sublists()):
        return None, None
    if any(_is_operator(t) for t in node.flatten()):
        return None, None
    return node.get_parent_name(), node.get_real_name()


def _functions(node) -> Iterator[Function]:
    if isinstance(node, Function):
        yield node
    if isinstance(node, TokenList):
        for child in node.tokens:
            yield from _functions(child)


def _describe(item) -> tuple[str, _Output]:
    """The output column name PostgreSQL gives `item`, and what it reads."""
    alias = item.get_alias() if isinstance(item, Identifier) else None

    aggregates = [f for f in _functions(item) if (f.get_name() or "").lower() in _AGGREGATES]
    if aggregates:
        agg = aggregates[0]
        name = alias or (item.get_name() or "").lower()
        # `sum(x) / count(*)` is a ratio, not a sum: only an aggregate that is
        # the whole item (round() and coalesce() around it are fine) is mapped.
        outside = sum(_is_operator(t) for t in item.flatten()) - sum(
            _is_operator(t) for t in agg.flatten()
        )
        if len(aggregates) > 1 or outside:
            return name, _Output()
        params = agg.get_parameters()
        qualifier, column = _single_column(params[0]) if len(params) == 1 else (None, None)
        return name, _Output(agg.get_name().lower(), qualifier, column)

    if isinstance(item, Function):
        return (item.get_name() or "").lower(), _Output()

    qualifier, column = _single_column(item)
    if column is not None:
        return alias or column, _Output(qualifier=qualifier, column=column)
    return alias or "?column?", _Output()


def _select_list(statement: Statement) -> tuple[bool, dict[str, _Output]]:
    """(has DISTINCT, output name -> _Output) for the outermost SELECT."""
    tokens = [t for t in statement.tokens if not t.is_whitespace]
    start = next(
        (i for i, t in enumerate(tokens) if t.ttype is T.DML and t.value.upper() == "SELECT"),
        None,
    )
    if start is None:
        return False, {}

    distinct = False
    outputs: dict[str, _Output] = {}
    for tok in tokens[start + 1:]:
        if tok.ttype is T.Keyword and tok.value.upper().startswith("DISTINCT"):
            distinct = True
            continue
        if tok.ttype is T.Keyword:
            break
        items = tok.get_identifiers() if isinstance(tok, IdentifierList) else [tok]
        for item in items:
            if isinstance(item, (Identifier, Function)):
                name, output = _describe(item)
                outputs.setdefault(name, output)
        break
    return distinct, outputs


def _filter_text(statement: Statement) -> str:
    """WHERE and HAVING at any depth: a column filtered there may well be constant."""
    parts: list[str] = []

    def walk(node) -> None:
        for child in getattr(node, "tokens", []):
            if isinstance(child, (Where, Having)):
                parts.append(str(child))
            elif isinstance(child, TokenList):
                walk(child)

    walk(statement)
    # sqlparse 0.6 does not group HAVING, so take everything after the keyword.
    having = re.search(r"\bHAVING\b(.*)", str(statement), re.IGNORECASE | re.DOTALL)
    if having:
        parts.append(having.group(1))
    return "\n".join(parts)


def _string_literals(statement: Statement) -> list[str]:
    return [
        t.value[1:-1].replace("''", "'")
        for t in statement.flatten()
        if t.ttype is T.Literal.String.Single
    ]


def _table_refs(sql: str, schema: DatabaseSchema | None) -> dict[str, TableInfo]:
    """Alias and table name -> TableInfo, for every schema table the SQL reads."""
    if schema is None:
        return {}
    unquoted = re.sub(r"'(?:[^']|'')*'", "''", sql)
    refs: dict[str, TableInfo] = {}
    for match in _TABLE_REF.finditer(unquoted):
        table = schema.table(match.group(1).split(".")[-1].lower())
        if table is None:
            continue
        refs.setdefault(table.name, table)
        alias = match.group(2)
        if alias and alias.lower() not in _NOT_AN_ALIAS:
            refs[alias.lower()] = table
    return refs


# --------------------------------------------------------- reading the profile


def _float(value: str | None) -> float | None:
    try:
        return None if value is None else float(value)
    except ValueError:
        return None


def _is_temporal(col: ColumnInfo) -> bool:
    return col.sql_type.startswith(("timestamp", "date"))


def _numeric_range(col: ColumnInfo | None) -> tuple[float, float] | None:
    if col is None or _is_temporal(col):
        return None
    low, high = _float(col.min_value), _float(col.max_value)
    return None if low is None or high is None else (low, high)


def _timestamp(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except ValueError:
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _temporal_range(columns: list[ColumnInfo]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """The widest span covered by these columns' profiles."""
    lows = [t for c in columns if _is_temporal(c) and (t := _timestamp(c.min_value)) is not None]
    highs = [t for c in columns if _is_temporal(c) and (t := _timestamp(c.max_value)) is not None]
    if not lows or not highs:
        return None
    return min(lows), max(highs)


def _profile_has_one_value(col: ColumnInfo | None) -> bool:
    if col is None:
        return False
    if col.distinct_count == 1 or (col.enum_values is not None and len(col.enum_values) == 1):
        return True
    if col.true_count is not None and col.false_count is not None:
        return col.true_count == 0 or col.false_count == 0
    return col.min_value is not None and col.min_value == col.max_value


# ------------------------------------------------------------- reading the rows


def _as_numeric(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_bool_dtype(series):
        return None
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    values = series.dropna()
    if len(values) and all(
        isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in values
    ):
        return pd.to_numeric(series, errors="coerce").astype(float)
    return None


def _as_datetimes(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True)
    values = series.dropna()
    if len(values) and all(isinstance(v, (date, datetime)) for v in values):
        return pd.to_datetime(series, utc=True)
    return None


def _fmt(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def _floor(ts: pd.Timestamp, unit: str | None) -> pd.Timestamp:
    ts = ts.normalize()
    if unit == "week":
        return ts - pd.Timedelta(days=ts.weekday())
    if unit == "month":
        return ts.replace(day=1)
    if unit == "quarter":
        return ts.replace(month=3 * ((ts.month - 1) // 3) + 1, day=1)
    if unit == "year":
        return ts.replace(month=1, day=1)
    return ts


# ------------------------------------------------------------------ the context


@dataclass
class _Column:
    name: str
    values: pd.Series
    output: _Output
    source: ColumnInfo | None
    table: TableInfo | None

    @property
    def label(self) -> str:
        """`orders.total_amount` when the source is known, else the result name."""
        if self.source is not None and self.table is not None:
            return f"{self.table.name}.{self.source.name}"
        return self.name


@dataclass
class _Context:
    question: str
    sql: str
    frame: pd.DataFrame
    schema: DatabaseSchema | None
    statement: Statement
    tables: dict[str, TableInfo]
    distinct: bool
    columns: list[_Column]

    @property
    def referenced(self) -> list[TableInfo]:
        return list({t.name: t for t in self.tables.values()}.values())

    @property
    def profiled_at(self) -> str:
        return f"{self.schema.extracted_at:%Y-%m-%d}" if self.schema else "unknown"


def _resolve(
    output: _Output, tables: dict[str, TableInfo]
) -> tuple[TableInfo | None, ColumnInfo | None]:
    if output.column is None:
        return None, None
    candidates = (
        [tables[output.qualifier.lower()]]
        if output.qualifier and output.qualifier.lower() in tables
        else list({t.name: t for t in tables.values()}.values())
    )
    for table in candidates:
        for col in table.columns:
            if col.name == output.column.lower():
                return table, col
    return None, None


def _build_context(
    question: str, sql: str, frame: pd.DataFrame, schema: DatabaseSchema | None
) -> _Context:
    statement = sqlparse.parse(sql)[0]
    distinct, outputs = _select_list(statement)
    tables = _table_refs(sql, schema)

    columns: list[_Column] = []
    for index, name in enumerate(frame.columns):
        # `SELECT *` and anything unparsed fall back to a column of the same name.
        output = outputs.get(name, _Output(column=str(name)))
        table, source = _resolve(output, tables)
        columns.append(_Column(str(name), frame.iloc[:, index], output, source, table))

    return _Context(question, sql, frame, schema, statement, tables, distinct, columns)


# ------------------------------------------------------------------- the checks


def _filter_hints(ctx: _Context) -> list[str]:
    """Concrete reasons a filter in the SQL might match nothing."""
    hints: list[str] = []
    literals = _string_literals(ctx.statement)

    for literal in literals:
        for table in ctx.referenced:
            for col in table.columns:
                if not col.enum_values or literal in col.enum_values:
                    continue
                near = [v for v in col.enum_values if v.casefold() == literal.casefold()]
                if near:
                    hints.append(
                        f"'{literal}' is not a stored value of {table.name}.{col.name}, "
                        f"which holds '{near[0]}' (values are case-sensitive)"
                    )

    span = _temporal_range([c for t in ctx.referenced for c in t.columns])
    if span is not None:
        for literal in literals:
            if not _ISO_DATE.match(literal):
                continue
            stamp = _timestamp(literal)
            if stamp is None:
                continue
            if stamp > span[1]:
                hints.append(f"the date {literal} is after the latest date in the data ({span[1]:%Y-%m-%d})")
            elif stamp < _floor(span[0], None):
                hints.append(f"the date {literal} is before the earliest date in the data ({span[0]:%Y-%m-%d})")
    return hints


def _matched_nothing(ctx: _Context) -> bool:
    """Whether a one-row result is an aggregate over zero rows.

    sum() and avg() of nothing are NULL, and count() of nothing is 0, so an
    empty match still comes back as one row. A coalesced sum of 0 is treated
    the same way. `WHERE status = 'Cancelled'` returns count = 0, not zero
    rows, and without this it was never seen as empty.
    """
    aggregates = [c for c in ctx.columns if c.output.aggregate is not None]
    if not aggregates:
        # Unparsed select list: a row of nothing but NULLs is still empty.
        return bool(ctx.columns) and all(c.values.isna().all() for c in ctx.columns)

    def empty(col: _Column) -> bool:
        if col.values.isna().all():
            return True
        if col.output.aggregate in {"count", "sum"}:
            numbers = _as_numeric(col.values)
            return numbers is not None and bool((numbers.fillna(0) == 0).all())
        return False

    return all(empty(c) for c in aggregates)


def _check_empty(ctx: _Context) -> list[SanityFlag]:
    frame = ctx.frame
    if len(frame) == 0:
        what = "The query matched no rows"
    elif len(frame) == 1 and _matched_nothing(ctx):
        what = "The query returned a single row of empty aggregates (zero or NULL), so it matched no rows"
    else:
        return []

    hints = _filter_hints(ctx)
    hint_text = f" Possible cause: {'; '.join(hints)}." if hints else ""

    if _EXISTENCE.search(ctx.question):
        return [SanityFlag(
            CHECK_EMPTY, INFO,
            f"{what}. {ctx.question!r} reads as a yes/no question, so none may be the "
            f"answer.{hint_text}",
        )]
    return [SanityFlag(
        CHECK_EMPTY, WARN,
        f"{what}, but {ctx.question!r} asks for data that should exist. A WHERE "
        f"clause, a join condition or a date range is probably excluding "
        f"everything.{hint_text}",
    )]


def _nulls_first_by_desc(ctx: _Context, col: _Column) -> bool:
    """The result leads with NULLs because the outer ORDER BY sorts this column DESC.

    PostgreSQL's default for DESC is NULLS FIRST, so `ORDER BY lifetime_value
    DESC LIMIT 5` returns the NULL rows before any real value.
    """
    if col.source is not None and not col.source.nullable:
        return False
    if not pd.isna(col.values.iloc[0]):
        return False  # NULLS FIRST would put a NULL in the first row
    unquoted = re.sub(r"'(?:[^']|'')*'", "''", ctx.sql)
    clauses = re.findall(r"\bORDER\s+BY\b(.*?)(?=\bLIMIT\b|\bOFFSET\b|\bFETCH\b|\)|$)", unquoted, re.IGNORECASE | re.DOTALL)
    if not clauses:
        return False
    names = {col.name} | ({col.source.name} if col.source else set())
    return any(
        re.search(
            rf"(?:\b\w+\.)?\b{re.escape(name)}\s+DESC\b(?!\s+NULLS\s+LAST)",
            clauses[-1],
            re.IGNORECASE,
        )
        for name in names
    )


def _check_null_heavy(ctx: _Context) -> list[SanityFlag]:
    flags: list[SanityFlag] = []
    for col in ctx.columns:
        if len(col.values) < NULL_HEAVY_MIN_ROWS:
            continue
        fraction = float(col.values.isna().mean())
        if fraction <= NULL_HEAVY_FRACTION:
            continue
        observed = f"{col.name} is {fraction:.0%} NULL ({int(col.values.isna().sum())} of {len(col.values)} rows)"

        source = col.source
        if _nulls_first_by_desc(ctx, col):
            # Not a join problem: the top-N picked the NULLs, because they sort first.
            flags.append(SanityFlag(
                CHECK_NULL_HEAVY, WARN,
                f"{observed}; the query sorts {col.label} DESC and Postgres sorts NULLs "
                f"first in DESC, so the NULL rows fill the top of the result -- consider "
                f"NULLS LAST.",
                col.name,
            ))
            continue
        if source is not None and col.output.aggregate is None and source.null_fraction is not None:
            if source.null_fraction > NULL_HEAVY_FRACTION:
                flags.append(SanityFlag(
                    CHECK_NULL_HEAVY, INFO,
                    f"{observed}, in line with {col.label}, which is "
                    f"{source.null_fraction:.0%} NULL in the table itself.",
                    col.name,
                ))
                continue
            in_table = (
                "is declared NOT NULL"
                if not source.nullable
                else f"is only {source.null_fraction:.0%} NULL in the table"
            )
            reason = (
                f"{col.label} {in_table}, so the NULLs were introduced by the query -- "
                f"most likely an outer join whose condition did not match"
            )
        else:
            reason = "likely a bad JOIN: an outer join on the wrong key, or one whose condition did not match"
        flags.append(SanityFlag(CHECK_NULL_HEAVY, WARN, f"{observed}; {reason}.", col.name))
    return flags


def _check_constant(ctx: _Context) -> list[SanityFlag]:
    flags: list[SanityFlag] = []
    filters = _filter_text(ctx.statement)
    for col in ctx.columns:
        if len(col.values) < CONSTANT_MIN_ROWS or col.values.isna().any():
            continue
        try:
            if col.values.nunique() != 1:
                continue
        except TypeError:  # unhashable values, e.g. json
            continue
        names = {col.name} | ({col.source.name} if col.source else set())
        if any(re.search(rf"\b{re.escape(n)}\b", filters, re.IGNORECASE) for n in names):
            continue  # WHERE status = 'paid' makes every status 'paid'
        if _profile_has_one_value(col.source):
            continue  # orders.currency is 'USD' in every row of the table
        value = col.values.iloc[0]
        flags.append(SanityFlag(
            CHECK_CONSTANT, WARN,
            f"Every one of the {len(col.values)} rows has {col.name} = {value!r}, and nothing "
            f"in the query filters on it. That is the signature of an expression "
            f"evaluated once instead of per row (the Phase 0 seed bug) or of a join "
            f"that pinned the column to one match.",
            col.name,
        ))
    return flags


def _check_date_span(ctx: _Context) -> list[SanityFlag]:
    flags: list[SanityFlag] = []
    units = [u.lower() for u in _DATE_TRUNC.findall(ctx.sql) if u.lower() in _TRUNC_ORDER]
    unit = max(units, key=_TRUNC_ORDER.index) if units else None
    fallback_columns = [
        c for t in (ctx.referenced or (ctx.schema.tables if ctx.schema else [])) for c in t.columns
    ]

    for col in ctx.columns:
        stamps = _as_datetimes(col.values)
        if stamps is None or stamps.dropna().empty:
            continue
        if col.source is not None and _is_temporal(col.source):
            span, against = _temporal_range([col.source]), f"{col.label}"
        else:
            span, against = _temporal_range(fallback_columns), "any date column it reads"
        if span is None:
            continue

        low = _floor(span[0], unit)
        high = span[1].normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        outside = stamps[(stamps < low) | (stamps > high)]
        if outside.empty:
            continue
        flags.append(SanityFlag(
            CHECK_DATE_SPAN, WARN,
            f"{len(outside)} value(s) of {col.name} fall outside the data, which for "
            f"{against} runs {span[0]:%Y-%m-%d} to {span[1]:%Y-%m-%d} (profiled "
            f"{ctx.profiled_at}); earliest {outside.min():%Y-%m-%d}, latest "
            f"{outside.max():%Y-%m-%d}. Check date arithmetic, interval signs and "
            f"literal dates in the query.",
            col.name,
        ))
    return flags


def _check_negative(ctx: _Context) -> list[SanityFlag]:
    flags: list[SanityFlag] = []
    for col in ctx.columns:
        numbers = _as_numeric(col.values)
        if numbers is None or not (numbers < 0).any():
            continue
        words = set(re.split(r"[^a-z]+", col.name.lower())) - {""}
        if words & _SIGNED_WORDS:
            continue
        source_range = _numeric_range(col.source)
        if source_range is not None and source_range[0] < 0:
            continue  # the data itself has negatives

        from_count = col.output.aggregate == "count"
        from_non_negative = source_range is not None and col.output.aggregate in {
            None, "sum", "avg", "min", "max"
        }
        if not (from_count or from_non_negative or words & _QUANTITY_WORDS):
            continue

        count, lowest = int((numbers < 0).sum()), float(numbers.min())
        if from_count:
            why = "it comes from count(), which cannot be negative"
        elif from_non_negative:
            why = f"it comes from {col.label}, whose smallest value is {_fmt(source_range[0])}"
        else:
            why = "an amount, count or quantity should not be"
        flags.append(SanityFlag(
            CHECK_NEGATIVE, FAIL if (from_count or from_non_negative) else WARN,
            f"{count} row(s) have a negative {col.name} (lowest {_fmt(lowest)}), but "
            f"{why}. Look for a reversed subtraction or a sign error.",
            col.name,
        ))
    return flags


def _check_aggregates(ctx: _Context) -> list[SanityFlag]:
    flags: list[SanityFlag] = []
    largest = max(ctx.referenced, key=lambda t: t.row_count, default=None)

    for col in ctx.columns:
        aggregate = col.output.aggregate
        numbers = _as_numeric(col.values) if aggregate else None
        if numbers is None or numbers.dropna().empty:
            continue
        top, bottom = float(numbers.max()), float(numbers.min())
        source_range = _numeric_range(col.source)

        if aggregate == "sum" and source_range is not None and col.table is not None:
            ceiling = source_range[1] * col.table.row_count
            if source_range[1] > 0 and top > ceiling:
                flags.append(SanityFlag(
                    CHECK_AGGREGATE, FAIL,
                    f"sum({col.label}) reaches {_fmt(top)}, but the column's largest "
                    f"value ({_fmt(source_range[1])}) in every one of the table's "
                    f"{col.table.row_count:,} rows only adds up to {_fmt(ceiling)} "
                    f"(profiled {ctx.profiled_at}). A join is repeating rows before "
                    f"they are summed.",
                    col.name,
                ))

        elif aggregate == "count" and largest is not None and largest.row_count > 0:
            if top > largest.row_count:
                flags.append(SanityFlag(
                    CHECK_AGGREGATE, WARN,
                    f"{col.name} counts {_fmt(top)}, more than the {largest.row_count:,} "
                    f"rows in {largest.name}, the largest table the query reads. Joining "
                    f"two one-to-many relationships multiplies rows.",
                    col.name,
                ))

        elif aggregate in {"avg", "min", "max"} and source_range is not None:
            low, high = source_range
            if top > high + _AGGREGATE_TOLERANCE or bottom < low - _AGGREGATE_TOLERANCE:
                flags.append(SanityFlag(
                    CHECK_AGGREGATE, WARN,
                    f"{aggregate}({col.label}) gives values from {_fmt(bottom)} to "
                    f"{_fmt(top)}, outside the column's profiled range {_fmt(low)} to "
                    f"{_fmt(high)} (profiled {ctx.profiled_at}). Either the data has "
                    f"changed or the aggregate is not over the column it claims.",
                    col.name,
                ))
    return flags


def _check_duplicates(ctx: _Context) -> list[SanityFlag]:
    if ctx.distinct or len(ctx.frame) < 2:
        return []
    keys = [c for c in ctx.columns if c.source is not None and c.source.is_primary_key]
    if not keys:
        return []
    try:
        duplicates = int(ctx.frame.duplicated().sum())
    except TypeError:
        return []
    if not duplicates:
        return []
    key = keys[0]
    return [SanityFlag(
        CHECK_DUPLICATES, WARN,
        f"{duplicates} of {len(ctx.frame)} rows are exact duplicates, although the "
        f"result includes the primary key {key.label} and the query has no DISTINCT. "
        f"A join is matching each {key.table.name} row more than once.",
        key.name,
    )]


def _sums_total_amount(ctx: _Context) -> str | None:
    """The output name of the sum over total_amount, '' if it is nested, None if absent."""
    for function in _functions(ctx.statement):
        if (function.get_name() or "").lower() != "sum":
            continue
        if any(t.ttype in T.Name and t.value.lower() == "total_amount" for t in function.flatten()):
            for col in ctx.columns:
                if col.output.aggregate == "sum" and (col.output.column or "").lower() == "total_amount":
                    return col.name
            return ""
    return None


def _statuses_kept(filters: str) -> set[str]:
    """Which of NOT_REVENUE_STATUSES the WHERE/HAVING status predicates still let through.

    Predicates are read as if ANDed together. Under an OR that overstates what is
    excluded, so the check can miss a query, never flag a correct one.
    """
    kept = set(NOT_REVENUE_STATUSES)
    for op, operand in _STATUS_PREDICATE.findall(filters):
        values = {v.replace("''", "'") for v in _QUOTED.findall(operand)}
        op = " ".join(op.upper().split())
        if op in {"=", "IN"}:
            kept &= values
        else:  # <>, !=, NOT IN
            kept -= values
    return kept


def _check_revenue_status(ctx: _Context) -> list[SanityFlag]:
    if not re.search(r"\b(?:FROM|JOIN)\s+(?:\w+\.)?orders\b", ctx.sql, re.IGNORECASE):
        return []
    output = _sums_total_amount(ctx)
    if output is None:
        return []
    names = " ".join(c.name for c in ctx.columns)
    if not (_REVENUE_WORDS.search(ctx.question) or _REVENUE_WORDS.search(names.replace("_", " "))):
        return []
    kept = _statuses_kept(_filter_text(ctx.statement))
    if not kept:
        return []
    statuses = " and ".join(f"'{s}'" for s in NOT_REVENUE_STATUSES if s in kept)
    return [SanityFlag(
        CHECK_REVENUE_STATUS, WARN,
        f"The query sums orders.total_amount as a revenue or spend total, but no status "
        f"filter excludes {statuses} orders. Glossary: {_REVENUE_RULE}.",
        output or None,
    )]


_CHECKS = (
    _check_empty,
    _check_null_heavy,
    _check_constant,
    _check_date_span,
    _check_negative,
    _check_aggregates,
    _check_duplicates,
    _check_revenue_status,
)


def check_result(
    question: str,
    sql: str,
    execution_result: ExecutionResult,
    schema: DatabaseSchema | None,
) -> list[SanityFlag]:
    """Flag result shapes that are rarely correct, most severe first.

    `sql` should be the SQL that ran. With `schema` None the profile-based
    checks have no ranges to compare against and find nothing; the purely
    structural ones (empty, NULL-heavy, constant, duplicate) still run.
    A result that did not run has no rows to check and yields no flags.
    """
    if not execution_result.ok or execution_result.rows is None:
        return []
    ctx = _build_context(question, sql, execution_result.rows, schema)
    flags = [flag for check in _CHECKS for flag in check(ctx)]
    return sorted(flags, key=lambda f: _SEVERITY_ORDER[f.severity])
