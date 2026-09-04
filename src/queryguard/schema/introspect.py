"""Introspect the QueryGuard database and render it compactly for an LLM prompt.

Structure comes from SQLAlchemy's Inspector; the interesting part is the
*profile* pass, which runs one cheap, timeout-guarded query per column so the
prompt can tell the model what the data actually looks like — that
`orders.status` only ever holds six values, that `phone` is 15% null — rather
than just its declared type.

The rendered text goes into every API call, so it is plain text with no JSON
and no markdown: every character is a token someone pays for.

Usage:  uv run python -m queryguard.schema.introspect --refresh
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy import types as sqltypes
from sqlalchemy.exc import SQLAlchemyError

from queryguard.config import database_url, schema_cache_path

# Per-query ceiling for the profile pass. A wide or unindexed table must never
# be able to hang a schema rebuild; a column that overruns is simply reported
# without a profile.
PROFILE_TIMEOUT_MS = 3000

# At or below this many distinct values a text column is treated as an
# enumeration and every value is stored. Above it, only examples plus a count.
ENUM_MAX_DISTINCT = 25
EXAMPLE_VALUES = 3

# Rough characters-per-token for English-ish text. Exact counting would need an
# Anthropic API round-trip, which is not worth a network call in a CLI whose
# job is rebuilding a local cache.
CHARS_PER_TOKEN = 4

# Postgres' canonical spellings are shorter than SQLAlchemy's compiled ones,
# and this string is repeated once per column in every prompt.
_SHORT_TYPE_NAMES = {
    "timestamp with time zone": "timestamptz",
    "timestamp without time zone": "timestamp",
    "time with time zone": "timetz",
    "time without time zone": "time",
    "double precision": "float8",
    "character varying": "varchar",
    "character": "char",
}


# --------------------------------------------------------------------- models


class ColumnInfo(BaseModel):
    name: str
    sql_type: str
    nullable: bool
    is_primary_key: bool = False
    comment: str | None = None
    references: str | None = None  # "customers.customer_id" for an outgoing FK

    # --- profile: only the fields for this column's type category are set ---
    enum_values: list[str] | None = None  # text with <= ENUM_MAX_DISTINCT values
    distinct_count: int | None = None  # text
    example_values: list[str] | None = None  # text above the enum threshold
    min_value: str | None = None  # numeric and temporal
    max_value: str | None = None
    null_fraction: float | None = None
    true_count: int | None = None  # boolean
    false_count: int | None = None
    null_count: int | None = None
    profile_error: str | None = None  # set when the guarded query did not finish


class ForeignKeyInfo(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class TableInfo(BaseModel):
    name: str
    comment: str | None = None
    row_count: int = 0
    columns: list[ColumnInfo] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    referenced_by: list[ForeignKeyInfo] = Field(default_factory=list)


class DatabaseSchema(BaseModel):
    tables: list[TableInfo] = Field(default_factory=list)
    extracted_at: datetime

    def table(self, name: str) -> TableInfo | None:
        return next((t for t in self.tables if t.name == name), None)

    def render_for_prompt(self) -> str:
        return "\n\n".join(_render_table(t) for t in self.tables if t.columns)


# ------------------------------------------------------------------ rendering


def _pct(fraction: float) -> str:
    """Percent with no false precision: '<1%' beats '0%' for a rare-but-real value."""
    pct = fraction * 100
    if 0 < pct < 1:
        return "<1%"
    if 0 < 100 - pct < 1 and pct < 100:
        return ">99%"
    return f"{round(pct)}%"


def _short_value(value: str) -> str:
    """One distinct value, safe to drop inside a comma-separated brace list."""
    flat = " ".join(value.split())
    if len(flat) > 24:
        flat = flat[:23] + "…"
    if not flat or "," in flat or "{" in flat or "}" in flat:
        return "'" + flat.replace("'", "''") + "'"
    return flat


def _short_bound(value: str, sql_type: str) -> str:
    """Range bounds: drop the time of day from timestamps.

    Date granularity is enough for the range reasoning a text-to-SQL model
    does, and the discarded '12:34:56.789+00:00' is pure token cost.
    """
    if sql_type.startswith("timestamp") and len(value) >= 10:
        return value[:10]
    return value


def _render_profile(col: ColumnInfo) -> str:
    if col.enum_values is not None:
        return "∈ {" + ", ".join(_short_value(v) for v in col.enum_values) + "}"
    if col.example_values:
        examples = ", ".join(_short_value(v) for v in col.example_values)
        return f"({col.distinct_count} distinct, e.g. {examples})"
    if col.min_value is not None and col.max_value is not None:
        lo = _short_bound(col.min_value, col.sql_type)
        hi = _short_bound(col.max_value, col.sql_type)
        return f"[{lo} .. {hi}]"
    if col.true_count is not None and col.false_count is not None:
        total = col.true_count + col.false_count + (col.null_count or 0)
        if total:
            return f"({_pct(col.true_count / total)} true, {_pct(col.false_count / total)} false)"
    return ""


def _render_column(col: ColumnInfo, width: int) -> str:
    parts = [col.sql_type]

    if col.is_primary_key:
        parts.append("PK")
    elif not col.nullable:
        parts.append("NOT NULL")
    else:
        # A nullable column with no nulls in practice is worth distinguishing
        # from one that is a third empty.
        if col.null_fraction:
            parts.append(f"NULL ({_pct(col.null_fraction)} null)")
        else:
            parts.append("NULL")

    profile = _render_profile(col)
    if profile:
        parts.append(profile)
    if col.references:
        parts.append(f"→ {col.references}")
    if col.comment:
        parts.append(f"-- {col.comment}")

    return f"  {col.name.ljust(width)}  {' '.join(parts)}"


def _render_table(table: TableInfo) -> str:
    header = f"{table.name} ({table.row_count} rows)"
    if table.comment:
        header += f"  -- {table.comment}"

    width = max(len(c.name) for c in table.columns)
    lines = [header] + [_render_column(c, width) for c in table.columns]

    # Reverse direction: which columns elsewhere point back at this table.
    for ref in sorted({f"{fk.from_table}.{fk.from_column}" for fk in table.referenced_by}):
        lines.append(f"  → {ref}")

    return "\n".join(lines)


def estimate_tokens(rendered: str) -> int:
    return max(1, len(rendered) // CHARS_PER_TOKEN) if rendered else 0


# ---------------------------------------------------------------- structure


def _short_type(type_: Any, engine: Engine) -> str:
    try:
        compiled = type_.compile(engine.dialect)
    except SQLAlchemyError:
        compiled = str(type_)
    lowered = compiled.lower()
    for verbose, short in _SHORT_TYPE_NAMES.items():
        if lowered.startswith(verbose):
            return (short + lowered[len(verbose) :]).replace(", ", ",")
    return lowered.replace(", ", ",")


def _collect_foreign_keys(inspector: Any, tables: Iterable[str]) -> list[ForeignKeyInfo]:
    """Every FK edge as a flat list, one entry per column pair.

    Both directions are derived from this single list, so a self-referencing
    constraint (categories.parent_category_id) lands correctly in the table's
    own foreign_keys *and* its referenced_by.
    """
    edges: list[ForeignKeyInfo] = []
    for table in tables:
        for fk in inspector.get_foreign_keys(table):
            referred = fk.get("referred_table")
            if not referred:
                continue
            for from_col, to_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                edges.append(
                    ForeignKeyInfo(
                        from_table=table,
                        from_column=from_col,
                        to_table=referred,
                        to_column=to_col,
                    )
                )
    return edges


# ---------------------------------------------------------------- profiling


class _ProfileTimeout(Exception):
    """A guarded query exceeded statement_timeout (or otherwise failed)."""


def _guarded(
    conn: Any, statement: str, timeout_ms: int, params: dict[str, Any] | None = None
) -> Any:
    """Run one statement under a transaction-local statement_timeout.

    set_config(..., is_local => true) is used rather than `SET statement_timeout`
    because it accepts a bind parameter, so no value is interpolated into SQL,
    and it resets itself when the transaction ends.
    """
    try:
        conn.execute(
            text("SELECT set_config('statement_timeout', :ms, true)"),
            {"ms": str(timeout_ms)},
        )
        row = conn.execute(text(statement), params or {}).one()
        conn.commit()
        return row
    except SQLAlchemyError as exc:
        conn.rollback()  # the aborted transaction is discarded; the connection survives
        raise _ProfileTimeout(str(exc).splitlines()[0]) from exc


def _guarded_scalars(conn: Any, statement: str, timeout_ms: int) -> list[Any]:
    try:
        conn.execute(
            text("SELECT set_config('statement_timeout', :ms, true)"),
            {"ms": str(timeout_ms)},
        )
        values = conn.execute(text(statement)).scalars().all()
        conn.commit()
        return list(values)
    except SQLAlchemyError as exc:
        conn.rollback()
        raise _ProfileTimeout(str(exc).splitlines()[0]) from exc


def _null_fraction(total: int, non_null: int) -> float:
    return 0.0 if total == 0 else 1.0 - (non_null / total)


def _profile_column(
    conn: Any, table_sql: str, col_sql: str, col: ColumnInfo, type_: Any, timeout_ms: int
) -> None:
    """Fill in col's profile fields with one cheap query per column.

    Boolean is checked before Integer on purpose: the two overlap conceptually
    and getting the order wrong silently profiles every flag as a number.
    """
    if isinstance(type_, sqltypes.Boolean):
        row = _guarded(
            conn,
            f"SELECT count(*) FILTER (WHERE {col_sql}),"
            f" count(*) FILTER (WHERE NOT {col_sql}),"
            f" count(*) FILTER (WHERE {col_sql} IS NULL) FROM {table_sql}",
            timeout_ms,
        )
        col.true_count, col.false_count, col.null_count = int(row[0]), int(row[1]), int(row[2])
        total = col.true_count + col.false_count + col.null_count
        col.null_fraction = _null_fraction(total, total - col.null_count)

    elif isinstance(type_, sqltypes.String):
        row = _guarded(
            conn,
            f"SELECT count(*), count({col_sql}), count(DISTINCT {col_sql}) FROM {table_sql}",
            timeout_ms,
        )
        total, non_null, distinct = int(row[0]), int(row[1]), int(row[2])
        col.distinct_count = distinct
        col.null_fraction = _null_fraction(total, non_null)
        if non_null:
            limit = ENUM_MAX_DISTINCT if distinct <= ENUM_MAX_DISTINCT else EXAMPLE_VALUES
            values = _guarded_scalars(
                conn,
                f"SELECT DISTINCT {col_sql} FROM {table_sql}"
                f" WHERE {col_sql} IS NOT NULL ORDER BY 1 LIMIT {limit}",
                timeout_ms,
            )
            rendered = [str(v) for v in values]
            if distinct <= ENUM_MAX_DISTINCT:
                col.enum_values = rendered
            else:
                col.example_values = rendered

    elif isinstance(type_, (sqltypes.Integer, sqltypes.Numeric)) or isinstance(
        type_, (sqltypes.Date, sqltypes.DateTime, sqltypes.Time)
    ):
        row = _guarded(
            conn,
            f"SELECT min({col_sql}), max({col_sql}), count(*), count({col_sql}) FROM {table_sql}",
            timeout_ms,
        )
        if row[0] is not None:
            col.min_value = str(row[0])
            col.max_value = str(row[1])
        col.null_fraction = _null_fraction(int(row[2]), int(row[3]))

    # Anything else (json, arrays, uuid, ...) has no cheap useful summary.


def _row_count(conn: Any, table_sql: str, qualified: str, timeout_ms: int) -> int:
    """Exact count, falling back to the planner's estimate if it overruns."""
    try:
        return int(_guarded(conn, f"SELECT count(*) FROM {table_sql}", timeout_ms)[0])
    except _ProfileTimeout:
        try:
            row = _guarded(
                conn,
                "SELECT reltuples::bigint FROM pg_class WHERE oid = CAST(:t AS regclass)",
                timeout_ms,
                {"t": qualified},
            )
            return max(0, int(row[0]))
        except _ProfileTimeout:
            return 0


# -------------------------------------------------------------- introspection


def introspect_database(
    url: Any = None, *, timeout_ms: int = PROFILE_TIMEOUT_MS
) -> DatabaseSchema:
    """Read structure and profile every column. Uses the owner connection."""
    engine = create_engine(url or database_url())
    try:
        inspector = inspect(engine)
        table_names = sorted(inspector.get_table_names())
        quote = engine.dialect.identifier_preparer.quote

        edges = _collect_foreign_keys(inspector, table_names)
        outgoing: dict[str, ForeignKeyInfo] = {
            f"{e.from_table}.{e.from_column}": e for e in edges
        }

        tables: list[TableInfo] = []
        with engine.connect() as conn:
            for name in table_names:
                table_sql = quote(name)
                pk_cols = set(inspector.get_pk_constraint(name).get("constrained_columns") or [])

                columns: list[ColumnInfo] = []
                for raw in inspector.get_columns(name):
                    edge = outgoing.get(f"{name}.{raw['name']}")
                    col = ColumnInfo(
                        name=raw["name"],
                        sql_type=_short_type(raw["type"], engine),
                        nullable=bool(raw["nullable"]),
                        is_primary_key=raw["name"] in pk_cols,
                        comment=raw.get("comment"),
                        references=f"{edge.to_table}.{edge.to_column}" if edge else None,
                    )
                    try:
                        _profile_column(
                            conn, table_sql, quote(raw["name"]), col, raw["type"], timeout_ms
                        )
                    except _ProfileTimeout as exc:
                        # One slow column degrades to "no profile", never a failed run.
                        col.profile_error = str(exc)
                    columns.append(col)

                tables.append(
                    TableInfo(
                        name=name,
                        comment=(inspector.get_table_comment(name) or {}).get("text"),
                        row_count=_row_count(conn, table_sql, name, timeout_ms),
                        columns=columns,
                        foreign_keys=[e for e in edges if e.from_table == name],
                        referenced_by=[e for e in edges if e.to_table == name],
                    )
                )

        return DatabaseSchema(tables=tables, extracted_at=datetime.now(timezone.utc))
    finally:
        engine.dispose()


# -------------------------------------------------------------------- cache


def save_schema(schema: DatabaseSchema, path: Path | None = None) -> Path:
    target = Path(path) if path else schema_cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(schema.model_dump_json(indent=2), encoding="utf-8")
    return target


def load_schema(
    refresh: bool = False,
    *,
    path: Path | None = None,
    timeout_ms: int = PROFILE_TIMEOUT_MS,
) -> DatabaseSchema:
    """Read the cached schema, rebuilding it from the database when needed."""
    target = Path(path) if path else schema_cache_path()

    if not refresh and target.is_file():
        try:
            return DatabaseSchema.model_validate_json(target.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # A truncated or stale-format cache is a reason to rebuild, not to fail.
            pass

    schema = introspect_database(timeout_ms=timeout_ms)
    save_schema(schema, target)
    return schema


# ---------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m queryguard.schema.introspect",
        description="Introspect the database and print the compact prompt schema.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="rebuild the cache from the database"
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=PROFILE_TIMEOUT_MS,
        help=f"per-query statement timeout (default {PROFILE_TIMEOUT_MS})",
    )
    parser.add_argument("--cache", type=Path, default=None, help="override the cache path")
    args = parser.parse_args(argv)

    try:
        schema = load_schema(args.refresh, path=args.cache, timeout_ms=args.timeout_ms)
    except SQLAlchemyError as exc:
        print(f"ERROR: could not read the database — {exc}", file=sys.stderr)
        print("Is it running?  docker compose up -d", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    rendered = schema.render_for_prompt()
    print(rendered)

    # Summary on stderr so the rendered schema can be piped somewhere useful.
    columns = sum(len(t.columns) for t in schema.tables)
    cache = Path(args.cache) if args.cache else schema_cache_path()
    print(
        f"\n{len(schema.tables)} tables, {columns} columns · {len(rendered)} chars"
        f" · ≈{estimate_tokens(rendered)} tokens (estimated) · cache: {cache}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
