"""Tests for result sanity checks.

Every frame here is shaped like what the executor actually returns -- numeric
columns as Decimal objects, timestamptz as tz-aware datetime64, `::date` as
datetime.date, counts as int64 -- because a check that only works on tidy float
columns would pass these tests and miss real results.

Ranges and row counts come from the profile in schema_cache.json, read straight
off disk with no database involved. That file is gitignored, so the fixture
skips with the command that rebuilds it rather than inventing numbers.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from queryguard.config import schema_cache_path
from queryguard.executor import OUTCOME_FAILED, OUTCOME_OK, ExecutionResult
from queryguard.schema.introspect import ColumnInfo, DatabaseSchema
from queryguard.validation.sanity import (
    CHECK_AGGREGATE,
    CHECK_CONSTANT,
    CHECK_DATE_SPAN,
    CHECK_DUPLICATES,
    CHECK_EMPTY,
    CHECK_NEGATIVE,
    CHECK_NULL_HEAVY,
    CHECK_REVENUE_STATUS,
    FAIL,
    INFO,
    WARN,
    SanityFlag,
    check_result,
)


@pytest.fixture(scope="module")
def schema() -> DatabaseSchema:
    path = schema_cache_path()
    if not path.is_file():
        pytest.skip(
            f"{path.name} not found, run `uv run python -m queryguard.schema.introspect --refresh`"
        )
    return DatabaseSchema.model_validate_json(path.read_text(encoding="utf-8"))


def _profile(schema: DatabaseSchema, table: str, column: str) -> ColumnInfo:
    return next(c for c in schema.table(table).columns if c.name == column)


def _ok(frame: pd.DataFrame) -> ExecutionResult:
    return ExecutionResult(
        outcome=OUTCOME_OK, sql_sha256="0" * 64, rows=frame, row_count=len(frame)
    )


def _flags(schema, frame, sql, question="list the orders") -> list[SanityFlag]:
    return check_result(question, sql, _ok(frame), schema)


def _only(flags: list[SanityFlag], check: str) -> list[SanityFlag]:
    return [f for f in flags if f.check == check]


def _utc(*stamps: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(stamps), utc=True))


# ------------------------------------------------------------------- empty


def test_an_empty_result_for_a_listing_question_warns(schema) -> None:
    frame = pd.DataFrame({"order_id": pd.Series([], dtype="int64"), "status": pd.Series([], dtype="str")})
    flags = _flags(
        schema, frame,
        "SELECT o.order_id, o.status FROM orders AS o WHERE o.total_amount > 50000",
        "List the orders over $50,000",
    )
    [flag] = _only(flags, CHECK_EMPTY)
    assert flag.severity == WARN
    assert "List the orders over $50,000" in flag.explanation


def test_an_empty_result_for_a_yes_no_question_is_only_info(schema) -> None:
    frame = pd.DataFrame({"refund_id": pd.Series([], dtype="int64")})
    flags = _flags(
        schema, frame,
        "SELECT r.refund_id FROM refunds AS r WHERE r.amount > 50000",
        "Are there any refunds over $50,000?",
    )
    [flag] = _only(flags, CHECK_EMPTY)
    assert flag.severity == INFO


def test_an_empty_result_names_a_case_mismatched_status(schema) -> None:
    """The model wrote 'Cancelled'; the table stores 'cancelled'."""
    frame = pd.DataFrame({"order_id": pd.Series([], dtype="int64")})
    flags = _flags(
        schema, frame, "SELECT o.order_id FROM orders AS o WHERE o.status = 'Cancelled'",
        "Show cancelled orders",
    )
    [flag] = _only(flags, CHECK_EMPTY)
    stored = next(v for v in _profile(schema, "orders", "status").enum_values if v.lower() == "cancelled")
    assert f"'{stored}'" in flag.explanation
    assert "orders.status" in flag.explanation


def test_an_empty_result_names_a_date_filter_beyond_the_data(schema) -> None:
    frame = pd.DataFrame({"order_id": pd.Series([], dtype="int64")})
    flags = _flags(
        schema, frame, "SELECT o.order_id FROM orders AS o WHERE o.order_date >= DATE '2030-01-01'",
        "Show orders from 2030",
    )
    [flag] = _only(flags, CHECK_EMPTY)
    assert "2030-01-01 is after the latest date" in flag.explanation


def test_a_single_row_of_null_aggregates_is_an_empty_result(schema) -> None:
    """sum() over nothing is one NULL row, not zero rows."""
    frame = pd.DataFrame({"revenue": [None], "orders": pd.Series([0], dtype="int64")})
    flags = _flags(
        schema, frame,
        "SELECT sum(o.total_amount) AS revenue, count(*) AS orders FROM orders AS o "
        "WHERE o.order_date >= DATE '2031-01-01'",
        "What was revenue in 2031?",
    )
    [flag] = _only(flags, CHECK_EMPTY)
    assert flag.severity == WARN
    assert "empty aggregates" in flag.explanation


def test_a_count_of_zero_is_an_empty_result_with_the_case_hint(schema) -> None:
    """The live run's injection (d): count(*) = 0 is one row, but it matched nothing."""
    frame = pd.DataFrame({"cancelled_orders": pd.Series([0], dtype="int64")})
    flags = _flags(
        schema, frame,
        "SELECT count(*) AS cancelled_orders FROM orders AS o WHERE o.status = 'Cancelled'",
        "How many orders were cancelled?",
    )
    [flag] = _only(flags, CHECK_EMPTY)
    assert flag.severity == WARN
    assert "'Cancelled' is not a stored value of orders.status" in flag.explanation


def test_a_coalesced_sum_of_zero_is_an_empty_result(schema) -> None:
    frame = pd.DataFrame({"revenue": [Decimal("0.00")]})
    sql = (
        "SELECT coalesce(sum(o.total_amount), 0) AS revenue FROM orders AS o "
        "WHERE o.order_date >= DATE '2031-01-01'"
    )
    [flag] = _only(_flags(schema, frame, sql, "What was revenue in 2031?"), CHECK_EMPTY)
    assert "2031-01-01 is after the latest date" in flag.explanation


def test_a_zero_count_for_a_yes_no_question_is_only_info(schema) -> None:
    frame = pd.DataFrame({"n": pd.Series([0], dtype="int64")})
    sql = "SELECT count(*) AS n FROM refunds AS r WHERE r.amount > 50000"
    [flag] = _only(_flags(schema, frame, sql, "Were there any refunds over $50,000?"), CHECK_EMPTY)
    assert flag.severity == INFO


def test_a_nonzero_count_is_not_empty(schema) -> None:
    frame = pd.DataFrame({"n": pd.Series([178], dtype="int64")})
    sql = "SELECT count(*) AS n FROM orders AS o WHERE o.status = 'cancelled'"
    assert _only(_flags(schema, frame, sql, "How many orders were cancelled?"), CHECK_EMPTY) == []


def test_a_zero_count_beside_a_real_value_is_not_empty(schema) -> None:
    """Only a row whose every aggregate is empty matched nothing."""
    frame = pd.DataFrame({"refunds": pd.Series([0], dtype="int64"), "orders": pd.Series([5000], dtype="int64")})
    sql = (
        "SELECT count(r.refund_id) AS refunds, count(*) AS orders FROM orders AS o "
        "LEFT JOIN refunds AS r ON r.order_id = o.order_id AND r.amount > 50000"
    )
    assert _only(_flags(schema, frame, sql), CHECK_EMPTY) == []


def test_a_failed_execution_has_nothing_to_check(schema) -> None:
    failed = ExecutionResult(outcome=OUTCOME_FAILED, sql_sha256="0" * 64, sqlstate="42P01")
    assert check_result("list orders", "SELECT * FROM nope", failed, schema) == []


# --------------------------------------------------------------- NULL-heavy


def test_nulls_in_a_not_null_column_point_at_the_join(schema) -> None:
    """A LEFT JOIN that mostly missed: orders.total_amount is NOT NULL in the table."""
    assert not _profile(schema, "orders", "total_amount").nullable
    frame = pd.DataFrame({
        "customer_id": pd.Series(range(1, 11), dtype="int64"),
        "total_amount": [Decimal("120.50"), Decimal("88.00"), Decimal("310.25")] + [None] * 7,
    })
    flags = _flags(
        schema, frame,
        "SELECT c.customer_id, o.total_amount FROM customers AS c "
        "LEFT JOIN orders AS o ON o.order_id = c.customer_id",
    )
    [flag] = _only(flags, CHECK_NULL_HEAVY)
    assert flag.severity == WARN
    assert flag.column == "total_amount"
    assert "70% NULL" in flag.explanation
    assert "NOT NULL" in flag.explanation


_TOP_BY_LIFETIME_VALUE = (
    "SELECT c.customer_id, c.lifetime_value AS total_spent FROM customers AS c "
    "ORDER BY c.lifetime_value DESC LIMIT 5"
)


def _top_five_nulls() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": pd.Series([5, 481, 482, 483, 484], dtype="int64"),
        "total_spent": [None] * 5,
    })


def test_nulls_from_a_desc_sort_are_explained_as_nulls_first(schema) -> None:
    """The live run's injection (c): DESC put the NULL lifetime values on top."""
    assert _profile(schema, "customers", "lifetime_value").nullable
    [flag] = _only(_flags(schema, _top_five_nulls(), _TOP_BY_LIFETIME_VALUE), CHECK_NULL_HEAVY)
    assert flag.severity == WARN
    assert "Postgres sorts NULLs first in DESC" in flag.explanation
    assert "NULLS LAST" in flag.explanation
    assert "outer join" not in flag.explanation


def test_desc_with_nulls_last_is_not_blamed_on_the_sort(schema) -> None:
    sql = _TOP_BY_LIFETIME_VALUE.replace("DESC", "DESC NULLS LAST")
    [flag] = _only(_flags(schema, _top_five_nulls(), sql), CHECK_NULL_HEAVY)
    assert "NULLs first" not in flag.explanation
    assert "outer join" in flag.explanation


def test_an_ascending_sort_is_not_blamed_for_nulls(schema) -> None:
    sql = _TOP_BY_LIFETIME_VALUE.replace(" DESC", "")
    [flag] = _only(_flags(schema, _top_five_nulls(), sql), CHECK_NULL_HEAVY)
    assert "NULLs first" not in flag.explanation


def test_a_column_that_is_mostly_null_in_the_table_is_only_info(schema) -> None:
    profile = _profile(schema, "orders", "discount_code")
    assert profile.null_fraction > 0.5, "premise: discount_code is mostly NULL in the data"
    frame = pd.DataFrame({
        "order_id": pd.Series(range(1, 11), dtype="int64"),
        "discount_code": ["SAVE5", "LOYAL15", "WELCOME10"] + [None] * 7,
    })
    [flag] = _only(_flags(schema, frame, "SELECT o.order_id, o.discount_code FROM orders AS o"), CHECK_NULL_HEAVY)
    assert flag.severity == INFO
    assert f"{profile.null_fraction:.0%} NULL in the table" in flag.explanation


def test_exactly_half_null_is_not_null_heavy(schema) -> None:
    frame = pd.DataFrame({
        "customer_id": pd.Series(range(1, 9), dtype="int64"),
        "phone": ["+1-200-3376", None, "+1-200-6100", None, "+1-202-7692", None, "+1-202-0001", None],
    })
    assert _only(_flags(schema, frame, "SELECT c.customer_id, c.phone FROM customers AS c"), CHECK_NULL_HEAVY) == []


def test_a_tiny_result_is_too_small_to_call_null_heavy(schema) -> None:
    frame = pd.DataFrame({"customer_id": pd.Series([1, 2, 3], dtype="int64"), "city": [None, None, "Lyon"]})
    assert _only(_flags(schema, frame, "SELECT c.customer_id, c.city FROM customers AS c"), CHECK_NULL_HEAVY) == []


# ----------------------------------------------------------------- constant


def test_one_date_in_every_row_is_the_seed_bug_signature(schema) -> None:
    """Phase 0's seed evaluated a volatile expression once; every row matched."""
    frame = pd.DataFrame({
        "order_id": pd.Series(range(1, 1001), dtype="int64"),
        "order_date": _utc(*["2025-03-14 09:00:00"] * 1000),
    })
    flags = _flags(schema, frame, "SELECT o.order_id, o.order_date FROM orders AS o")
    [flag] = _only(flags, CHECK_CONSTANT)
    assert flag.severity == WARN
    assert flag.column == "order_date"
    assert "1000 rows" in flag.explanation
    assert "seed bug" in flag.explanation


def test_a_column_the_query_filters_on_may_be_constant(schema) -> None:
    frame = pd.DataFrame({
        "order_id": pd.Series(range(1, 21), dtype="int64"),
        "status": ["cancelled"] * 20,
    })
    sql = "SELECT o.order_id, o.status FROM orders AS o WHERE o.status = 'cancelled'"
    assert _only(_flags(schema, frame, sql), CHECK_CONSTANT) == []


def test_a_boolean_filter_also_explains_a_constant_column(schema) -> None:
    frame = pd.DataFrame({"customer_id": pd.Series(range(1, 11), dtype="int64"), "is_active": [True] * 10})
    sql = "SELECT c.customer_id, c.is_active FROM customers AS c WHERE c.is_active"
    assert _only(_flags(schema, frame, sql), CHECK_CONSTANT) == []


def test_a_column_with_one_value_in_the_table_may_be_constant(schema) -> None:
    """orders.currency holds a single value in the profile, so this is just the data."""
    assert _profile(schema, "orders", "currency").distinct_count == 1
    frame = pd.DataFrame({
        "order_id": pd.Series(range(1, 51), dtype="int64"),
        "currency": _profile(schema, "orders", "currency").enum_values * 50,
    })
    assert _only(_flags(schema, frame, "SELECT o.order_id, o.currency FROM orders AS o"), CHECK_CONSTANT) == []


def test_five_identical_rows_are_not_enough(schema) -> None:
    frame = pd.DataFrame({"product_id": pd.Series(range(1, 6), dtype="int64"), "in_stock": [True] * 5})
    assert _only(_flags(schema, frame, "SELECT p.product_id, p.in_stock FROM products AS p"), CHECK_CONSTANT) == []


# -------------------------------------------------------------------- dates


def test_a_date_after_the_data_is_flagged_with_the_profiled_span(schema) -> None:
    profile = _profile(schema, "orders", "order_date")
    frame = pd.DataFrame({
        "order_id": pd.Series([1, 2, 3], dtype="int64"),
        "order_date": _utc("2025-01-10 10:00:00", "2025-02-11 11:00:00", "2031-06-01 00:00:00"),
    })
    [flag] = _only(_flags(schema, frame, "SELECT o.order_id, o.order_date FROM orders AS o"), CHECK_DATE_SPAN)
    assert flag.severity == WARN
    assert "1 value(s)" in flag.explanation
    assert pd.Timestamp(profile.min_value).strftime("%Y-%m-%d") in flag.explanation
    assert pd.Timestamp(profile.max_value).strftime("%Y-%m-%d") in flag.explanation
    assert "2031-06-01" in flag.explanation


def test_month_truncation_may_start_before_the_first_order(schema) -> None:
    """date_trunc('month') of the earliest order lands on the 1st, before it."""
    first = pd.Timestamp(_profile(schema, "orders", "order_date").min_value)
    month_start = first.replace(day=1, hour=0, minute=0, second=0)
    assert month_start < first, "premise: the data does not start on the 1st"
    frame = pd.DataFrame({
        "month": _utc(str(month_start), str(month_start + pd.DateOffset(months=1))),
        "orders": pd.Series([40, 160], dtype="int64"),
    })
    sql = (
        "SELECT date_trunc('month', o.order_date) AS month, count(*) AS orders "
        "FROM orders AS o GROUP BY 1 ORDER BY 1"
    )
    assert _only(_flags(schema, frame, sql), CHECK_DATE_SPAN) == []


def test_a_date_cast_outside_the_span_is_flagged(schema) -> None:
    """`::date` comes back as datetime.date objects, not datetime64."""
    frame = pd.DataFrame({
        "refunded_on": [date(2024, 5, 1), date(2019, 1, 1)],
        "amount": [Decimal("250.00"), Decimal("410.10")],
    })
    sql = "SELECT r.refunded_at::date AS refunded_on, r.amount FROM refunds AS r"
    [flag] = _only(_flags(schema, frame, sql), CHECK_DATE_SPAN)
    assert flag.column == "refunded_on"
    assert "refunds.refunded_at" in flag.explanation
    assert "2019-01-01" in flag.explanation


def test_a_computed_date_is_checked_against_the_tables_it_reads(schema) -> None:
    """No source column, so the span of every date column in orders applies."""
    frame = pd.DataFrame({
        "order_id": pd.Series([1, 2], dtype="int64"),
        "due": _utc("2025-01-01", "1999-12-31"),
    })
    sql = "SELECT o.order_id, o.order_date - INTERVAL '30 years' AS due FROM orders AS o"
    [flag] = _only(_flags(schema, frame, sql), CHECK_DATE_SPAN)
    assert "any date column" in flag.explanation


# ----------------------------------------------------------------- negatives


def test_a_negative_count_is_a_failure(schema) -> None:
    frame = pd.DataFrame({"status": ["paid", "refunded"], "order_count": pd.Series([812, -3], dtype="int64")})
    sql = "SELECT o.status, count(*) AS order_count FROM orders AS o GROUP BY o.status"
    [flag] = _only(_flags(schema, frame, sql), CHECK_NEGATIVE)
    assert flag.severity == FAIL
    assert "count()" in flag.explanation


def test_a_negative_sum_of_a_non_negative_column_is_a_failure(schema) -> None:
    low = _profile(schema, "order_items", "quantity").min_value
    frame = pd.DataFrame({"product_id": pd.Series([4, 9], dtype="int64"), "units": [Decimal("31"), Decimal("-12")]})
    sql = "SELECT oi.product_id, sum(oi.quantity) AS units FROM order_items AS oi GROUP BY 1"
    [flag] = _only(_flags(schema, frame, sql), CHECK_NEGATIVE)
    assert flag.severity == FAIL
    assert f"order_items.quantity, whose smallest value is {low}" in flag.explanation


def test_a_negative_net_figure_is_legitimate(schema) -> None:
    frame = pd.DataFrame({"month": _utc("2025-01-01", "2025-02-01"), "net_revenue": [Decimal("5120.00"), Decimal("-310.40")]})
    sql = (
        "SELECT date_trunc('month', o.order_date) AS month, "
        "sum(o.total_amount) - sum(r.amount) AS net_revenue "
        "FROM orders AS o LEFT JOIN refunds AS r ON r.order_id = o.order_id GROUP BY 1"
    )
    assert _only(_flags(schema, frame, sql), CHECK_NEGATIVE) == []


def test_a_negative_computed_amount_warns_on_its_name_alone(schema) -> None:
    frame = pd.DataFrame({"order_id": pd.Series([1, 2], dtype="int64"), "line_amount": [Decimal("10.00"), Decimal("-4.50")]})
    sql = "SELECT oi.order_id, oi.unit_price * oi.quantity - oi.discount AS line_amount FROM order_items AS oi"
    [flag] = _only(_flags(schema, frame, sql), CHECK_NEGATIVE)
    assert flag.severity == WARN


# ---------------------------------------------------------------- aggregates


def test_a_sum_beyond_max_times_row_count_is_a_fan_out(schema) -> None:
    profile = _profile(schema, "orders", "total_amount")
    ceiling = float(profile.max_value) * schema.table("orders").row_count
    frame = pd.DataFrame({"revenue": [Decimal(str(round(ceiling * 1.5, 2)))]})
    sql = (
        "SELECT sum(o.total_amount) AS revenue FROM orders AS o "
        "JOIN order_items AS oi ON oi.order_id = o.order_id"
    )
    [flag] = _only(_flags(schema, frame, sql, "What is total revenue?"), CHECK_AGGREGATE)
    assert flag.severity == FAIL
    assert "sum(orders.total_amount)" in flag.explanation
    assert f"{schema.table('orders').row_count:,} rows" in flag.explanation


def test_a_sum_within_the_ceiling_is_fine(schema) -> None:
    frame = pd.DataFrame({"country": ["USA", "Japan"], "revenue": [Decimal("2350112.40"), Decimal("1904413.05")]})
    sql = (
        "SELECT c.country, round(sum(o.total_amount), 2) AS revenue FROM orders AS o "
        "JOIN customers AS c ON c.customer_id = o.customer_id GROUP BY c.country"
    )
    assert _only(_flags(schema, frame, sql), CHECK_AGGREGATE) == []


def test_a_count_larger_than_any_table_read_warns(schema) -> None:
    largest = max(schema.table(t).row_count for t in ("orders", "order_items", "refunds"))
    frame = pd.DataFrame({"n": pd.Series([largest * 3], dtype="int64")})
    sql = (
        "SELECT count(*) AS n FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id "
        "JOIN refunds AS r ON r.order_id = o.order_id"
    )
    [flag] = _only(_flags(schema, frame, sql, "How many order lines were refunded?"), CHECK_AGGREGATE)
    assert flag.severity == WARN
    assert f"{largest:,}" in flag.explanation


def test_an_average_above_the_column_maximum_warns(schema) -> None:
    high = float(_profile(schema, "orders", "total_amount").max_value)
    frame = pd.DataFrame({"status": ["paid"], "avg_value": [Decimal(str(round(high * 2, 2)))]})
    sql = "SELECT o.status, round(avg(o.total_amount), 2) AS avg_value FROM orders AS o GROUP BY 1"
    [flag] = _only(_flags(schema, frame, sql), CHECK_AGGREGATE)
    assert flag.severity == WARN
    assert "avg(orders.total_amount)" in flag.explanation


def test_a_ratio_of_aggregates_is_not_read_as_a_sum(schema) -> None:
    """sum(x) / count(*) is a mean; comparing it to a sum ceiling would be wrong either way."""
    frame = pd.DataFrame({"per_order": [Decimal("3777.41")]})
    sql = "SELECT sum(o.total_amount) / count(*) AS per_order FROM orders AS o"
    assert _only(_flags(schema, frame, sql), CHECK_AGGREGATE) == []


# --------------------------------------------------------------- duplicates


def _fanned_out_orders() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": pd.Series([7, 7, 7, 12, 12], dtype="int64"),
        "total_amount": [Decimal("412.30")] * 3 + [Decimal("99.00")] * 2,
    })


def test_duplicate_rows_with_a_primary_key_mean_a_join_repeated_them(schema) -> None:
    sql = (
        "SELECT o.order_id, o.total_amount FROM orders AS o "
        "JOIN order_items AS oi ON oi.order_id = o.order_id"
    )
    [flag] = _only(_flags(schema, _fanned_out_orders(), sql), CHECK_DUPLICATES)
    assert flag.severity == WARN
    assert "3 of 5 rows" in flag.explanation
    assert "orders.order_id" in flag.explanation


def test_distinct_rules_out_the_duplicate_check(schema) -> None:
    sql = (
        "SELECT DISTINCT o.order_id, o.total_amount FROM orders AS o "
        "JOIN order_items AS oi ON oi.order_id = o.order_id"
    )
    assert _only(_flags(schema, _fanned_out_orders(), sql), CHECK_DUPLICATES) == []


def test_duplicates_without_a_primary_key_are_not_flagged(schema) -> None:
    """Two orders from the same country on the same status is just data."""
    frame = pd.DataFrame({"country": ["USA", "USA", "Japan"], "status": ["paid", "paid", "paid"]})
    sql = "SELECT c.country, o.status FROM orders AS o JOIN customers AS c ON c.customer_id = o.customer_id"
    assert _only(_flags(schema, frame, sql), CHECK_DUPLICATES) == []


def test_a_foreign_key_column_is_not_a_primary_key(schema) -> None:
    """oi.order_id repeats by design: order_items has several lines per order."""
    frame = pd.DataFrame({"order_id": pd.Series([7, 7], dtype="int64"), "product_id": pd.Series([3, 3], dtype="int64")})
    sql = "SELECT oi.order_id, oi.product_id FROM order_items AS oi"
    assert _only(_flags(schema, frame, sql), CHECK_DUPLICATES) == []


# ------------------------------------------------------------ revenue status


def _one_total(name: str = "gross_revenue") -> pd.DataFrame:
    return pd.DataFrame({name: [Decimal("6177714.13")]})


_GROSS_2025 = (
    "SELECT sum(o.total_amount) AS gross_revenue FROM orders AS o "
    "WHERE {status}o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'"
)
_GROSS_Q = "What was gross revenue from orders placed in 2025, before refunds?"


def test_revenue_without_a_status_filter_is_flagged_with_the_glossary_rule(schema) -> None:
    # refund_04 as generated in live-2026-10-01: every 2025 order, pending included.
    flags = _only(_flags(schema, _one_total(), _GROSS_2025.format(status=""), _GROSS_Q), CHECK_REVENUE_STATUS)
    assert len(flags) == 1
    assert flags[0].severity == WARN
    assert flags[0].column == "gross_revenue"
    assert "'pending' and 'cancelled'" in flags[0].explanation
    assert "pending and cancelled orders are not revenue" in flags[0].explanation


def test_a_status_filter_dropped_inside_a_cte_is_still_flagged(schema) -> None:
    # refund_01's drop_where mutation: the sum reads the CTE, the CTE reads orders.
    sql = (
        "WITH paid_orders AS (SELECT o.order_id, o.total_amount FROM orders AS o "
        "WHERE o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01') "
        "SELECT (SELECT sum(po.total_amount) FROM paid_orders AS po) - (SELECT coalesce(sum(r.amount), 0) "
        "FROM refunds AS r JOIN paid_orders AS po ON po.order_id = r.order_id) AS net_revenue"
    )
    frame = _one_total("net_revenue")
    flags = _only(_flags(schema, frame, sql, "What was net revenue from orders placed in 2025, after refunds?"),
                  CHECK_REVENUE_STATUS)
    assert len(flags) == 1


@pytest.mark.parametrize("status", [
    "o.status IN ('paid', 'shipped', 'delivered', 'refunded') AND ",
    "o.status = 'delivered' AND ",
    "o.status NOT IN ('pending', 'cancelled') AND ",
    "o.status <> 'pending' AND o.status != 'cancelled' AND ",
])
def test_a_filter_excluding_pending_and_cancelled_is_not_flagged(schema, status) -> None:
    flags = _flags(schema, _one_total(), _GROSS_2025.format(status=status), _GROSS_Q)
    assert _only(flags, CHECK_REVENUE_STATUS) == []


def test_excluding_only_cancelled_still_flags_pending(schema) -> None:
    sql = _GROSS_2025.format(status="o.status <> 'cancelled' AND ")
    flags = _only(_flags(schema, _one_total(), sql, _GROSS_Q), CHECK_REVENUE_STATUS)
    assert len(flags) == 1
    assert "'pending' orders" in flags[0].explanation


def test_a_total_that_is_not_called_revenue_is_left_alone(schema) -> None:
    # date_02: "total order amount" is every order's amount, by the question's own words.
    sql = ("SELECT sum(o.total_amount) AS total FROM orders AS o "
           "WHERE o.order_date >= DATE '2026-03-01' AND o.order_date < DATE '2026-04-01'")
    flags = _flags(schema, _one_total("total"), sql,
                   "What was the total order amount for orders placed in March 2026?")
    assert _only(flags, CHECK_REVENUE_STATUS) == []


def test_a_revenue_sum_over_another_column_is_left_alone(schema) -> None:
    sql = "SELECT sum(r.amount) AS refunded_revenue FROM refunds AS r"
    flags = _flags(schema, _one_total("refunded_revenue"), sql, "How much revenue was refunded?")
    assert _only(flags, CHECK_REVENUE_STATUS) == []


# ------------------------------------------------------------------ overall


def test_a_clean_result_raises_no_flags(schema) -> None:
    frame = pd.DataFrame({
        "category": ["Electronics", "Apparel", "Books"],
        "product_count": pd.Series([31, 24, 18], dtype="int64"),
    })
    sql = (
        "SELECT cat.name AS category, count(*) AS product_count FROM products AS p "
        "JOIN categories AS cat ON cat.category_id = p.category_id GROUP BY cat.name"
    )
    assert _flags(schema, frame, sql, "How many products per category?") == []


def test_flags_are_ordered_most_severe_first(schema) -> None:
    frame = pd.DataFrame({
        "order_id": pd.Series(range(1, 11), dtype="int64"),
        "shipped_at": _utc(*["2025-04-01"] * 3).tolist() + [pd.NaT] * 7,
        "order_count": pd.Series([1] * 9 + [-1], dtype="int64"),
    })
    sql = (
        "SELECT o.order_id, o.shipped_at, count(*) AS order_count FROM orders AS o "
        "LEFT JOIN order_items AS oi ON oi.order_id = o.order_id GROUP BY 1, 2"
    )
    flags = _flags(schema, frame, sql)
    assert [f.severity for f in flags] == sorted(
        (f.severity for f in flags), key=[FAIL, WARN, INFO].index
    )
    assert flags[0].check == CHECK_NEGATIVE


def test_without_a_schema_only_the_structural_checks_run() -> None:
    frame = pd.DataFrame({
        "order_id": pd.Series(range(1, 11), dtype="int64"),
        "order_date": _utc(*["2031-01-01"] * 10),
    })
    flags = check_result("list orders", "SELECT o.order_id, o.order_date FROM orders AS o", _ok(frame), None)
    assert [f.check for f in flags] == [CHECK_CONSTANT]
