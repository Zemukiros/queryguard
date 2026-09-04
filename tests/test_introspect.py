"""Tests for the schema introspection layer."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from queryguard.schema.introspect import (
    ColumnInfo,
    DatabaseSchema,
    ForeignKeyInfo,
    TableInfo,
    estimate_tokens,
    load_schema,
    save_schema,
)

EXPECTED_TABLES = {
    "categories",
    "customers",
    "products",
    "orders",
    "order_items",
    "refunds",
}


def test_all_six_tables_found(live_schema: DatabaseSchema) -> None:
    assert {t.name for t in live_schema.tables} == EXPECTED_TABLES


def test_every_table_has_columns_and_rows(live_schema: DatabaseSchema) -> None:
    for table in live_schema.tables:
        assert table.columns, f"{table.name} has no columns"
        assert table.row_count > 0, f"{table.name} is empty"


def test_orders_status_enum_contains_cancelled(live_schema: DatabaseSchema) -> None:
    orders = live_schema.table("orders")
    assert orders is not None
    status = next(c for c in orders.columns if c.name == "status")

    # Six distinct values is well under the enum threshold, so the profiler must
    # have stored them all rather than falling back to examples.
    assert status.enum_values is not None
    assert status.example_values is None
    assert "cancelled" in status.enum_values


def test_high_cardinality_text_stores_examples_not_enum(live_schema: DatabaseSchema) -> None:
    """The other branch: 500 distinct emails must not be inlined."""
    customers = live_schema.table("customers")
    assert customers is not None
    email = next(c for c in customers.columns if c.name == "email")

    assert email.enum_values is None
    assert email.distinct_count == 500
    assert email.example_values is not None and len(email.example_values) == 3


def test_customers_orders_fk_detected_in_both_directions(live_schema: DatabaseSchema) -> None:
    orders = live_schema.table("orders")
    customers = live_schema.table("customers")
    assert orders is not None and customers is not None

    edge = ForeignKeyInfo(
        from_table="orders",
        from_column="customer_id",
        to_table="customers",
        to_column="customer_id",
    )

    # Forward: orders declares the constraint.
    assert edge in orders.foreign_keys
    # Reverse: customers knows it is pointed at.
    assert edge in customers.referenced_by
    # And the column itself carries the target, for inline rendering.
    customer_id = next(c for c in orders.columns if c.name == "customer_id")
    assert customer_id.references == "customers.customer_id"


def test_self_referencing_fk_appears_on_both_sides(live_schema: DatabaseSchema) -> None:
    categories = live_schema.table("categories")
    assert categories is not None
    edge = ForeignKeyInfo(
        from_table="categories",
        from_column="parent_category_id",
        to_table="categories",
        to_column="category_id",
    )
    assert edge in categories.foreign_keys
    assert edge in categories.referenced_by


def test_profiles_cover_every_type_category(live_schema: DatabaseSchema) -> None:
    customers = live_schema.table("customers")
    assert customers is not None
    by_name = {c.name: c for c in customers.columns}

    assert by_name["lifetime_value"].min_value is not None  # numeric
    assert by_name["signup_date"].min_value is not None  # timestamp
    assert by_name["is_active"].true_count is not None  # boolean
    assert by_name["is_active"].false_count is not None
    assert 0.0 < by_name["phone"].null_fraction < 1.0  # nullable text
    assert by_name["email"].null_fraction == 0.0

    assert all(c.profile_error is None for c in customers.columns)


# --------------------------------------------------------------------- render


def _assert_not_json(rendered: str) -> None:
    with pytest.raises(json.JSONDecodeError):
        json.loads(rendered)
    assert not rendered.lstrip().startswith(("{", "["))
    # The enum syntax uses bare braces, so look for JSON's punctuation digraphs
    # rather than braces alone.
    assert '{"' not in rendered
    assert '":' not in rendered
    assert "|" not in rendered  # no markdown tables either


def test_render_for_prompt_is_compact_plain_text(live_schema: DatabaseSchema) -> None:
    rendered = live_schema.render_for_prompt()

    assert rendered.strip()
    _assert_not_json(rendered)

    for name in EXPECTED_TABLES:
        assert f"\n{name} (" in "\n" + rendered

    assert "∈ {cancelled," in rendered  # enum values are inlined
    assert "→ customers.customer_id" in rendered  # outgoing FK on the column
    assert "→ orders.customer_id" in rendered  # reverse FK at table level
    assert estimate_tokens(rendered) > 0


def test_render_from_synthetic_schema_needs_no_database() -> None:
    """Rendering is pure, so it is checked without a live database too."""
    schema = DatabaseSchema(
        extracted_at=datetime.now(timezone.utc),
        tables=[
            TableInfo(
                name="widgets",
                row_count=3,
                columns=[
                    ColumnInfo(
                        name="widget_id", sql_type="integer", nullable=False, is_primary_key=True
                    ),
                    ColumnInfo(
                        name="kind",
                        sql_type="text",
                        nullable=False,
                        enum_values=["a", "b, c"],
                        distinct_count=2,
                    ),
                    ColumnInfo(
                        name="note", sql_type="text", nullable=True, null_fraction=0.34
                    ),
                ],
                referenced_by=[
                    ForeignKeyInfo(
                        from_table="gadgets",
                        from_column="widget_id",
                        to_table="widgets",
                        to_column="widget_id",
                    )
                ],
            )
        ],
    )

    rendered = schema.render_for_prompt()
    _assert_not_json(rendered)
    assert rendered.splitlines()[0] == "widgets (3 rows)"
    assert "widget_id  integer PK" in rendered
    assert "∈ {a, 'b, c'}" in rendered  # a value with a comma is quoted
    assert "note       text NULL (34% null)" in rendered
    assert "  → gadgets.widget_id" in rendered


# ---------------------------------------------------------------------- cache


def test_cache_round_trip(live_schema: DatabaseSchema, tmp_path) -> None:
    path = tmp_path / "schema_cache.json"
    save_schema(live_schema, path)

    loaded = load_schema(refresh=False, path=path)
    assert loaded == live_schema
    assert loaded.render_for_prompt() == live_schema.render_for_prompt()


def test_load_schema_builds_cache_when_missing(live_schema: DatabaseSchema, tmp_path) -> None:
    path = tmp_path / "nested" / "schema_cache.json"
    assert not path.exists()

    built = load_schema(refresh=False, path=path)
    assert path.is_file()
    assert {t.name for t in built.tables} == EXPECTED_TABLES


def test_corrupt_cache_is_rebuilt(live_schema: DatabaseSchema, tmp_path) -> None:
    path = tmp_path / "schema_cache.json"
    path.write_text("{ this is not valid json")

    rebuilt = load_schema(refresh=False, path=path)
    assert {t.name for t in rebuilt.tables} == EXPECTED_TABLES
