"""Tests for the golden eval set and its mutation negatives.

The mutators are pure string functions and are tested without a database. The
integrity tests read the committed files -- golden.yaml, golden_results.json,
mutation_results.json -- so an edit to one that is not followed by a re-run of
evals.run_golden / evals.mutations fails here rather than silently skewing an
eval. Nothing here calls the API or the database.
"""

from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal

import pandas as pd
import pytest

from evals.common import GOLDEN_RESULTS, MUTATION_RESULTS, load_golden, result_hash
from evals.mutations import (
    MUTATORS,
    agg_swap,
    column_swap,
    date_shift,
    drop_where,
    fan_out_join,
    inner_to_left,
    literal_case,
    null_flip,
    order_flip,
)
from queryguard.llm.examples import EXAMPLES

GOLDEN = load_golden()


# ------------------------------------------------------------------ the YAML


def test_fifty_questions_across_eight_categories() -> None:
    counts = Counter(e["category"] for e in GOLDEN)
    assert len(GOLDEN) == 50
    assert set(counts) == {
        "simple_lookup", "join", "aggregation", "date_range", "top_n",
        "refund_trap", "ambiguous", "unanswerable",
    }
    assert all(5 <= n <= 8 for n in counts.values()), counts


def test_ids_are_unique_and_every_entry_names_its_trap() -> None:
    assert len({e["id"] for e in GOLDEN}) == len(GOLDEN)
    assert all(e.get("notes") for e in GOLDEN)


def test_each_entry_has_sql_or_an_expected_outcome_never_both() -> None:
    for e in GOLDEN:
        assert ("golden_sql" in e) != ("expected_outcome" in e), e["id"]
        if e["category"] == "ambiguous":
            assert e["expected_outcome"] == "clarification"
        if e["category"] == "unanswerable":
            assert e["expected_outcome"] == "refusal_or_clarification"


def test_no_golden_question_is_a_few_shot_example() -> None:
    """The prompt has memorised its examples; testing on them measures nothing."""
    examples = {ex.question.lower().rstrip("?.") for ex in EXAMPLES}
    assert not [e["id"] for e in GOLDEN if e["question"].lower().rstrip("?.") in examples]


def test_golden_sql_never_depends_on_the_clock() -> None:
    for e in GOLDEN:
        sql = e.get("golden_sql", "").lower()
        assert "now()" not in sql and "current_date" not in sql, e["id"]


# --------------------------------------------------------- the stored results


def test_every_golden_entry_has_a_stored_result() -> None:
    results = json.loads(GOLDEN_RESULTS.read_text())["results"]
    assert set(results) == {e["id"] for e in GOLDEN}
    for e in GOLDEN:
        if "golden_sql" in e:
            assert results[e["id"]]["row_count"] > 0 or e.get("empty_ok"), e["id"]
            assert len(results[e["id"]]["result_sha256"]) == 64


def test_kept_mutations_are_labelled_wrong_and_differ_from_golden() -> None:
    golden = json.loads(GOLDEN_RESULTS.read_text())["results"]
    mutations = json.loads(MUTATION_RESULTS.read_text())
    assert mutations["kept"], "no negatives were generated"
    for m in mutations["kept"]:
        assert m["label"] == "wrong"
        assert m["mutation"] in MUTATORS
        assert m["diff"].startswith(("disagree", "incomparable")), m["id"]
        assert m["result_sha256"] != golden[m["golden_id"]]["result_sha256"], m["id"]
    per_question = Counter(m["golden_id"] for m in mutations["kept"])
    assert max(per_question.values()) <= 3


def test_no_discarded_mutation_is_also_kept() -> None:
    mutations = json.loads(MUTATION_RESULTS.read_text())
    kept = {(m["golden_id"], m["mutation"]) for m in mutations["kept"]}
    discarded = {(m["golden_id"], m["mutation"]) for m in mutations["discarded"]}
    assert not kept & discarded


# ------------------------------------------------------------------- hashing


def test_the_hash_ignores_row_order_names_and_numeric_type_unless_ordered() -> None:
    a = pd.DataFrame({"x": ["b", "a"], "n": [Decimal("3.10"), Decimal("2")]})
    b = pd.DataFrame({"y": ["a", "b"], "m": [2, 3.1]})
    assert result_hash(a, ordered=False) == result_hash(b, ordered=False)
    assert result_hash(a, ordered=True) != result_hash(b, ordered=True)


def test_the_hash_normalises_timezones() -> None:
    a = pd.DataFrame({"t": pd.to_datetime(["2025-01-01 10:00"], utc=True)})
    b = pd.DataFrame({"t": pd.to_datetime(["2025-01-01 11:00+01:00"])})
    assert result_hash(a, ordered=False) == result_hash(b, ordered=False)


# ------------------------------------------------------------------ mutators


def test_literal_case_flips_a_status_but_never_a_date() -> None:
    sql = "SELECT count(*) FROM orders AS o WHERE o.order_date >= DATE '2025-01-01' AND o.status = 'cancelled'"
    assert literal_case(sql).endswith("o.status = 'Cancelled'")
    assert "DATE '2025-01-01'" in literal_case(sql)
    assert literal_case("SELECT 1 FROM orders AS o WHERE o.order_date >= DATE '2025-01-01'") is None
    assert "'sku-00042'" in literal_case("SELECT p.price FROM products AS p WHERE p.sku = 'SKU-00042'")


def test_drop_where_prefers_a_non_date_condition() -> None:
    sql = (
        "SELECT count(*) FROM orders AS o\n"
        "WHERE o.status <> 'cancelled'\n  AND o.order_date >= DATE '2025-01-01'\n"
        "GROUP BY o.status"
    )
    mutated = drop_where(sql)
    assert "cancelled" not in mutated
    assert "o.order_date >= DATE '2025-01-01'" in mutated
    assert "GROUP BY o.status" in mutated


def test_drop_where_removes_a_lone_condition_entirely() -> None:
    mutated = drop_where("SELECT c.email FROM customers AS c WHERE c.customer_id = 117")
    assert "WHERE" not in mutated and mutated.strip().endswith("customers AS c")


def test_agg_swap_order_of_preference() -> None:
    assert "count(o.customer_id)" in agg_swap("SELECT count(DISTINCT o.customer_id) FROM orders AS o")
    assert agg_swap("SELECT sum(r.amount) FROM refunds AS r").startswith("SELECT avg(")
    assert agg_swap("SELECT avg(p.price) FROM products AS p").startswith("SELECT sum(")
    grouped = agg_swap("SELECT r.approved_by, count(*) FROM refunds AS r GROUP BY r.approved_by")
    assert "count(r.approved_by)" in grouped
    assert agg_swap("SELECT c.email FROM customers AS c") is None


def test_column_swap_needs_the_neighbour_in_scope() -> None:
    joined = (
        "SELECT sum(o.total_amount) FROM orders AS o "
        "JOIN customers AS c ON c.customer_id = o.customer_id"
    )
    assert "sum(c.lifetime_value)" in column_swap(joined)
    # Same-table neighbour: order_date -> shipped_at, every occurrence.
    dated = "SELECT count(*) FROM orders AS o WHERE o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'"
    assert column_swap(dated).count("o.shipped_at") == 2
    # total_amount's neighbour lives in customers, which is not joined here.
    assert column_swap("SELECT o.total_amount FROM orders AS o") is None


def test_inner_to_left_skips_joins_that_are_already_outer() -> None:
    assert inner_to_left("SELECT 1 FROM a AS x JOIN b AS y ON y.id = x.id").count("LEFT JOIN") == 1
    assert "LEFT JOIN b" in inner_to_left("SELECT 1 FROM a AS x INNER JOIN b AS y ON true")
    two = "SELECT 1 FROM a AS x LEFT JOIN b AS y ON true JOIN c AS z ON true"
    assert inner_to_left(two).count("LEFT JOIN") == 2
    assert inner_to_left("SELECT 1 FROM a AS x LEFT JOIN b AS y ON true") is None


def test_date_shift_moves_the_first_bound_back_a_year() -> None:
    sql = "WHERE o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'"
    assert date_shift(sql) == "WHERE o.order_date >= DATE '2024-01-01' AND o.order_date < DATE '2026-01-01'"


def test_order_flip_flips_only_the_first_key() -> None:
    assert order_flip("SELECT 1 ORDER BY t DESC, id LIMIT 10") == "SELECT 1 ORDER BY t ASC, id LIMIT 10"
    assert order_flip("SELECT 1 ORDER BY t LIMIT 3") == "SELECT 1 ORDER BY t DESC LIMIT 3"
    assert order_flip("SELECT 1") is None


def test_fan_out_join_adds_a_child_unless_it_is_already_there() -> None:
    mutated = fan_out_join("SELECT o.status, count(*) FROM orders AS o GROUP BY o.status")
    assert "JOIN order_items AS fan_order_items ON fan_order_items.order_id = o.order_id" in mutated
    assert mutated.index("JOIN") < mutated.index("GROUP BY")
    assert fan_out_join("SELECT 1 FROM orders AS o JOIN order_items AS oi ON oi.order_id = o.order_id") is None


def test_null_flip_inverts_anti_joins() -> None:
    assert null_flip("WHERE o.order_id IS NULL") == "WHERE o.order_id IS NOT NULL"
    assert null_flip("WHERE r.approved_by IS NOT NULL") == "WHERE r.approved_by IS NULL"
    assert null_flip("WHERE NOT EXISTS (SELECT 1)") == "WHERE EXISTS (SELECT 1)"


@pytest.mark.parametrize("name", list(MUTATORS))
def test_every_mutator_returns_none_when_it_does_not_apply(name) -> None:
    assert MUTATORS[name]("SELECT 1") is None
