"""Tests for the pre-execution SQL gate.

Nothing here touches a database or the network, so this file must never skip:
the guardrail is the component whose failure is least visible in production --
a rule that quietly stops matching does not raise, it just starts saying yes --
and a suite that can skip is exactly how that goes unnoticed.

Every rejection is asserted against the rule that produced it, not merely
against `allowed is False`. `verify_db.py:expect_denied()` makes the same
distinction and for the same reason: SQL refused for the wrong reason is a bug
wearing a passing test as a disguise.
"""

from __future__ import annotations

import pytest
import sqlparse

from queryguard.guardrails import (
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_SUBQUERY_DEPTH,
    GuardrailConfig,
    GuardrailResult,
    check,
)
from queryguard.llm.examples import EXAMPLES


def _assert_rejected(sql: str, rule: str, config: GuardrailConfig | None = None) -> GuardrailResult:
    """Reject, and for the stated reason."""
    result = check(sql, config)
    assert not result.allowed, f"expected a rejection, got {result.rewritten_sql or result.sql!r}"
    assert result.rule == rule, f"rejected by {result.rule!r} ({result.reason}), expected {rule!r}"
    assert result.reason, "a rejection must carry a human-readable reason"
    assert result.rewritten_sql is None, "a rejected query must not offer runnable SQL"
    return result


def _assert_allowed(sql: str, config: GuardrailConfig | None = None) -> GuardrailResult:
    result = check(sql, config)
    assert result.allowed, f"rejected by {result.rule}: {result.reason}"
    return result


# ------------------------------------------------------------ parse / one only


def test_empty_sql_is_rejected() -> None:
    _assert_rejected("", "parse")


def test_whitespace_only_sql_is_rejected() -> None:
    _assert_rejected("   \n\t ", "parse")


def test_two_statements_are_rejected() -> None:
    _assert_rejected("SELECT 1; SELECT 2", "single_statement")


def test_a_statement_smuggled_after_a_semicolon_is_rejected() -> None:
    """The gap `generate.py:_must_be_a_single_select` documents and defers here."""
    result = _assert_rejected("SELECT id FROM users; DROP TABLE users", "single_statement")
    assert "DROP" in result.reason


def test_an_unclosed_string_literal_is_rejected() -> None:
    _assert_rejected("SELECT 'abc", "parse")


def test_unbalanced_parentheses_are_rejected() -> None:
    """sqlparse accepts this silently, so the check has to be explicit."""
    _assert_rejected("SELECT * FROM (SELECT 1", "parse")


def test_a_single_trailing_semicolon_is_accepted() -> None:
    """Every few-shot example ends in one; rejecting it would reject all of them."""
    _assert_allowed("SELECT id FROM users;")


def test_a_redundant_second_semicolon_is_not_a_second_statement() -> None:
    """`SELECT 1;;` parses as two statements, the second one empty."""
    _assert_allowed("SELECT id FROM users;;")


# ---------------------------------------------------------------- statement type


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO orders (id) VALUES (1)",
        "UPDATE orders SET status = 'paid'",
        "DELETE FROM orders",
        "MERGE INTO orders AS o USING staging AS s ON o.id = s.id",
        "CREATE TABLE t (a int)",
        "ALTER TABLE orders ADD COLUMN c int",
        "DROP TABLE orders",
        "TRUNCATE orders",
        "GRANT SELECT ON orders TO queryguard_ro",
        "REVOKE SELECT ON orders FROM queryguard_ro",
        "SET work_mem = '1GB'",
        "COPY orders TO '/tmp/orders.csv'",
        "CALL do_something()",
        "DO $$ BEGIN PERFORM 1; END $$",
        "VACUUM orders",
        "this is not sql at all",
    ],
)
def test_non_select_statements_are_rejected(sql) -> None:
    _assert_rejected(sql, "statement_type")


@pytest.mark.parametrize("sql", ["EXPLAIN SELECT 1", "EXPLAIN ANALYZE SELECT count(*) FROM orders"])
def test_explain_is_rejected_with_and_without_analyze(sql) -> None:
    """ANALYZE actually runs the query; plain EXPLAIN is refused for consistency."""
    _assert_rejected(sql, "statement_type")


def test_a_plain_select_and_a_cte_are_both_accepted() -> None:
    _assert_allowed("SELECT id FROM orders")
    _assert_allowed("WITH t AS (SELECT 1 AS n) SELECT n FROM t")


# ----------------------------------------------------------- forbidden constructs


@pytest.mark.parametrize(
    "sql",
    [
        "WITH x AS (INSERT INTO orders (id) VALUES (1) RETURNING *) SELECT * FROM x",
        "WITH x AS (UPDATE orders SET status = 'paid' RETURNING *) SELECT * FROM x",
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
    ],
)
def test_a_data_modifying_cte_is_rejected(sql) -> None:
    """sqlparse types all three of these as SELECT, which is why they get their own test.

    A statement-type check on its own passes them straight through to a write.
    """
    assert sqlparse.parse(sql)[0].get_type() == "SELECT", "premise of this test changed"
    _assert_rejected(sql, "forbidden_construct")


def test_select_into_is_rejected() -> None:
    _assert_rejected("SELECT id INTO archived_orders FROM orders", "forbidden_construct")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM orders FOR UPDATE",
        "SELECT id FROM orders FOR SHARE",
        "SELECT id FROM orders FOR NO KEY UPDATE",
        "SELECT id FROM orders FOR KEY SHARE",
    ],
)
def test_row_locking_clauses_are_rejected(sql) -> None:
    _assert_rejected(sql, "forbidden_construct")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_sleep(10)",
        "SELECT * FROM dblink('dbname=other', 'SELECT 1') AS t(a int)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_ls_dir('/')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT lo_export(loid, '/tmp/out') FROM big",
    ],
)
def test_functions_that_escape_the_database_are_rejected(sql) -> None:
    """None of these need a write privilege to read files, stall, or reach a host."""
    _assert_rejected(sql, "forbidden_construct")


def test_a_quoted_forbidden_function_is_still_matched() -> None:
    """Quoting an identifier must not be a way around the name check.

    sqlparse types `"lo_import"` as String.Symbol rather than Name, so a check
    that reads only Name tokens lets this exact query -- which runs -- straight
    through.
    """
    _assert_rejected('SELECT "lo_import"(\'/etc/passwd\')', "forbidden_construct")
    _assert_rejected('SELECT "LO_IMPORT"(\'/etc/passwd\')', "forbidden_construct")


# ----------------------------------------------------------------- false positives


def test_a_column_named_updated_at_does_not_look_like_an_update() -> None:
    """The reason the rules read tokens: substring matching rejects this."""
    sql = "SELECT o.updated_at FROM orders AS o WHERE o.updated_at > now()"
    result = _assert_allowed(sql)
    assert result.rewritten_sql == f"{sql}\nLIMIT {DEFAULT_MAX_ROWS + 1}", "only the cap should change"


def test_a_string_literal_containing_drop_is_not_a_drop() -> None:
    sql = "SELECT o.id FROM orders AS o WHERE o.note = 'DROP TABLE users'"
    result = _assert_allowed(sql)
    assert result.rewritten_sql == f"{sql}\nLIMIT {DEFAULT_MAX_ROWS + 1}", "only the cap should change"


def test_ordinary_column_and_alias_names_survive_the_keyword_rules() -> None:
    sql = (
        "SELECT c.country AS grant_region, count(*) AS delete_count "
        "FROM customers AS c GROUP BY c.country"
    )
    _assert_allowed(sql)


# --------------------------------------------------------------- subquery depth


def test_subqueries_at_the_depth_limit_are_accepted() -> None:
    sql = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT 1 AS n) AS c) AS b) AS a"
    _assert_allowed(sql)


def test_subqueries_past_the_depth_limit_are_rejected() -> None:
    sql = (
        "SELECT * FROM (SELECT * FROM (SELECT * FROM "
        "(SELECT * FROM (SELECT 1 AS n) AS d) AS c) AS b) AS a"
    )
    result = _assert_rejected(sql, "subquery_depth")
    assert str(DEFAULT_MAX_SUBQUERY_DEPTH) in result.reason


def test_nested_function_calls_and_in_lists_are_not_subqueries() -> None:
    """Few-shot example 5 nests `round(avg(...), 2)`.

    A parenthesis counter scores this 2 and would spend most of the budget on
    arithmetic, eventually rejecting perfectly ordinary aggregate SQL.
    """
    sql = (
        "SELECT round(avg(coalesce(o.total_amount, 0)), 2) AS avg_value "
        "FROM orders AS o WHERE o.status IN ('paid', 'shipped', 'refunded')"
    )
    _assert_allowed(sql, GuardrailConfig(max_subquery_depth=0))


def test_a_cte_counts_as_one_level_of_nesting() -> None:
    sql = "WITH c AS (SELECT 1 AS n) SELECT n FROM c"
    _assert_allowed(sql, GuardrailConfig(max_subquery_depth=1))
    _assert_rejected(sql, "subquery_depth", GuardrailConfig(max_subquery_depth=0))


# -------------------------------------------------------------------- row limit


def test_a_limit_above_the_maximum_is_rejected() -> None:
    result = _assert_rejected("SELECT id FROM orders LIMIT 5000", "row_limit")
    assert "5000" in result.reason and str(DEFAULT_MAX_ROWS) in result.reason


def test_a_limit_at_the_maximum_is_accepted_untouched() -> None:
    result = _assert_allowed(f"SELECT id FROM orders LIMIT {DEFAULT_MAX_ROWS}")
    assert result.rewritten_sql is None


def test_the_overflow_row_the_rewrite_adds_is_within_the_ceiling() -> None:
    """max_rows + 1 is what _append_limit writes, so it must pass a re-check."""
    _assert_allowed(f"SELECT id FROM orders LIMIT {DEFAULT_MAX_ROWS + 1}")
    _assert_rejected(f"SELECT id FROM orders LIMIT {DEFAULT_MAX_ROWS + 2}", "row_limit")


def test_an_oversized_limit_inside_a_cte_is_rejected() -> None:
    """The outer query is capped, but the CTE still materialises 50k rows."""
    sql = "WITH c AS (SELECT * FROM orders LIMIT 50000) SELECT * FROM c LIMIT 10"
    _assert_rejected(sql, "row_limit")


def test_limit_all_is_rejected() -> None:
    _assert_rejected("SELECT id FROM orders LIMIT ALL", "row_limit")


def test_a_non_literal_limit_cannot_be_verified_and_is_rejected() -> None:
    _assert_rejected("SELECT id FROM orders LIMIT $1", "row_limit")


def test_a_missing_limit_is_rewritten_not_rejected() -> None:
    result = _assert_allowed("SELECT id FROM orders")
    assert result.rewritten_sql == f"SELECT id FROM orders\nLIMIT {DEFAULT_MAX_ROWS + 1}"


def test_the_rewrite_survives_a_trailing_order_by() -> None:
    sql = "SELECT o.id FROM orders AS o ORDER BY o.order_date DESC;"
    result = _assert_allowed(sql)
    _assert_valid_capped_select(result.rewritten_sql)
    assert result.rewritten_sql.endswith(f"DESC\nLIMIT {DEFAULT_MAX_ROWS + 1}")


def test_the_rewrite_caps_the_outer_query_of_a_cte_not_the_cte_body() -> None:
    """The CTE's own LIMIT 10 does not cap what the outer query returns."""
    sql = "WITH c AS (SELECT * FROM orders LIMIT 10) SELECT * FROM c;"
    result = _assert_allowed(sql)
    _assert_valid_capped_select(result.rewritten_sql)
    assert result.rewritten_sql.endswith(f"SELECT * FROM c\nLIMIT {DEFAULT_MAX_ROWS + 1}")


def test_the_rewrite_strips_the_trailing_semicolon() -> None:
    """Appending after the semicolon would produce a second, nonsense statement."""
    result = _assert_allowed("SELECT id FROM orders;")
    assert ";" not in result.rewritten_sql
    _assert_valid_capped_select(result.rewritten_sql)


def test_a_rewritten_query_passes_the_guardrails_unchanged() -> None:
    """Idempotence: the executor's input must itself be clean."""
    first = _assert_allowed("SELECT o.id FROM orders AS o ORDER BY o.id")
    second = _assert_allowed(first.rewritten_sql)
    assert second.rewritten_sql is None


def test_fetch_first_is_recognised_as_a_cap() -> None:
    """PostgreSQL rejects LIMIT alongside FETCH, so no second cap may be added."""
    result = _assert_allowed("SELECT id FROM orders FETCH FIRST 10 ROWS ONLY")
    assert result.rewritten_sql is None


def test_an_oversized_fetch_first_is_rejected() -> None:
    _assert_rejected("SELECT id FROM orders FETCH FIRST 50000 ROWS ONLY", "row_limit")


def test_auto_limit_off_rejects_an_uncapped_query_instead_of_rewriting_it() -> None:
    _assert_rejected("SELECT id FROM orders", "row_limit", GuardrailConfig(auto_limit=False))


def test_max_rows_is_configurable() -> None:
    tight = GuardrailConfig(max_rows=10)
    _assert_rejected("SELECT id FROM orders LIMIT 50", "row_limit", tight)
    assert _assert_allowed("SELECT id FROM orders", tight).rewritten_sql.endswith("LIMIT 11")


# --------------------------------------------------------------------- comments


def test_a_line_comment_is_rejected() -> None:
    _assert_rejected("SELECT id FROM orders -- everything is fine", "comment")


def test_a_block_comment_is_rejected() -> None:
    _assert_rejected("SELECT /* nothing to see */ id FROM orders", "comment")


def test_a_comment_after_a_semicolon_is_rejected() -> None:
    """sqlparse reads this as one statement, so only the comment rule catches it."""
    _assert_rejected("SELECT id FROM orders; -- DROP TABLE orders", "comment")


def test_comments_can_be_allowed_and_the_cap_still_lands_outside_them() -> None:
    result = _assert_allowed(
        "SELECT id FROM orders -- a note", GuardrailConfig(allow_comments=True)
    )
    assert result.rewritten_sql == f"SELECT id FROM orders\nLIMIT {DEFAULT_MAX_ROWS + 1}"


# ----------------------------------------------------------------------- result


def test_sql_to_execute_returns_the_rewritten_query_when_one_was_made() -> None:
    result = _assert_allowed("SELECT id FROM orders")
    assert result.sql_to_execute == result.rewritten_sql


def test_sql_to_execute_returns_the_original_when_nothing_changed() -> None:
    sql = "SELECT id FROM orders LIMIT 10"
    assert _assert_allowed(sql).sql_to_execute == sql


def test_sql_to_execute_raises_on_a_rejected_query() -> None:
    """The whole point: a caller that forgets to check `allowed` cannot run it."""
    result = check("DROP TABLE orders")
    with pytest.raises(RuntimeError, match="statement_type"):
        result.sql_to_execute


# ------------------------------------------------------------- few-shot examples


def _example_queries() -> list[tuple[str, str]]:
    """Every runnable SQL string in examples.py, including both ambiguous readings."""
    queries: list[tuple[str, str]] = []
    for number, example in enumerate(EXAMPLES, 1):
        if example.sql:
            queries.append((f"example{number}", example.sql))
        for label, sql in example.interpretations:
            queries.append((f"example{number}-{label}", sql))
    return queries


EXAMPLE_QUERIES = _example_queries()


def _assert_valid_capped_select(sql: str) -> None:
    """The rewritten text is one SELECT whose outermost clause is the new cap."""
    statements = [s for s in sqlparse.parse(sql) if str(s).strip().strip(";").strip()]
    assert len(statements) == 1, f"rewrite produced {len(statements)} statements"
    assert statements[0].get_type() == "SELECT"
    assert check(sql).allowed, "the rewritten SQL must itself pass"


def test_the_example_corpus_is_the_size_this_file_assumes() -> None:
    """Fails loudly if examples.py grows, rather than silently testing less."""
    assert len(EXAMPLES) == 7
    assert len(EXAMPLE_QUERIES) == 8, "6 single queries plus example 7's two readings"


@pytest.mark.parametrize("name,sql", EXAMPLE_QUERIES, ids=[n for n, _ in EXAMPLE_QUERIES])
def test_every_few_shot_example_passes_the_guardrails(name, sql) -> None:
    """The examples are the prompt's definition of good SQL.

    A guardrail that rejects one of them is telling the model to generate SQL the
    executor will refuse, and the whole pipeline argues with itself.
    """
    result = check(sql)
    assert result.allowed, f"{name} rejected by {result.rule}: {result.reason}"

    if result.rewritten_sql is None:
        assert "LIMIT" in sql.upper(), f"{name} was left alone but has no LIMIT"
    else:
        assert result.rewritten_sql == f"{sql.rstrip().rstrip(';')}\nLIMIT {DEFAULT_MAX_ROWS + 1}"
        _assert_valid_capped_select(result.rewritten_sql)


def test_only_the_two_examples_that_already_limit_are_left_untouched() -> None:
    """Examples 2 and 6 ask for 5 rows; the other six are uncapped."""
    untouched = {
        name for name, sql in EXAMPLE_QUERIES if check(sql).rewritten_sql is None
    }
    assert untouched == {"example2", "example6"}
