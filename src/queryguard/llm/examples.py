"""Few-shot examples for SQL generation.

Every query here has been executed against the seeded database and its result
inspected. That verification is the whole point of the file: a few-shot example
that returns nothing, or quietly returns the wrong thing, does not fail loudly --
it teaches the model to make exactly that mistake on every subsequent call, and
the mistake then looks confident and well-formed.

Kept deliberately small. These go into every request, so each one has to earn
its tokens by demonstrating a pattern the others do not.
"""

from __future__ import annotations

from typing import NamedTuple


class Example(NamedTuple):
    """A question and its answer: either one query, or competing readings.

    `interpretations` is (label, sql) pairs. An ambiguous example cannot be
    expressed as a single `sql` string, and pre-rendering one into that field
    would hide from render_examples() what the example actually demonstrates.
    """

    question: str
    sql: str = ""
    interpretations: tuple[tuple[str, str], ...] = ()


EXAMPLES: list[Example] = [
    # 1. Simple single-table lookup. Establishes the baseline shape.
    Example(
        question="What is the email address of customer 42?",
        sql="SELECT c.customer_id, c.email\n"
        "FROM customers AS c\n"
        "WHERE c.customer_id = 42;",
    ),
    # 2. Two-table join. Explicit JOIN ... ON, aliased, never a comma join.
    Example(
        question="List the 5 most recent orders with the customer's name.",
        sql="SELECT o.order_id,\n"
        "       o.order_date,\n"
        "       c.first_name || ' ' || c.last_name AS customer_name\n"
        "FROM orders AS o\n"
        "JOIN customers AS c ON c.customer_id = o.customer_id\n"
        "ORDER BY o.order_date DESC\n"
        "LIMIT 5;",
    ),
    # 3. Aggregation with GROUP BY, grouping through a join to get a label.
    Example(
        question="How many products are in each category?",
        sql="SELECT cat.name AS category,\n"
        "       count(*) AS product_count\n"
        "FROM products AS p\n"
        "JOIN categories AS cat ON cat.category_id = p.category_id\n"
        "GROUP BY cat.name\n"
        "ORDER BY product_count DESC;",
    ),
    # 4. Date range. Half-open interval on a timestamptz column -- >= start and
    #    < the day after the end, never BETWEEN, which would silently drop rows
    #    with a time-of-day component on the final day.
    Example(
        question="How many orders were placed in the first quarter of 2026?",
        sql="SELECT count(*) AS order_count\n"
        "FROM orders AS o\n"
        "WHERE o.order_date >= DATE '2026-01-01'\n"
        "  AND o.order_date <  DATE '2026-04-01';",
    ),
    # 5. Cancelled orders must be excluded. Deliberately an AVERAGE, not a sum:
    #    cancelled orders carry total_amount = 0, so on a SUM the filter is
    #    invisible and teaches nothing. On an average it shifts every value and
    #    reorders the result, which is the error this example exists to prevent.
    Example(
        question="What is the average order value by country?",
        sql="SELECT c.country,\n"
        "       round(avg(o.total_amount), 2) AS avg_order_value,\n"
        "       count(*) AS order_count\n"
        "FROM orders AS o\n"
        "JOIN customers AS c ON c.customer_id = o.customer_id\n"
        "WHERE o.status <> 'cancelled'\n"
        "GROUP BY c.country\n"
        "ORDER BY avg_order_value DESC;",
    ),
    # 6. A CTE joined back to a table with an explicit JOIN ... ON. Added after
    #    a live run answered "revenue last quarter" with `FROM orders AS o,
    #    bounds AS b` -- the no-comma-join rule held on every shape the examples
    #    demonstrated and broke on the one shape none of them did.
    Example(
        question="Which 5 products sold the most units?",
        sql="WITH product_units AS (\n"
        "    SELECT oi.product_id, sum(oi.quantity) AS units\n"
        "    FROM order_items AS oi GROUP BY oi.product_id\n"
        ")\n"
        "SELECT p.name, u.units\n"
        "FROM product_units AS u JOIN products AS p ON p.product_id = u.product_id\n"
        "ORDER BY u.units DESC LIMIT 5;",
    ),
    # 7. An ambiguous question. "Active" means either the stored flag or recent
    #    purchasing, and the schema supports both: is_active is a real column,
    #    and orders.order_date makes recency computable. Neither reading is
    #    strained, and they disagree loudly -- 470 rows against 255 -- which is
    #    the point. A near-identical pair would teach that flagging ambiguity is
    #    pedantry.
    #
    #    Deliberately NOT "revenue last quarter", even though that is the
    #    canonical ambiguous question here. An example whose question matches
    #    the one being evaluated teaches recall, not detection, and the
    #    evaluation then measures nothing.
    Example(
        question="How many active customers do we have?",
        interpretations=(
            (
                "flagged_active",
                "SELECT count(*) AS active_customers\n"
                "FROM customers AS c\n"
                "WHERE c.is_active;",
            ),
            (
                "purchased_recently",
                "SELECT count(DISTINCT o.customer_id) AS active_customers\n"
                "FROM orders AS o\n"
                "WHERE o.order_date >= now() - INTERVAL '90 days'\n"
                "  AND o.status <> 'cancelled';",
            ),
        ),
    ),
]


def render_examples() -> str:
    """Compact plain-text rendering. No JSON, no markdown fences."""
    parts = ["Worked examples:"]
    for example in EXAMPLES:
        if example.interpretations:
            readings = "\n".join(
                f"[{label}]\n{sql}" for label, sql in example.interpretations
            )
            parts.append(
                f"\nQ: {example.question}\n"
                f"AMBIGUOUS - {len(example.interpretations)} defensible readings, "
                f"no single query:\n{readings}"
            )
        else:
            parts.append(f"\nQ: {example.question}\n{example.sql}")
    return "\n".join(parts)
