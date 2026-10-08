"""A deterministic stand-in for anthropic.Anthropic, for the demo UI at $0.

    QUERYGUARD_FAKE_LLM=1 uv run python -m queryguard.api

Not a model. It answers from the golden set (evals/golden.yaml), so the whole
pipeline -- guardrail, read-only executor, sanity checks, back-translation,
agreement, confidence -- runs for real on real SQL, and only the four LLM
calls are canned:

- generate: a golden question gets its golden SQL; an ambiguous one gets the
  hand-written readings below, every one of them runnable; an unanswerable one,
  or any question not in the golden set, gets a refusal that says so.
- second query: echoes the first, so agreement is "agree".
- back-translation: a golden SQL maps back to its golden question; any other
  SQL (a user's edit) gets a generic description.
- judge: 1.0 when the back-translation is the question asked, else 0.4 with a
  discrepancy -- enough to show the verification UI both ways.

Usage is reported as zero tokens, so every cost is honestly $0. Each call
sleeps QUERYGUARD_FAKE_LLM_DELAY_MS (default 250) so the timeline animates.

Relative dates ("last quarter") are pinned to the seeded data, which ends on
2026-09-02, so the readings return rows on any day; each says which period it
used. This is separate from evals/run_eval.py's FakeSDK, which exists to
project eval costs and is deliberately cruder.
"""

from __future__ import annotations

import os
import time
from functools import cache
from types import SimpleNamespace
from typing import Any

import yaml

from queryguard.config import REPO_ROOT
from queryguard.generate import Ambiguity, GeneratedSQL, Interpretation
from queryguard.text import normalize_question, normalize_sql
from queryguard.validation.backtranslate import AlignmentJudgement, BackTranslation

GOLDEN_YAML = REPO_ROOT / "evals" / "golden.yaml"
DEFAULT_DELAY_MS = 250
NOT_IN_GOLDEN = (
    "Fake LLM mode only answers the golden-set questions (evals/golden.yaml). "
    "Try one of the example questions, or edit the SQL and run it yourself."
)

_PAID = "o.status IN ('paid', 'shipped', 'delivered', 'refunded')"
_Q2_2026 = "o.order_date >= DATE '2026-04-01' AND o.order_date < DATE '2026-07-01'"
_Y2025 = "o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'"

# Readings for the ambiguous golden questions, keyed by golden id.
READINGS: dict[str, list[tuple[str, str, str]]] = {
    "ambig_01": [
        ("gross_revenue", f"""SELECT sum(o.total_amount) AS gross_revenue
FROM orders AS o
WHERE {_PAID}
  AND {_Q2_2026}""",
         "Order totals for Q2 2026 (the last complete quarter in the data), before refunds; unpaid and cancelled orders excluded."),
        ("net_revenue", f"""SELECT sum(o.total_amount) - coalesce((
         SELECT sum(r.amount) FROM refunds AS r
         JOIN orders AS o ON o.order_id = r.order_id
         WHERE {_Q2_2026}), 0) AS net_revenue
FROM orders AS o
WHERE {_PAID}
  AND {_Q2_2026}""",
         "The same Q2 2026 total, minus refunds issued on those orders."),
    ],
    "ambig_02": [
        ("by_total_spend", """SELECT c.customer_id, c.first_name, c.last_name, sum(o.total_amount) AS total_spend
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
WHERE o.status IN ('paid', 'shipped', 'delivered')
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY total_spend DESC
LIMIT 10""",
         "Ranked by what they paid for orders that were not cancelled, refunded or still pending."),
        ("by_order_count", """SELECT c.customer_id, c.first_name, c.last_name, count(*) AS orders
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
WHERE o.status <> 'cancelled'
GROUP BY c.customer_id, c.first_name, c.last_name
ORDER BY orders DESC, c.customer_id
LIMIT 10""",
         "Ranked by how many non-cancelled orders they placed."),
        ("by_lifetime_value", """SELECT c.customer_id, c.first_name, c.last_name, c.lifetime_value
FROM customers AS c
ORDER BY c.lifetime_value DESC NULLS LAST
LIMIT 10""",
         "Ranked by the stored lifetime_value column."),
    ],
    "ambig_03": [
        ("gross_spend_2025", f"""SELECT o.customer_id, sum(o.total_amount) AS spend
FROM orders AS o
WHERE {_PAID}
  AND {_Y2025}
GROUP BY o.customer_id
ORDER BY spend DESC
LIMIT 10""",
         "2025 order totals before refunds, paid statuses only."),
        ("net_spend_2025", f"""WITH spend AS (
  SELECT o.customer_id, sum(o.total_amount) AS spend
  FROM orders AS o
  WHERE {_PAID} AND {_Y2025}
  GROUP BY o.customer_id
), refunded AS (
  SELECT o.customer_id, sum(r.amount) AS refunded
  FROM refunds AS r
  JOIN orders AS o ON o.order_id = r.order_id
  WHERE {_Y2025}
  GROUP BY o.customer_id
)
SELECT s.customer_id, s.spend - coalesce(f.refunded, 0) AS net_spend
FROM spend AS s
LEFT JOIN refunded AS f ON f.customer_id = s.customer_id
ORDER BY net_spend DESC
LIMIT 10""",
         "2025 order totals minus what was refunded on them."),
    ],
    "ambig_04": [
        ("by_signup_date", """SELECT count(*) AS new_customers
FROM customers AS c
WHERE c.signup_date >= DATE '2026-08-01'
  AND c.signup_date < DATE '2026-09-01'""",
         "Customers whose signup_date is in August 2026 (the last complete month in the data)."),
        ("by_first_order", """WITH firsts AS (
  SELECT o.customer_id, min(o.order_date) AS first_order
  FROM orders AS o
  GROUP BY o.customer_id
)
SELECT count(*) AS new_customers
FROM firsts AS f
WHERE f.first_order >= DATE '2026-08-01'
  AND f.first_order < DATE '2026-09-01'""",
         "Customers whose first order was placed in August 2026."),
    ],
    "ambig_05": [
        ("by_units", """SELECT p.product_id, p.name, sum(oi.quantity) AS units
FROM order_items AS oi
JOIN orders AS o ON o.order_id = oi.order_id
JOIN products AS p ON p.product_id = oi.product_id
WHERE o.status <> 'cancelled'
GROUP BY p.product_id, p.name
ORDER BY units DESC
LIMIT 1""",
         "Most units sold on non-cancelled orders."),
        ("by_revenue", """SELECT p.product_id, p.name, sum(oi.quantity * (oi.unit_price - oi.discount)) AS revenue
FROM order_items AS oi
JOIN orders AS o ON o.order_id = oi.order_id
JOIN products AS p ON p.product_id = oi.product_id
WHERE o.status <> 'cancelled'
GROUP BY p.product_id, p.name
ORDER BY revenue DESC
LIMIT 1""",
         "Most line-item revenue after per-unit discounts, on non-cancelled orders."),
    ],
}


@cache
def _golden() -> tuple[dict[str, dict], dict[str, str]]:
    """(entries by normalised question, golden question by normalised golden SQL)."""
    entries = yaml.safe_load(GOLDEN_YAML.read_text(encoding="utf-8"))["questions"]
    by_question = {normalize_question(e["question"]): e for e in entries}
    by_sql = {normalize_sql(e["golden_sql"]): e["question"] for e in entries if "golden_sql" in e}
    return by_question, by_sql


def _answer(sql: str, explanation: str, confidence: float = 0.9, *, interpretations=None) -> GeneratedSQL:
    return GeneratedSQL(
        sql=sql, explanation=explanation, confidence=confidence, tables_used=[], columns_used=[],
        assumptions=[] if sql or interpretations else ["fake LLM: not a golden-set question"],
        ambiguity=Ambiguity(is_ambiguous=bool(interpretations), interpretations=interpretations or []),
    )


def generate(question: str) -> GeneratedSQL:
    entry = _golden()[0].get(normalize_question(question))
    if entry is None:
        return _answer("", NOT_IN_GOLDEN, 0.0)
    if entry["id"] in READINGS:
        return _answer("", "Fake LLM: this golden question is ambiguous.", 0.4, interpretations=[
            Interpretation(label=label, sql=sql, explanation=why) for label, sql, why in READINGS[entry["id"]]
        ])
    if "golden_sql" in entry:
        return _answer(entry["golden_sql"].strip(), f"Fake LLM: the golden SQL for {entry['id']}.")
    return _answer("", f"Fake LLM: {entry['id']} is unanswerable from this schema. {entry.get('notes', '')}".strip(), 0.0)


def back_translate(sql: str) -> BackTranslation:
    question = _golden()[1].get(normalize_sql(sql))
    if question is not None:
        return BackTranslation(question=question, details=["fake LLM: matched a golden SQL"])
    return BackTranslation(
        question="What does this query return? (fake back-translation of SQL outside the golden set)",
        details=["fake LLM: no golden SQL matched"],
    )


def judge(original: str, back_translated: str) -> AlignmentJudgement:
    if normalize_question(original) == normalize_question(back_translated):
        return AlignmentJudgement(alignment=1.0, discrepancies=[])
    return AlignmentJudgement(
        alignment=0.4, discrepancies=["fake judge: this SQL is not the golden answer to the question asked"]
    )


class DemoFakeSDK:
    """Duck-types the one SDK method LLMClient uses: messages.parse()."""

    def __init__(self, delay_ms: int | None = None) -> None:
        self.messages = self
        if delay_ms is None:
            delay_ms = int(os.getenv("QUERYGUARD_FAKE_LLM_DELAY_MS", str(DEFAULT_DELAY_MS)))
        self.delay_s = max(delay_ms, 0) / 1000
        self.calls = 0

    def parse(self, *, output_format: Any, messages: list[dict], **_: Any) -> SimpleNamespace:
        content: str = messages[0]["content"]
        if output_format is BackTranslation:
            parsed: Any = back_translate(content.removeprefix("SQL:\n"))
        elif output_format is AlignmentJudgement:
            original = content.split("\n\nBACK-TRANSLATED: ", 1)[0].removeprefix("ORIGINAL: ")
            back = content.split("\n\nBACK-TRANSLATED: ", 1)[1].split("\n", 1)[0]
            parsed = judge(original, back)
        elif "\nEarlier query:\n" in content:
            first_sql = content.split("\nEarlier query:\n", 1)[1]
            parsed = _answer(first_sql, "Fake LLM: the second query echoes the first.")
        else:
            parsed = generate(content.removeprefix("Q: "))
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        usage = SimpleNamespace(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0,
                                cache_read_input_tokens=0)
        return SimpleNamespace(parsed_output=parsed, usage=usage)
