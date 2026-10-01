"""Live check of Phase 3 part 2: does the hallucination detector catch anything?

Five questions against the real API, two of them with a deliberately wrong
query injected in place of generation:

  a) How many orders were cancelled?                      expect high confidence
  b) Which 5 customers spent the most in 2025?            expect agree
  c) (b) answered from customers.lifetime_value           must be flagged
  d) (a) with status = 'Cancelled', wrong case            which signals catch it?
  e) What was the average order value by country?

Hard cap of 20 API calls for the whole run (at most 4 + 4 + 3 + 3 + 4 = 18),
and no retries -- the SDK's own retry is switched off too, so every request
made is a request counted. Costs come from the LLM call log, so a question
that ends in a clarification is still costed.

Usage:  uv run python scripts/phase3_live_check.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

os.environ["QUERYGUARD_MAX_REQUESTS"] = "20"

import anthropic  # noqa: E402

from queryguard.config import load_env  # noqa: E402
from queryguard.generate import Ambiguity, ClarificationNeeded, GeneratedSQL  # noqa: E402
from queryguard.llm.client import LLMClient, log_path, request_count  # noqa: E402
from queryguard.pipeline import run_answer, run_question  # noqa: E402
from queryguard.schema.introspect import load_schema  # noqa: E402
from queryguard.validation.backtranslate import VALIDATION_MODEL  # noqa: E402

WRONG_TOP_SPENDERS = (
    "SELECT c.customer_id, c.first_name, c.last_name, c.lifetime_value AS total_spent\n"
    "FROM customers AS c\n"
    "ORDER BY c.lifetime_value DESC\n"
    "LIMIT 5"
)
WRONG_CANCELLED = "SELECT count(*) AS cancelled_orders\nFROM orders AS o\nWHERE o.status = 'Cancelled'"


def _injected(sql: str) -> GeneratedSQL:
    """A confident wrong answer: what a hallucination looks like from outside."""
    return GeneratedSQL(
        sql=sql,
        explanation="Injected for the live check.",
        confidence=0.9,
        tables_used=[],
        columns_used=[],
        assumptions=[],
        ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]),
    )


def _log_lines() -> list[dict]:
    path = log_path()
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _report(label: str, outcome, new_calls: list[dict]) -> dict:
    cost = sum(c["estimated_cost_usd"] for c in new_calls)
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for c in new_calls:
        print(
            f"  call {c['model']:<18} in={c['input_tokens']:>5} cache_write={c['cache_creation_input_tokens']:>5} "
            f"cache_read={c['cache_read_input_tokens']:>5} out={c['output_tokens']:>5} ${c['estimated_cost_usd']:.4f}"
        )
    print(f"  {len(new_calls)} calls, ${cost:.4f}")
    summary = {"label": label, "calls": len(new_calls), "cost_usd": round(cost, 5)}

    if isinstance(outcome, ClarificationNeeded):
        print("  CLARIFICATION NEEDED -- nothing executed, nothing validated")
        for i in outcome.interpretations:
            print(f"   [{i.label}] {i.explanation}")
        summary["clarification"] = [i.label for i in outcome.interpretations]
        return summary

    print(f"\nSQL:\n{outcome.answer.sql}")
    if outcome.execution is not None and outcome.execution.ok:
        print(f"\n{outcome.execution.rows.head(10).to_string(index=False)}")
    for flag in outcome.sanity:
        print(f"  SANITY {flag.severity} [{flag.check}] {flag.explanation}")
    print(f"\n  back-translation: {outcome.back_translation}")
    print(f"  alignment: {outcome.alignment}")
    for d in outcome.discrepancies:
        print(f"    - {d}")
    if outcome.agreement:
        print(f"  agreement: {outcome.agreement.outcome} -- {outcome.agreement.explanation}")
        print(f"  second SQL:\n{outcome.agreement.second_sql}")
    else:
        print("  agreement: not run")
    for e in outcome.validation_errors:
        print(f"  VALIDATION ERROR {e}")
    print(f"  self-confidence: {outcome.answer.confidence}")
    print(f"  features: {outcome.features.to_dict()}")
    print(f"  confidence: {outcome.confidence:.3f}")
    print(f"  breakdown: {outcome.confidence_breakdown}")
    summary.update(
        sql=outcome.answer.sql,
        self_confidence=outcome.answer.confidence,
        back_translation=outcome.back_translation,
        alignment=outcome.alignment,
        discrepancies=list(outcome.discrepancies),
        agreement=outcome.agreement.outcome if outcome.agreement else None,
        agreement_explanation=outcome.agreement.explanation if outcome.agreement else None,
        sanity=[(f.severity, f.check) for f in outcome.sanity],
        confidence=round(outcome.confidence, 3),
        breakdown=outcome.confidence_breakdown,
        errors=list(outcome.validation_errors),
    )
    return summary


def main() -> int:
    load_env()
    sdk = anthropic.Anthropic(max_retries=0)
    clients = {
        "client": LLMClient(sdk_client=sdk),
        "validation_client": LLMClient(sdk_client=sdk, model=VALIDATION_MODEL),
    }
    schema = load_schema()

    runs = [
        ("a) How many orders were cancelled?", lambda: run_question(
            "How many orders were cancelled?", schema=schema, **clients)),
        ("b) Which 5 customers spent the most in 2025?", lambda: run_question(
            "Which 5 customers spent the most in 2025?", schema=schema, **clients)),
        ("c) INJECTED: (b) answered from customers.lifetime_value", lambda: run_answer(
            "Which 5 customers spent the most in 2025?", _injected(WRONG_TOP_SPENDERS), schema=schema, **clients)),
        ("d) INJECTED: (a) with status = 'Cancelled'", lambda: run_answer(
            "How many orders were cancelled?", _injected(WRONG_CANCELLED), schema=schema, **clients)),
        ("e) What was the average order value by country?", lambda: run_question(
            "What was the average order value by country?", schema=schema, **clients)),
    ]

    summaries = []
    start = len(_log_lines())
    for label, run in runs:
        before = len(_log_lines())
        try:
            outcome = run()
        except Exception:  # noqa: BLE001 - report and move on; no retries
            print(f"\n{label}: FAILED\n{traceback.format_exc()}")
            summaries.append({"label": label, "error": traceback.format_exc(limit=1)})
            continue
        summaries.append(_report(label, outcome, _log_lines()[before:]))

    made = _log_lines()[start:]
    total = sum(c["estimated_cost_usd"] for c in made)
    print(f"\nTOTAL: {request_count()} API calls ({len(made)} logged), ${total:.4f}")
    print("\nSUMMARY_JSON " + json.dumps(summaries, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
