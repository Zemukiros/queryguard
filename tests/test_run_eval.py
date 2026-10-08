"""Tests for the eval harness. No API: every client is evals.run_eval.FakeSDK.

The end-to-end tests run a handful of items against the real database (the
pipeline executes SQL), with results, ledger and logs in tmp_path.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from evals import run_eval
from evals.run_eval import (
    GENERATED,
    GOLDEN,
    MUTATION,
    STEPS,
    Budget,
    BudgetExhausted,
    Context,
    FakeSDK,
    Item,
    TransientFailure,
    _transient_errors,
    build_items,
    run_item,
    select_items,
    finished_ids,
    golden_frames,
    label_generated,
    run,
    token_profile,
)
from queryguard.executor import OUTCOME_OK, ExecutionResult
from queryguard.generate import Ambiguity, ClarificationNeeded, GeneratedSQL
from queryguard.guardrails import GuardrailResult
from queryguard.pipeline import PipelineResult
from queryguard.schema.introspect import load_schema


def _profile(max_cost: float = 0.01) -> dict:
    usage = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    return {s: {"samples": 1, "mean_usage": usage, "mean_cost": max_cost / 2, "max_cost": max_cost} for s in STEPS}


# ------------------------------------------------------------------ the items


def test_all_three_populations_are_built_and_interleaved() -> None:
    items = build_items()
    counts = {p: sum(i.population == p for i in items) for p in (GENERATED, MUTATION, GOLDEN)}
    assert counts == {GENERATED: 50, MUTATION: 104, GOLDEN: 40}
    assert len({i.id for i in items}) == len(items) == 194
    assert [i.population for i in items[:3]] == [GENERATED, MUTATION, GOLDEN], "a run stopped early stays balanced"
    assert all(i.sql for i in items if i.population != GENERATED)
    assert all(i.mutation for i in items if i.population == MUTATION)


def test_a_targeted_rerun_selects_by_category_and_population() -> None:
    items = select_items(build_items(), ["refund_trap"], [GENERATED, MUTATION])
    assert {i.category for i in items} == {"refund_trap"}
    assert {i.population for i in items} == {GENERATED, MUTATION}
    assert sum(i.population == GENERATED for i in items) == 6
    assert "mut:refund_04__drop_where" in {i.id for i in items}
    assert select_items(build_items()) == build_items()


# --------------------------------------------------------------- token profile


def test_token_profile_tells_steps_apart_by_model_and_size(tmp_path) -> None:
    rows = [
        {"model": "claude-sonnet-5", "input_tokens": 18, "output_tokens": 600, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 6473, "estimated_cost_usd": 0.01},
        {"model": "claude-sonnet-5", "input_tokens": 260, "output_tokens": 300, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 6473, "estimated_cost_usd": 0.005},
        {"model": "claude-haiku-4-5", "input_tokens": 2950, "output_tokens": 80, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "estimated_cost_usd": 0.0033},
        {"model": "claude-haiku-4-5", "input_tokens": 700, "output_tokens": 40, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "estimated_cost_usd": 0.0009},
    ]
    path = tmp_path / "calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    profile = token_profile(path)
    assert profile["generate"]["mean_usage"]["output_tokens"] == 600
    assert profile["second_sql"]["mean_usage"]["input_tokens"] == 260
    assert profile["back_translate"]["max_cost"] == 0.0033
    assert profile["judge"]["samples"] == 1


def test_a_missing_step_refuses_to_project(tmp_path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text(json.dumps({"model": "claude-sonnet-5", "input_tokens": 18, "output_tokens": 1, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "estimated_cost_usd": 0.0}))
    with pytest.raises(SystemExit, match="cannot project"):
        token_profile(path)


# --------------------------------------------------------------------- budget


def _ledger(path, calls: int, cost_each: float) -> None:
    path.write_text("".join(json.dumps({"estimated_cost_usd": cost_each}) + "\n" for _ in range(calls)))


def test_a_reservation_counts_against_both_caps(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    budget = Budget(ledger, max_calls=10, max_cost=1.0, profile=_profile(0.01))
    first = budget.reserve(GENERATED)  # 4 calls, 4 x 0.01
    second = budget.reserve(GENERATED)
    assert first == (4, pytest.approx(0.04)) and second is not None
    assert budget.reserve(MUTATION) is None, "4 + 4 + 3 > 10 calls"
    budget.release(second)
    assert budget.reserve(MUTATION) is not None


def test_spent_calls_and_cost_come_from_the_ledger(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _ledger(ledger, 5, 0.19)
    budget = Budget(ledger, max_calls=600, max_cost=1.0, profile=_profile(0.01))
    assert budget.spent() == (5, pytest.approx(0.95))
    assert budget.reserve(GENERATED) is not None, "$0.95 spent + $0.04 reserved fits under $1.00"
    assert budget.reserve(GENERATED) is None, "a second $0.04 reservation does not"


def test_the_cost_cap_refuses_an_item_that_could_cross_it(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _ledger(ledger, 3, 0.33)  # $0.99 spent
    budget = Budget(ledger, max_calls=600, max_cost=1.0, profile=_profile(0.01))
    assert budget.reserve(GOLDEN) is None  # worst case 3 x 0.01 = 0.03 > 0.01 left


def test_check_call_is_the_last_line_of_defence(tmp_path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    _ledger(ledger, 2, 0.0)
    with pytest.raises(BudgetExhausted):
        Budget(ledger, max_calls=2, max_cost=1.0, profile=_profile()).check_call()


# ------------------------------------------------------------------ labelling


def _item(category: str, expected: str | None = None) -> Item:
    return Item("gen:x", GENERATED, category, "q?", "lookup_02", expected_outcome=expected)


def _executed(sql: str, frame: pd.DataFrame, confidence: float = 0.9) -> PipelineResult:
    answer = GeneratedSQL(sql=sql, explanation="", confidence=confidence, tables_used=[], columns_used=[],
                          assumptions=[], ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]))
    return PipelineResult(
        question="q?", answer=answer, guardrail=GuardrailResult(allowed=True, sql=sql),
        execution=ExecutionResult(outcome=OUTCOME_OK, sql_sha256="0" * 64, rows=frame, row_count=len(frame)),
    )


FRAMES = {"lookup_02": pd.DataFrame({"n": [178]})}


def test_a_matching_result_is_correct_and_a_different_one_wrong() -> None:
    item = _item("simple_lookup")
    assert label_generated(item, _executed("SELECT 178", pd.DataFrame({"count": [178]})), FRAMES)[0] == "correct"
    label, why = label_generated(item, _executed("SELECT 0", pd.DataFrame({"count": [0]})), FRAMES)
    assert label == "wrong" and why.startswith("disagree")


def test_a_clarification_is_correct_only_where_one_was_expected() -> None:
    clarification = ClarificationNeeded(question="q?", interpretations=[])
    assert label_generated(_item("ambiguous", "clarification"), clarification, FRAMES)[0] == "correct"
    assert label_generated(_item("simple_lookup"), clarification, FRAMES)[0] == "wrong"


def test_an_ambiguous_question_answered_silently_is_wrong() -> None:
    outcome = _executed("SELECT 1", pd.DataFrame({"x": [1]}))
    assert label_generated(_item("ambiguous", "clarification"), outcome, FRAMES)[0] == "wrong"


def test_unanswerable_is_correct_on_low_self_confidence_or_clarification() -> None:
    item = _item("unanswerable", "refusal_or_clarification")
    low = _executed("SELECT 1", pd.DataFrame({"x": [1]}), confidence=0.2)
    high = _executed("SELECT 1", pd.DataFrame({"x": [1]}), confidence=0.8)
    assert label_generated(item, low, FRAMES)[0] == "correct"
    assert label_generated(item, high, FRAMES)[0] == "wrong"
    assert label_generated(item, ClarificationNeeded(question="q?", interpretations=[]), FRAMES)[0] == "correct"


def test_rate_limits_swallowed_by_validation_are_transient() -> None:
    class Outcome:
        validation_errors = ("alignment: RateLimitError: 429", "agreement: ValueError: bad output")

    assert _transient_errors(Outcome()) == ["alignment: RateLimitError: 429"]


# ---------------------------------------------------------------- end to end


def _context(tmp_path, *, max_calls=600, max_cost=4.0) -> Context:
    profile = _profile()
    budget = Budget(tmp_path / "llm_calls.jsonl", max_calls, max_cost, profile)  # conftest's LLM log
    return Context("dryrun-test", True, FakeSDK(profile), budget, load_schema(), golden_frames(),
                   tmp_path / "results.jsonl")


def test_a_dry_run_records_every_field_and_resumes_without_rerunning(tmp_path, live_database) -> None:
    items = build_items()[:6]
    ctx = _context(tmp_path)
    assert run(items, ctx, concurrency=4) == "all items finished"

    rows = [json.loads(line) for line in ctx.results_path.read_text().splitlines()]
    assert {r["id"] for r in rows} == {i.id for i in items}
    for r in rows:
        for key in ("id", "population", "category", "mutation", "label", "features", "confidence",
                    "detectors", "calls", "n_calls", "cost_usd", "latency_ms"):
            assert key in r, key
        assert r["n_calls"] == len(r["calls"]) and r["cost_usd"] > 0
    assert {r["label"] for r in rows if r["population"] == MUTATION} == {"wrong"}
    assert {r["label"] for r in rows if r["population"] == GOLDEN} == {"correct"}

    calls_before = ctx.budget.spent()[0]
    assert run(items, ctx, concurrency=4) == "all items finished"
    assert ctx.budget.spent()[0] == calls_before, "finished items are never re-run"
    assert len(ctx.results_path.read_text().splitlines()) == 6


def test_the_run_stops_cleanly_before_crossing_the_call_cap(tmp_path, live_database) -> None:
    ctx = _context(tmp_path, max_calls=10)
    reason = run(build_items()[:9], ctx, concurrency=4)
    assert reason.startswith("budget:")
    assert ctx.budget.spent()[0] <= 10
    assert 0 < len(finished_ids(ctx.results_path)) < 9


def test_a_transient_failure_is_requeued_not_recorded(tmp_path, live_database, monkeypatch) -> None:
    monkeypatch.setattr(run_eval, "BACKOFF_START_S", 0.01)
    real = run_eval.run_item
    failures = {"left": 1}

    def flaky(item, ctx):
        if failures["left"]:
            failures["left"] -= 1
            raise TransientFailure("RateLimitError: 429")
        return real(item, ctx)

    monkeypatch.setattr(run_eval, "run_item", flaky)
    ctx = _context(tmp_path)
    assert run(build_items()[:1], ctx, concurrency=1) == "all items finished"
    assert len(finished_ids(ctx.results_path)) == 1


# ------------------------------------------------------------------------ cli


def test_live_needs_an_explicit_run_id() -> None:
    with pytest.raises(SystemExit):
        run_eval.main(["--live"])


def test_a_dry_run_cannot_take_a_real_looking_run_id() -> None:
    with pytest.raises(SystemExit):
        run_eval.main(["--run-id", "live-2026-10-01"])


# ------------------------------------------------- first-live-run fixes


def test_a_refusal_is_correct_for_an_unanswerable_question_only() -> None:
    from queryguard.generate import CannotAnswer

    refusal = CannotAnswer(question="q?", explanation="no warehouse data")
    assert label_generated(_item("unanswerable", "refusal_or_clarification"), refusal, FRAMES)[0] == "correct"
    assert label_generated(_item("simple_lookup"), refusal, FRAMES)[0] == "wrong"
    assert label_generated(_item("ambiguous", "clarification"), refusal, FRAMES)[0] == "wrong"


def test_a_generated_answer_may_add_columns_but_not_drop_them() -> None:
    item = _item("simple_lookup")
    with_name = _executed("SELECT", pd.DataFrame({"name": ["x"], "count": [178]}))
    assert label_generated(item, with_name, FRAMES)[0] == "correct"


def test_compare_columns_ignores_the_golden_id_column() -> None:
    from evals.run_eval import compare_to_golden

    item = Item("gen:join_06", GENERATED, "join", "q?", "join_06", compare_columns=("product", "category"))
    golden = pd.DataFrame({"order_item_id": [7500], "product": ["Compact Skillet"], "category": ["Cookware"]})
    generated = pd.DataFrame({"order_id": [2500], "product_id": [105], "product_name": ["Compact Skillet"], "category_name": ["Cookware"]})
    assert compare_to_golden(item, golden, generated)[0] == "agree"
    strict = Item("gen:join_06", GENERATED, "join", "q?", "join_06")
    assert compare_to_golden(strict, golden, generated)[0] != "agree"


def test_null_label_ok_accepts_one_consistent_label_for_the_null_group() -> None:
    from evals.run_eval import compare_to_golden

    item = Item("gen:agg_05", GENERATED, "aggregation", "q?", "agg_05", null_label_ok=True)
    golden = pd.DataFrame({"approved_by": ["j.kim", None], "refunds": [40, 101]})
    generated = pd.DataFrame({"approver": ["auto-approved", "j.kim"], "n": [101, 40]})
    assert compare_to_golden(item, golden, generated)[0] == "agree"
    wrong_count = generated.assign(n=[100, 40])
    assert compare_to_golden(item, golden, wrong_count)[0] != "agree"


def test_unfinished_items_mean_a_non_zero_exit_and_an_honest_reason() -> None:
    from evals.run_eval import final_status

    assert final_status("all items finished", []) == ("all items finished", 0)
    reason, code = final_status("all items finished", ["gen:unans_02"])
    assert code == 3 and "unfinished" in reason
    assert final_status("budget: ...", ["mut:x"]) == ("budget: ...", 3)


def test_recompute_fixes_agreement_features_and_confidence_offline(live_database) -> None:
    """A row scored under the old width rule is re-evaluated from its logged SQL."""
    from evals.recompute import recompute_row
    from queryguard.validation.confidence import Features, score

    features = dict(
        executed=True, self_confidence=0.9, alignment=1.0, discrepancy_count=0, sanity_fail=0,
        sanity_warn=0, sanity_info=0, agreement="incomparable", guardrail_rewrote=True, row_count_bucket="2-10",
    )
    before, _ = score(Features(**features))
    row = {
        "id": "gold:agg_02", "population": GOLDEN, "question": "How many customers are there in each country?",
        "sql": "SELECT c.country, count(*) AS customers FROM customers AS c GROUP BY c.country",
        "outcome": "executed", "label": "correct", "label_reason": "known golden",
        "features": features, "confidence": before, "confidence_breakdown": {},
        "detectors": {"agreement": {
            "outcome": "incomparable", "explanation": "the queries return different columns (2 vs 3)", "flagged": True,
            "second_sql": "SELECT c.country, count(*) AS n, count(DISTINCT c.city) AS cities FROM customers AS c GROUP BY c.country",
        }},
    }
    item = Item("gold:agg_02", GOLDEN, "aggregation", row["question"], "agg_02")
    fixed = recompute_row(row, item, {})

    assert fixed["detectors"]["agreement"]["outcome"] == "agree"
    assert fixed["detectors"]["agreement"]["flagged"] is False
    assert fixed["features"]["agreement"] == "agree"
    assert fixed["confidence"] > before
    assert fixed["corrections"]["agreement"]["outcome"] == "incomparable"
    assert row["detectors"]["agreement"]["outcome"] == "incomparable", "the input row is not mutated"


def test_recompute_applies_a_new_sanity_rule_to_old_rows(live_database) -> None:
    """refund_04's drop_where mutation passed every sanity check in live-2026-10-01."""
    from evals.recompute import recompute_row
    from queryguard.schema.introspect import load_schema
    from queryguard.validation.confidence import Features, score

    features = dict(
        executed=True, self_confidence=0.9, alignment=1.0, discrepancy_count=0, sanity_fail=0,
        sanity_warn=0, sanity_info=0, agreement="agree", guardrail_rewrote=True, row_count_bucket="1",
    )
    before, _ = score(Features(**features))
    row = {
        "id": "mut:refund_04__drop_where", "population": MUTATION,
        "question": "What was gross revenue from orders placed in 2025, before refunds?",
        "sql": "SELECT sum(o.total_amount) AS gross_revenue FROM orders AS o "
               "WHERE o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'",
        "outcome": "executed", "label": "wrong", "label_reason": "known mutation",
        "features": features, "confidence": before, "confidence_breakdown": {},
        "detectors": {"sanity": {"fail": 0, "warn": 0, "info": 0, "checks": [], "flagged": False}},
    }
    item = Item(row["id"], MUTATION, "refund_trap", row["question"], "refund_04")
    fixed = recompute_row(row, item, {}, load_schema())

    assert fixed["detectors"]["sanity"]["checks"] == ["warn:revenue_status"]
    assert fixed["detectors"]["sanity"]["flagged"] is True
    assert fixed["features"]["sanity_warn"] == 1
    assert fixed["confidence"] < before
    assert fixed["corrections"]["sanity"]["flagged"] is False


def test_merge_replaces_rerun_rows_and_rejects_unknown_ids() -> None:
    from evals.merge import merge

    base = [{"id": "gen:a", "label": "wrong"}, {"id": "gen:b", "label": "correct"}]
    merged = merge(base, [{"id": "gen:a", "label": "correct"}], "old", "new")
    assert merged == [
        {"id": "gen:a", "label": "correct", "source_run": "new"},
        {"id": "gen:b", "label": "correct", "source_run": "old"},
    ]
    with pytest.raises(SystemExit):
        merge(base, [{"id": "gen:zzz"}], "old", "new")


def test_reusing_second_queries_rejudges_only_and_keeps_the_base_agreement(tmp_path, live_database) -> None:
    from dataclasses import replace

    item = next(i for i in build_items() if i.id == "mut:refund_04__drop_where")
    base = {
        "id": item.id, "run_id": "base-run", "population": MUTATION, "category": item.category,
        "golden_id": item.golden_id, "mutation": item.mutation, "question": item.question, "sql": item.sql,
        "outcome": "executed", "label": "wrong", "label_reason": "known mutation",
        "features": dict(executed=True, self_confidence=0.9, alignment=0.2, discrepancy_count=3, sanity_fail=0,
                         sanity_warn=1, sanity_info=0, agreement="disagree", guardrail_rewrote=True,
                         row_count_bucket="1"),
        "detectors": {"agreement": {"outcome": "disagree", "explanation": "x", "flagged": True,
                                    "second_sql": "SELECT 1"},
                      "sanity": {"fail": 0, "warn": 1, "info": 0, "checks": ["warn:revenue_status"], "flagged": True},
                      "alignment": {"score": 0.2, "discrepancies": ["a", "b", "c"], "back_translation": "old",
                                    "flagged": True}},
        "corrections": {"sanity": {}}, "source_run": "base-run",
    }
    ctx = _context(tmp_path)
    ctx.budget.reuse_second_sql = True
    ctx = replace(ctx, reuse={item.id: base}, prompt_version="p-test")
    assert ctx.budget.worst_case(MUTATION)[0] == 2

    row = run_item(item, ctx)

    assert [c["step"] for c in row["calls"]] == ["back_translate", "judge"]
    assert row["calls"][0]["output"] == {"question": "(dry run)", "details": []}, "the judge's input is kept"
    assert row["detectors"]["agreement"] == base["detectors"]["agreement"]
    assert row["features"]["agreement"] == "disagree"
    assert row["features"]["alignment"] == 1.0 and row["features"]["discrepancy_count"] == 0
    assert row["detectors"]["alignment"]["back_translation"] == "(dry run)"
    assert row["second_sql_from"] == "base-run" and row["prompt_version"] == "p-test"
    assert "corrections" not in row and row["run_id"] == "dryrun-test"
    assert base["features"]["alignment"] == 0.2, "the base row is not mutated"

