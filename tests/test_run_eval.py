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
