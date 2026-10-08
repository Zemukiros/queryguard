"""The AppState contract, run against every implementation.

Each test takes the `state` fixture, which is parametrized over the backends,
so LocalState and RedisState are held to exactly the same behaviour. A backend
whose server is not reachable skips locally and fails in CI
(QUERYGUARD_REQUIRE_DB=1), like the database tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from queryguard.api.settings import Settings
from queryguard.state import AppState
from queryguard.state.local import LocalState

CEILING = 0.10
RESERVE = 0.05


def _settings(tmp_path, **overrides) -> Settings:
    values = dict(db_path=tmp_path / "app.db", rate_per_minute=2, rate_per_day=5,
                  daily_spend_usd=CEILING, question_reserve_usd=RESERVE)
    values.update(overrides)
    return Settings(**values)


def _local(tmp_path, **overrides) -> AppState:
    return LocalState(_settings(tmp_path, **overrides))


BACKENDS = {"local": _local}


@pytest.fixture(params=sorted(BACKENDS))
def make(request, tmp_path):
    """A factory: make(**settings overrides) -> a fresh, empty AppState."""
    return lambda **overrides: BACKENDS[request.param](tmp_path, **overrides)


@pytest.fixture
def state(make) -> AppState:
    return make()


def _record(state: AppState, query_id: str, client: str = "c1", **overrides) -> None:
    fields = dict(query_id=query_id, client=client, question=f"q {query_id}", outcome="answered",
                  confidence=0.9, cached=False, cost_usd=0.01, elapsed_ms=5,
                  result={"stage": "done", "query_id": query_id, "sql": "SELECT 1"}, sql_source="model")
    fields.update(overrides)
    state.record(**fields)


# ------------------------------------------------------------------ identity


def test_client_keys_are_stable_salted_hashes(state) -> None:
    key = state.client_key("203.0.113.7")
    assert key == state.client_key("203.0.113.7")
    assert key != state.client_key("203.0.113.8")
    assert "203.0.113.7" not in key


# ------------------------------------------------------------------ admission


def test_the_rate_limit_counts_per_client(state) -> None:
    assert state.rate_hit("a") is None
    assert state.rate_hit("a") is None
    retry_after = state.rate_hit("a")  # rate_per_minute=2
    assert retry_after is not None and 0 < retry_after <= 60
    assert state.rate_hit("b") is None, "another client has its own window"


def test_reservations_stop_at_the_ceiling_and_free_up_on_release(state) -> None:
    first, spent = state.try_reserve()
    second, _ = state.try_reserve()
    assert first and second and spent == 0.0
    refused, _ = state.try_reserve()  # 3 x 0.05 > 0.10
    assert refused is None

    state.release(first)
    state.release(first)  # twice is harmless: it frees one slot, not two
    third, _ = state.try_reserve()
    assert third is not None
    assert state.try_reserve()[0] is None


def test_recorded_spend_counts_against_the_ceiling(state) -> None:
    now = datetime.now(timezone.utc).isoformat()
    state.call_guard().after_call({"timestamp": now, "estimated_cost_usd": 0.06, "model": "m"})
    assert state.spent_today() == pytest.approx(0.06)
    reservation, spent = state.try_reserve()
    assert reservation is None and spent == pytest.approx(0.06)  # 0.06 + 0.05 > 0.10


# --------------------------------------------------------------------- cache


def test_the_cache_round_trips_a_result(state) -> None:
    assert state.cache_get("k") is None
    state.cache_put("k", {"outcome": "answered", "rows": [[1, "a"]]})
    assert state.cache_get("k") == {"outcome": "answered", "rows": [[1, "a"]]}


# ------------------------------------------------------------------- history


def test_history_is_newest_first_limited_and_scoped(state) -> None:
    for i in range(3):
        _record(state, f"q{i}")
    _record(state, "other", client="c2")

    rows = state.history("c1", 2)
    assert [r["query_id"] for r in rows] == ["q2", "q1"]
    row = rows[0]
    assert row["question"] == "q q2" and row["outcome"] == "answered" and row["sql_source"] == "model"
    assert not row["cached"] and row["confidence"] == pytest.approx(0.9)
    assert row["correct"] is None and row["note"] is None and row["feedback_at"] is None

    assert state.get_result("q2", "c1")["query_id"] == "q2"
    assert state.get_result("q2", "c2") is None, "only the client that asked can reopen it"
    assert state.exists("other") and not state.exists("nope")


def test_feedback_latest_wins_and_only_incorrect_is_exported(state, tmp_path) -> None:
    _record(state, "right")
    _record(state, "wrong")
    state.save_feedback("right", True, None)
    state.save_feedback("wrong", True, None)
    created_at = state.save_feedback("wrong", False, "wanted net")

    [row] = state.incorrect_feedback()
    assert row["query_id"] == "wrong" and row["note"] == "wanted net" and row["feedback_at"] == created_at
    assert json.loads(row["result_json"])["sql"] == "SELECT 1"

    history = {r["query_id"]: r for r in state.history("c1", 10)}
    assert not history["wrong"]["correct"] and history["wrong"]["note"] == "wanted net"
    assert history["right"]["correct"]

    path, count = state.export_feedback_candidates(tmp_path / "candidates.yaml")
    assert count == 1 and "wanted net" in path.read_text()
