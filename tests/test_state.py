"""The AppState contract, run against every implementation.

Each test takes the `state` fixture, which is parametrized over the backends,
so LocalState and RedisState are held to exactly the same behaviour. A backend
whose server is not reachable skips locally and fails in CI
(QUERYGUARD_REQUIRE_DB=1), like the database tests.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
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


REDIS_URL = os.getenv("QUERYGUARD_TEST_REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture(scope="session")
def redis_client():
    """A Redis for the RedisState tests (db 15, so dev data in db 0 is never touched)."""
    import redis

    from queryguard.state.redis import connect

    client = connect(REDIS_URL)
    try:
        client.ping()
    except redis.RedisError as exc:
        reason = f"Redis not reachable at {REDIS_URL}, run `docker compose up -d redis` ({exc})"
        if os.getenv("QUERYGUARD_REQUIRE_DB") == "1":
            pytest.fail(f"QUERYGUARD_REQUIRE_DB=1 but {reason}", pytrace=False)
        pytest.skip(reason)
    yield client
    client.close()


@pytest.fixture
def redis_prefix(redis_client):
    """A key prefix of this test's own; its keys are deleted afterwards."""
    prefix = f"test:{uuid.uuid4().hex[:8]}:"
    yield prefix
    keys = list(redis_client.scan_iter(f"{prefix}*"))
    if keys:
        redis_client.delete(*keys)


BACKENDS = ("local", "redis")


@pytest.fixture(params=BACKENDS)
def make(request, tmp_path):
    """A factory: make(**settings overrides) -> a fresh, empty AppState.

    Every RedisState it makes in one test shares a prefix, so two of them
    behave like two instances of one deployment.
    """
    if request.param == "local":
        return lambda **overrides: _local(tmp_path, **overrides)
    from queryguard.state.redis import RedisState

    client = request.getfixturevalue("redis_client")
    prefix = request.getfixturevalue("redis_prefix")
    return lambda **overrides: RedisState(_settings(tmp_path, state_prefix=prefix, **overrides), client=client)


@pytest.fixture
def make_redis(redis_client, redis_prefix, tmp_path):
    """Like `make`, RedisState only: for behaviour that needs more than one process."""
    from queryguard.state.redis import RedisState

    return lambda **overrides: RedisState(
        _settings(tmp_path, state_prefix=redis_prefix, **overrides), client=redis_client)


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


# --------------------------------------------------------------- concurrency


def _in_parallel(n: int, fn) -> list:
    """Run fn n times on n threads released together; return the results."""
    barrier = threading.Barrier(n)
    results: list = [None] * n

    def worker(i: int) -> None:
        barrier.wait()
        try:
            results[i] = fn()
        except Exception as exc:  # noqa: BLE001 - the test inspects it
            results[i] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_parallel_reservations_never_exceed_the_ceiling(make) -> None:
    state = make(daily_spend_usd=1.00, question_reserve_usd=0.05)
    results = _in_parallel(200, state.try_reserve)
    admitted = [r for r, _ in results if r is not None]
    assert len(admitted) == 20  # 20 x 0.05 = 1.00; the 21st would not fit
    assert len({r.id for r in admitted}) == 20


def test_parallel_hits_never_exceed_the_rate_limit(make) -> None:
    state = make(rate_per_minute=5, rate_per_day=100)
    results = _in_parallel(50, lambda: state.rate_hit("same-client"))
    assert sum(r is None for r in results) == 5


def test_two_instances_share_reservations_and_spend(make_redis) -> None:
    a, b = make_redis(), make_redis()
    first, _ = a.try_reserve()
    second, _ = b.try_reserve()
    assert first and second
    assert a.try_reserve()[0] is None and b.try_reserve()[0] is None  # 0.10 used across both
    b.release(first)  # any instance can release any reservation
    assert a.try_reserve()[0] is not None

    now = datetime.now(timezone.utc).isoformat()
    a.call_guard().after_call({"timestamp": now, "estimated_cost_usd": 0.03})
    assert b.spent_today() == pytest.approx(0.03)


def test_a_reservation_from_a_crashed_instance_expires(make_redis) -> None:
    state = make_redis(reservation_ttl_s=0.3)
    assert state.try_reserve()[0] and state.try_reserve()[0]  # never released, as if crashed
    assert state.try_reserve()[0] is None
    time.sleep(0.4)
    assert state.try_reserve()[0] is not None, "expired reservations stop counting"


def test_the_daily_call_cap_holds_across_parallel_calls(make_redis) -> None:
    from queryguard.llm.client import RequestCapExceeded

    state = make_redis(daily_call_cap=10)
    guard = state.call_guard()
    results = _in_parallel(50, guard.before_call)
    assert sum(r is None for r in results) == 10
    assert all(isinstance(r, RequestCapExceeded) for r in results if r is not None)


def test_prefixes_isolate_deployments(redis_client, tmp_path) -> None:
    from queryguard.state.redis import RedisState

    prod = RedisState(_settings(tmp_path, state_prefix=f"t-prod-{uuid.uuid4().hex[:6]}:"), client=redis_client)
    preview = RedisState(_settings(tmp_path, state_prefix=f"t-prev-{uuid.uuid4().hex[:6]}:"), client=redis_client)
    try:
        now = datetime.now(timezone.utc).isoformat()
        prod.call_guard().after_call({"timestamp": now, "estimated_cost_usd": 0.07})
        assert prod.spent_today() == pytest.approx(0.07) and preview.spent_today() == 0.0
    finally:
        for state in (prod, preview):
            keys = list(redis_client.scan_iter(f"{state._prefix}*"))
            if keys:
                redis_client.delete(*keys)



# ----------------------------------------------------------- the API on Redis


def test_the_api_runs_on_redis_state(redis_client, redis_prefix, tmp_path) -> None:
    """A question, its cache hit, history and feedback, all through RedisState."""
    from fastapi.testclient import TestClient

    from queryguard.api.app import create_app
    from queryguard.llm.client import LLMClient
    from queryguard.state.redis import RedisState
    from queryguard.validation.backtranslate import VALIDATION_MODEL
    from tests.test_pipeline import FakeAnthropic, _ambiguous_answer, _synthetic_schema

    settings = _settings(tmp_path, redis_url=REDIS_URL, state_prefix=redis_prefix,
                         daily_spend_usd=1.00, rate_per_minute=10, rate_per_day=50)
    fake = FakeAnthropic(_ambiguous_answer())
    app = create_app(settings, schema=_synthetic_schema(), client=LLMClient(sdk_client=fake),
                     validation_client=LLMClient(sdk_client=fake, model=VALIDATION_MODEL))
    assert isinstance(app.state.qg.store, RedisState)
    client = TestClient(app)

    first = client.post("/v1/query", json={"question": "What was revenue?"}).json()
    again = client.post("/v1/query", json={"question": "What was revenue?"}).json()
    assert first["outcome"] == "clarification" and not first["cached"]
    assert again["cached"] and again["cost_usd"] == 0.0
    assert len(fake.calls) == 1, "the second answer came from the Redis cache"

    saved = client.post("/v1/feedback", json={"query_id": first["query_id"], "correct": False, "note": "net"})
    assert saved.status_code == 200
    history = client.get("/v1/history").json()
    assert [h["query_id"] for h in history] == [again["query_id"], first["query_id"]]
    assert history[1]["feedback"]["note"] == "net"
    assert client.get(f"/v1/history/{first['query_id']}").status_code == 200
    assert redis_client.zcard(f"{redis_prefix}reservations") == 0, "every reservation was released"
