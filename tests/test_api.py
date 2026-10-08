"""Tests for the HTTP API: streaming order, cost controls, cache, feedback, schema.

Every test injects fake-backed LLM clients and a synthetic schema, so nothing
here can reach the API. The normal-question test executes for real against the
read-only role and takes `live_database`; the rest stop before execution.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import yaml
from fastapi.testclient import TestClient

from queryguard.api.app import create_app, normalize_question
from queryguard.api.limits import RateLimiter, spent_today_usd
from queryguard.api.settings import Settings
from queryguard.llm.client import LLMClient, log_path, reset_request_count
from queryguard.validation.backtranslate import VALIDATION_MODEL
from tests.test_pipeline import FakeAnthropic, _ambiguous_answer, _answer, _synthetic_schema

COUNT_SQL = "SELECT count(*) AS n FROM orders WHERE status = 'cancelled'"


@pytest.fixture(autouse=True)
def _isolate_counter():
    reset_request_count()
    yield
    reset_request_count()


def _client(tmp_path, parsed, **settings) -> tuple[TestClient, FakeAnthropic]:
    fake = FakeAnthropic(parsed)
    app = create_app(
        Settings(db_path=tmp_path / "app.db", **settings),
        schema=_synthetic_schema(),
        client=LLMClient(sdk_client=fake),
        validation_client=LLMClient(sdk_client=fake, model=VALIDATION_MODEL),
    )
    return TestClient(app), fake


def _events(response) -> list[tuple[str, dict]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        data = json.loads(fields["data"])
        assert data["stage"] == fields["event"]
        events.append((fields["event"], data))
    return events


def _stream(client: TestClient, question: str):
    return client.post("/v1/query/stream", json={"question": question})


# ---------------------------------------------------------------- event order


def test_a_normal_question_streams_every_stage_in_order(tmp_path, live_database) -> None:
    client, fake = _client(tmp_path, _answer(COUNT_SQL))
    events = _events(_stream(client, "How many orders were cancelled?"))

    assert [stage for stage, _ in events] == [
        "generating", "guardrails", "executing", "sanity",
        "backtranslate", "agreement", "confidence", "done",
    ]
    elapsed = [data["elapsed_ms"] for _, data in events]
    assert elapsed == sorted(elapsed)
    done = events[-1][1]["payload"]
    assert done["outcome"] == "answered" and done["cached"] is False
    assert done["columns"] == ["n"] and done["rows"][0][0] > 0
    assert done["agreement"] == "agree"
    assert 0 <= done["confidence"] <= 1
    assert done["n_calls"] == len(fake.calls) == 4  # an aggregate: every validation step ran


def test_a_clarification_streams_and_stops(tmp_path) -> None:
    client, fake = _client(tmp_path, _ambiguous_answer())
    events = _events(_stream(client, "What was revenue?"))

    assert [stage for stage, _ in events] == ["generating", "clarification", "done"]
    assert len(events[1][1]["payload"]["interpretations"]) == 2
    done = events[-1][1]["payload"]
    assert done["outcome"] == "clarification" and done["sql"] is None
    assert len(fake.calls) == 1


def test_a_guardrail_rejection_streams_no_execution(tmp_path) -> None:
    client, fake = _client(tmp_path, _answer("SELECT 1; DELETE FROM orders"))
    events = _events(_stream(client, "Delete the orders"))

    assert [stage for stage, _ in events] == ["generating", "guardrails", "confidence", "done"]
    guardrails = events[1][1]["payload"]
    assert guardrails["allowed"] is False and guardrails["rule"]
    done = events[-1][1]["payload"]
    assert done["outcome"] == "blocked" and done["executed_sql"] is None
    assert done["confidence"] == 0.0
    assert len(fake.calls) == 1  # nothing validated a query that never ran


def test_an_exception_becomes_a_final_error_event_without_internals(tmp_path) -> None:
    client, _ = _client(tmp_path, RuntimeError("connection to db.internal:5432 refused"))
    events = _events(_stream(client, "How many orders?"))

    assert [stage for stage, _ in events] == ["error"]
    error = events[0][1]["payload"]
    assert error["failed_stage"] == "generating" and error["error_type"] == "RuntimeError"
    assert "db.internal" not in json.dumps(error)

    response = client.post("/v1/query", json={"question": "How many orders, again?"})
    assert response.status_code == 500
    assert response.json()["failed_stage"] == "generating"
    assert "db.internal" not in response.text


# ---------------------------------------------------------------- cost controls


def test_the_rate_limit_answers_429_with_retry_after(tmp_path) -> None:
    client, fake = _client(tmp_path, _ambiguous_answer(), rate_per_minute=2)

    for question in ("revenue one?", "revenue two?"):
        assert client.post("/v1/query", json={"question": question}).status_code == 200
    refused = client.post("/v1/query", json={"question": "revenue three?"})

    assert refused.status_code == 429
    assert 1 <= int(refused.headers["retry-after"]) <= 60
    assert len(fake.calls) == 2


def test_the_rate_limiter_has_a_daily_window_too() -> None:
    now = [0.0]
    limiter = RateLimiter(per_minute=10, per_day=3, clock=lambda: now[0])
    for _ in range(3):
        assert limiter.hit("a") is None
        now[0] += 120
    assert limiter.hit("a") == pytest.approx(86_400 - 360)
    assert limiter.hit("b") is None  # per client


def _spend_today(amount: float) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "estimated_cost_usd": amount}
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
        handle.write('{"timestamp": "2000-01-01T00:00:00+00:00", "estimated_cost_usd": 99}\n')
        handle.write('{"truncated')


def test_the_spend_ceiling_answers_503_before_any_call(tmp_path) -> None:
    _spend_today(0.98)
    assert spent_today_usd() == pytest.approx(0.98)  # other days and a torn line are ignored
    client, fake = _client(tmp_path, _answer(COUNT_SQL), daily_spend_usd=1.00)

    for endpoint in ("/v1/query", "/v1/query/stream"):
        response = client.post(endpoint, json={"question": "How many orders were cancelled?"})
        assert response.status_code == 503
        assert "daily budget" in response.json()["detail"]
        assert int(response.headers["retry-after"]) > 0
    assert fake.calls == []


# ---------------------------------------------------------------------- cache


def test_a_repeated_question_is_served_from_cache_for_free(tmp_path) -> None:
    client, fake = _client(tmp_path, _ambiguous_answer())

    first = client.post("/v1/query", json={"question": "What was revenue?"}).json()
    again = client.post("/v1/query", json={"question": "  what WAS revenue  "}).json()
    streamed = _events(_stream(client, "What was revenue"))

    assert first["cached"] is False and first["n_calls"] == 1
    assert again["cached"] is True and again["cost_usd"] == 0.0 and again["n_calls"] == 0
    assert again["query_id"] != first["query_id"]
    assert again["interpretations"] == first["interpretations"]
    assert [stage for stage, _ in streamed] == ["done"] and streamed[0][1]["payload"]["cached"] is True
    assert len(fake.calls) == 1


def test_a_cache_hit_needs_no_rate_limit_allowance(tmp_path) -> None:
    client, fake = _client(tmp_path, _ambiguous_answer(), rate_per_minute=1)
    for _ in range(3):
        assert client.post("/v1/query", json={"question": "What was revenue?"}).status_code == 200
    assert len(fake.calls) == 1


def test_failures_are_not_cached(tmp_path) -> None:
    client, fake = _client(tmp_path, RuntimeError("boom"))
    for _ in range(2):
        assert client.post("/v1/query", json={"question": "How many orders?"}).status_code == 500
    assert len(fake.calls) == 2


def test_question_normalisation() -> None:
    assert normalize_question("  How many  ORDERS? ") == normalize_question("how many orders")
    assert normalize_question("orders by country") != normalize_question("orders by city")


# ------------------------------------------------------- feedback and history


def test_feedback_round_trip_and_export(tmp_path) -> None:
    client, _ = _client(tmp_path, _ambiguous_answer())
    query_id = client.post("/v1/query", json={"question": "What was revenue?"}).json()["query_id"]

    saved = client.post("/v1/feedback", json={"query_id": query_id, "correct": False, "note": "wanted net"})
    assert saved.status_code == 200 and saved.json()["correct"] is False

    [item] = client.get("/v1/history").json()
    assert item["query_id"] == query_id and item["outcome"] == "clarification"
    assert item["feedback"] == {"correct": False, "note": "wanted net", "created_at": saved.json()["created_at"]}

    assert client.post("/v1/feedback", json={"query_id": "nope", "correct": True}).status_code == 404

    out = tmp_path / "feedback_candidates.yaml"
    path, count = client.app.state.qg.store.export_feedback_candidates(out)
    [candidate] = yaml.safe_load(path.read_text())["candidates"]
    assert count == 1
    assert candidate["question"] == "What was revenue?" and candidate["feedback_note"] == "wanted net"
    assert candidate["golden_sql"] is None


def test_correct_feedback_is_not_exported(tmp_path) -> None:
    client, _ = _client(tmp_path, _ambiguous_answer())
    query_id = client.post("/v1/query", json={"question": "What was revenue?"}).json()["query_id"]
    client.post("/v1/feedback", json={"query_id": query_id, "correct": True})
    _, count = client.app.state.qg.store.export_feedback_candidates(tmp_path / "out.yaml")
    assert count == 0


def test_history_is_scoped_to_the_caller(tmp_path) -> None:
    client, _ = _client(tmp_path, _ambiguous_answer(), trust_proxy=True)
    client.post("/v1/query", json={"question": "What was revenue?"}, headers={"x-forwarded-for": "1.1.1.1"})
    assert len(client.get("/v1/history", headers={"x-forwarded-for": "1.1.1.1"}).json()) == 1
    assert client.get("/v1/history", headers={"x-forwarded-for": "2.2.2.2"}).json() == []


# --------------------------------------------------------- schema and openapi


def test_schema_endpoint(tmp_path) -> None:
    client, _ = _client(tmp_path, _ambiguous_answer())
    body = client.get("/v1/schema").json()
    assert "orders" in body["rendered"]
    assert [t["name"] for t in body["tables"]] == ["orders"]
    assert len(body["schema_hash"]) == 64


def test_openapi_schema_generates(tmp_path) -> None:
    client, _ = _client(tmp_path, _ambiguous_answer())
    spec = client.get("/openapi.json").json()
    assert {"/v1/query", "/v1/query/stream", "/v1/schema", "/v1/history", "/v1/feedback", "/healthz"} <= set(spec["paths"])
    assert {"QueryResult", "StageEvent", "ErrorResponse", "FeedbackRequest"} <= set(spec["components"]["schemas"])


def test_question_validation(tmp_path) -> None:
    client, fake = _client(tmp_path, _ambiguous_answer())
    assert client.post("/v1/query", json={"question": "   "}).status_code == 422
    assert client.post("/v1/query", json={"question": "x" * 501}).status_code == 422
    assert fake.calls == []


def test_cors_allows_only_the_configured_origin(tmp_path) -> None:
    client, _ = _client(tmp_path, _ambiguous_answer(), frontend_origins=("https://demo.example",))
    allowed = client.options("/v1/query", headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"})
    other = client.options("/v1/query", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "https://demo.example"
    assert "access-control-allow-origin" not in other.headers
