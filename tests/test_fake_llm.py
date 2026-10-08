"""The demo fake (llm/fake.py) and QUERYGUARD_FAKE_LLM mode. No API, ever.

The ambiguous readings are hand-written SQL, so they are executed here: a
reading the guardrail refuses or the database rejects would break the demo.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from evals.common import load_golden, run_guarded
from queryguard.api.app import create_app
from queryguard.api.settings import Settings
from queryguard.llm import fake
from queryguard.llm.client import reset_request_count


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    reset_request_count()
    monkeypatch.setenv("QUERYGUARD_FAKE_LLM_DELAY_MS", "0")
    # Fake mode setdefault()s these; owning them here means the test undoes it.
    monkeypatch.setenv("QUERYGUARD_MAX_REQUESTS", "50")
    yield
    reset_request_count()


@pytest.fixture
def client(tmp_path) -> TestClient:
    return TestClient(create_app(Settings(db_path=tmp_path / "app.db", fake_llm=True)))


def _events(response) -> list[tuple[str, dict]]:
    assert response.status_code == 200, response.text
    out = []
    for block in response.text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((fields["event"], json.loads(fields["data"])["payload"]))
    return out


def test_every_ambiguous_golden_question_has_readings() -> None:
    ambiguous = {e["id"] for e in load_golden() if e.get("expected_outcome") == "clarification"}
    assert ambiguous == set(fake.READINGS)


@pytest.mark.parametrize("golden_id", sorted(fake.READINGS))
def test_every_reading_passes_the_guardrail_and_executes(golden_id, live_database) -> None:
    for label, sql, _ in fake.READINGS[golden_id]:
        run = run_guarded(sql)
        assert run.ok, f"{golden_id}/{label}: {run.error}"
        assert run.execution.row_count > 0, f"{golden_id}/{label} returned no rows"


def test_generation_follows_the_golden_set() -> None:
    by_id = {e["id"]: e for e in load_golden()}
    assert fake.generate(by_id["lookup_01"]["question"]).sql == by_id["lookup_01"]["golden_sql"].strip()
    assert fake.generate(by_id["ambig_02"]["question"]).ambiguity.is_ambiguous
    assert fake.generate(by_id["unans_02"]["question"]).is_refusal
    assert fake.generate("What is the meaning of life?").is_refusal


def test_back_translation_and_judge_round_trip() -> None:
    entry = next(e for e in load_golden() if e["id"] == "lookup_01")
    back = fake.back_translate(entry["golden_sql"] + ";\n")
    assert back.question == entry["question"]
    assert fake.judge(entry["question"], back.question).alignment == 1.0
    assert fake.judge(entry["question"], fake.back_translate("SELECT 1").question).alignment < 0.7


def test_fake_mode_answers_a_golden_question_end_to_end(client, live_database) -> None:
    events = _events(client.post("/v1/query/stream", json={"question": "How many orders were cancelled?"}))
    stages = [s for s, _ in events]
    assert stages[0] == "generating" and stages[-1] == "done"
    done = events[-1][1]
    assert done["outcome"] == "answered" and done["agreement"] == "agree" and done["alignment"] == 1.0
    assert done["cost_usd"] == 0.0 and done["n_calls"] == 4
    assert client.get("/healthz").json()["fake_llm"] is True


def test_fake_mode_clarification_then_a_reading_runs(client, live_database) -> None:
    events = _events(client.post("/v1/query/stream", json={"question": "Who are our top 10 customers?"}))
    assert [s for s, _ in events] == ["generating", "clarification", "done"]
    reading = events[1][1]["interpretations"][0]

    ran = _events(client.post("/v1/run/stream", json={"question": "Who are our top 10 customers?",
                                                       "sql": reading["sql"]}))
    done = ran[-1][1]
    assert done["outcome"] == "answered" and done["row_count"] == 10 and done["sql_source"] == "user"


def test_fake_mode_refuses_unknown_questions(client) -> None:
    done = client.post("/v1/query", json={"question": "What is the meaning of life?"}).json()
    assert done["outcome"] == "cannot_answer" and "golden-set" in done["cannot_answer_reason"]


def test_fake_mode_moves_the_log_and_lifts_the_cap(tmp_path, monkeypatch) -> None:
    from queryguard.llm.client import log_path, max_requests

    monkeypatch.delenv("QUERYGUARD_LLM_LOG")
    monkeypatch.delenv("QUERYGUARD_MAX_REQUESTS")
    client = TestClient(create_app(Settings(db_path=tmp_path / "app.db", fake_llm=True)))
    assert log_path().name == "fake_llm_calls.jsonl"
    assert max_requests() > 10**6
    assert client.get("/healthz").json()["budget"]["spent_today_usd"] == 0.0
