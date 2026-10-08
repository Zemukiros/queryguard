"""Behaviour that changes on Vercel (VERCEL=1, set by Vercel itself).

The filesystem there is read-only apart from /tmp, instances are short-lived,
and the deployment holds only the read-only database URL. These tests switch
VERCEL on with monkeypatch; nothing here needs the platform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from queryguard.api.settings import Settings
from queryguard.config import log_target
from queryguard.llm.client import append_log


def test_append_log_can_print_or_drop(tmp_path, capsys) -> None:
    append_log({"b": 2, "a": 1}, Path("-"))
    assert json.loads(capsys.readouterr().out) == {"a": 1, "b": 2}
    append_log({"a": 1}, Path("off"))
    assert capsys.readouterr().out == ""
    written = append_log({"a": 1}, tmp_path / "x" / "log.jsonl")
    assert written.read_text() == '{"a": 1}\n'


def test_logs_default_to_stdout_or_off_on_vercel(monkeypatch) -> None:
    from queryguard.executor import log_path as executor_log
    from queryguard.llm.client import log_path as llm_log
    from queryguard.validation.confidence import features_log_path

    for name in ("QUERYGUARD_EXECUTOR_LOG", "QUERYGUARD_LLM_LOG", "QUERYGUARD_CONFIDENCE_LOG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    assert executor_log().name == "executions.jsonl"

    monkeypatch.setenv("VERCEL", "1")
    assert str(executor_log()) == "-"           # one JSON line per execution, in the function log
    assert str(llm_log()) == "off"               # the real ledger is Redis there
    assert str(features_log_path()) == "off"     # calibration is frozen
    monkeypatch.setenv("QUERYGUARD_EXECUTOR_LOG", "/tmp/x.jsonl")
    assert log_target("QUERYGUARD_EXECUTOR_LOG", "executions.jsonl", on_platform="-") == Path("/tmp/x.jsonl")


def test_vercel_settings_scope_state_to_the_environment(monkeypatch) -> None:
    monkeypatch.delenv("QUERYGUARD_STATE_PREFIX", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    assert Settings.from_env().state_prefix == "qg:"
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_ENV", "preview")
    settings = Settings.from_env()
    assert settings.vercel and settings.state_prefix == "preview:"
    monkeypatch.setenv("QUERYGUARD_STATE_PREFIX", "custom:")
    assert Settings.from_env().state_prefix == "custom:"


def test_on_vercel_the_client_ip_comes_from_x_real_ip(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from queryguard.api.app import create_app
    from tests.test_pipeline import _synthetic_schema

    vercel = Settings(db_path=tmp_path / "a.db", vercel=True)
    app = create_app(vercel, schema=_synthetic_schema())
    seen: list[str] = []
    original = app.state.qg.store.client_key
    app.state.qg.store.client_key = lambda ip: seen.append(ip) or original(ip)
    TestClient(app).get("/v1/history", headers={"x-real-ip": "198.51.100.4", "x-forwarded-for": "6.6.6.6"})
    assert seen == ["198.51.100.4"], "the spoofable x-forwarded-for is ignored"


def test_introspection_falls_back_to_the_readonly_role(monkeypatch, tmp_path, live_schema) -> None:
    """Vercel never holds the owner URL; a read-only filesystem must not stop the answer."""
    from queryguard.schema import introspect

    monkeypatch.delenv("DATABASE_URL", raising=False)
    import queryguard.config as config

    monkeypatch.setattr(config, "load_env", lambda: None)  # .env would supply the owner URL
    unwritable = tmp_path / "ro"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        schema = introspect.load_schema(path=unwritable / "nested" / "schema_cache.json")
    finally:
        unwritable.chmod(0o700)
    assert schema.model_dump(exclude={"extracted_at"}) == live_schema.model_dump(exclude={"extracted_at"})


def test_the_asgi_module_exposes_an_app(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("QUERYGUARD_APP_DB", str(tmp_path / "app.db"))
    import importlib

    import queryguard.api.asgi as asgi

    importlib.reload(asgi)
    assert asgi.app.title == "QueryGuard"


@pytest.fixture(autouse=True)
def _no_vercel_leak(monkeypatch):
    """Each test sets VERCEL itself; none may inherit it from the shell."""
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("VERCEL_ENV", raising=False)
