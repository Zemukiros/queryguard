"""Shared fixtures.

Most of these tests assert against the real seeded database, because the point
of introspection is that it reports what is actually there. When the container
is not running those tests skip with an actionable message rather than failing;
the pure-rendering tests still run, so the suite is never vacuous.
"""

from __future__ import annotations

import anthropic
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from queryguard.config import database_url
from queryguard.schema.introspect import DatabaseSchema, introspect_database


@pytest.fixture(scope="session")
def live_schema() -> DatabaseSchema:
    """Introspect the running database once for the whole session."""
    try:
        return introspect_database()
    except (SQLAlchemyError, RuntimeError) as exc:
        pytest.skip(f"queryguard-db not reachable, run `docker compose up -d` ({exc})")


@pytest.fixture(scope="session")
def live_database() -> None:
    """Skip when Postgres is not reachable through the read-only role.

    Separate from `live_schema` because the executor tests need a working
    `queryguard_ro` connection specifically -- the owner being reachable proves
    nothing about the role whose privileges are the thing under test.
    """
    try:
        engine = create_engine(database_url(readonly=True))
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        finally:
            engine.dispose()
    except (SQLAlchemyError, RuntimeError) as exc:
        pytest.skip(
            f"queryguard-db not reachable as queryguard_ro, "
            f"run `docker compose up -d` ({exc})"
        )


@pytest.fixture(autouse=True)
def _no_real_api_and_no_real_logs(tmp_path, monkeypatch):
    """No test may reach the API or write to logs/.

    A component that builds its own LLMClient when none is injected would
    otherwise construct a real anthropic.Anthropic -- and with a key in .env,
    spend money on every run. This happened once (Phase 3 part 2, 16 Haiku
    calls) before this fixture existed; constructing the SDK client now fails
    the test instead. logs/confidence_features.jsonl is calibration training
    data, so a test run writing fake rows into it would be worse than a crash.
    """

    def _refuse(*args, **kwargs):
        raise RuntimeError("a test tried to construct a real anthropic.Anthropic; inject a fake")

    monkeypatch.setattr(anthropic, "Anthropic", _refuse)
    monkeypatch.setenv("QUERYGUARD_LLM_LOG", str(tmp_path / "llm_calls.jsonl"))
    monkeypatch.setenv("QUERYGUARD_EXECUTOR_LOG", str(tmp_path / "executions.jsonl"))
    monkeypatch.setenv("QUERYGUARD_CONFIDENCE_LOG", str(tmp_path / "confidence_features.jsonl"))
