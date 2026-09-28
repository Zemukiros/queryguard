"""Shared fixtures.

Most of these tests assert against the real seeded database, because the point
of introspection is that it reports what is actually there. When the container
is not running those tests skip with an actionable message rather than failing;
the pure-rendering tests still run, so the suite is never vacuous.
"""

from __future__ import annotations

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
