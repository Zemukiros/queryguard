"""Shared fixtures.

Most of these tests assert against the real seeded database, because the point
of introspection is that it reports what is actually there. When the container
is not running those tests skip with an actionable message rather than failing;
the pure-rendering tests still run, so the suite is never vacuous.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import SQLAlchemyError

from queryguard.schema.introspect import DatabaseSchema, introspect_database


@pytest.fixture(scope="session")
def live_schema() -> DatabaseSchema:
    """Introspect the running database once for the whole session."""
    try:
        return introspect_database()
    except (SQLAlchemyError, RuntimeError) as exc:
        pytest.skip(f"queryguard-db not reachable, run `docker compose up -d` ({exc})")
