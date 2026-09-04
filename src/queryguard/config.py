"""Environment and path resolution shared by every QueryGuard component.

The project has two database identities and they must never be confused:
DATABASE_URL is the owner (used for schema introspection) and
DATABASE_URL_READONLY is the `queryguard_ro` role (used to execute
LLM-generated SQL). Keeping both behind one helper makes the choice explicit
at every call site.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url


def _find_repo_root() -> Path:
    """Walk up from this file until a pyproject.toml appears.

    Walking beats a fixed number of `.parent` hops because the package can be
    imported from an editable install, a wheel, or the source tree directly.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    # src/queryguard/config.py -> src/queryguard -> src -> repo root
    return here.parents[2]


REPO_ROOT = _find_repo_root()

_ENV_LOADED = False


def load_env() -> None:
    """Load .env once per process. Existing environment variables win."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(REPO_ROOT / ".env")
        _ENV_LOADED = True


def schema_cache_path() -> Path:
    """Where the introspected schema is cached. Overridable for tests."""
    override = os.getenv("QUERYGUARD_SCHEMA_CACHE")
    return Path(override) if override else REPO_ROOT / "schema_cache.json"


def database_url(*, readonly: bool = False) -> URL:
    """Return a SQLAlchemy URL for the owner (default) or read-only role.

    Returned as a URL object rather than a string so that repr()/logging
    redacts the password by default.
    """
    load_env()

    var = "DATABASE_URL_READONLY" if readonly else "DATABASE_URL"
    raw = os.getenv(var)
    if not raw:
        raise RuntimeError(f"{var} is not set (expected in {REPO_ROOT / '.env'})")

    url = make_url(raw)
    # .env carries the bare `postgresql://` scheme, which SQLAlchemy maps to
    # psycopg2 — a driver this project does not install. Pin psycopg 3.
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    return url
