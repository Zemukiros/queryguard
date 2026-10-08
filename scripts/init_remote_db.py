"""Seed a remote Postgres (Neon) from db/init and prove the read-only role holds.

    make neon-init        # reads .env.neon; prints no connection string or password

.env.neon (gitignored by `.env.*`) must hold:
    NEON_OWNER_URL         the owner's DIRECT (unpooled) connection string
    POSTGRES_RO_PASSWORD   the password to give queryguard_ro (>= 60 bits of entropy for Neon)

Steps:
1. Run db/init/*.sql in order as the owner, with the psql from the postgres:16
   image (03_readonly_user.sql needs psql's \\getenv). Secrets reach the
   container only as environment variables; the command line names them and
   never contains them. Skipped when the tables and the role already exist,
   so a re-run only re-verifies.
2. Build DATABASE_URL_READONLY from the owner URL: user queryguard_ro, the
   password percent-encoded (base64 has + / =), same host, database and
   query string. Written back into .env.neon, never printed.
3. Run scripts/verify_db.py against both URLs: row counts, and every write
   attempt by the read-only role must be refused.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import dotenv_values

REPO = Path(__file__).resolve().parent.parent
ENV_FILE = REPO / ".env.neon"
INIT_DIR = REPO / "db" / "init"
PSQL_IMAGE = "postgres:16"
RO_USER = "queryguard_ro"


def fail(message: str) -> int:
    print(f"FAIL  {message}")
    return 1


def psql(env: dict[str, str], *args: str) -> subprocess.CompletedProcess:
    """psql in a throwaway container. The URL is expanded inside it, from the environment."""
    return subprocess.run(
        ["docker", "run", "--rm", "-i", "-e", "NEON_OWNER_URL", "-e", "POSTGRES_RO_PASSWORD",
         "-v", f"{INIT_DIR}:/init:ro", PSQL_IMAGE,
         "sh", "-c", 'exec psql "$NEON_OWNER_URL" -v ON_ERROR_STOP=1 -q "$@"', "psql", *args],
        env=env, capture_output=True, text=True,
    )


def readonly_url(owner_url: str, password: str) -> str:
    parts = urlsplit(owner_url)
    if not parts.hostname:
        raise ValueError("NEON_OWNER_URL has no host")
    host = parts.hostname + (f":{parts.port}" if parts.port else "")
    return urlunsplit((parts.scheme, f"{RO_USER}:{quote(password, safe='')}@{host}",
                       parts.path, parts.query, parts.fragment))


def write_env_value(name: str, value: str) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if not re.match(rf"\s*{name}\s*=", line)]
    lines.append(f"{name}={value}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ENV_FILE.chmod(0o600)


def main() -> int:
    if not ENV_FILE.is_file():
        return fail(f"{ENV_FILE.name} not found")
    values = dotenv_values(ENV_FILE)
    missing = [k for k in ("NEON_OWNER_URL", "POSTGRES_RO_PASSWORD") if not values.get(k)]
    if missing:
        return fail(f"{ENV_FILE.name} is missing {', '.join(missing)}")
    owner_url, password = values["NEON_OWNER_URL"], values["POSTGRES_RO_PASSWORD"]
    ENV_FILE.chmod(0o600)
    if "-pooler" in (urlsplit(owner_url).hostname or ""):
        return fail("NEON_OWNER_URL is the pooled endpoint; use the direct (unpooled) connection string")

    env = {**os.environ, "NEON_OWNER_URL": owner_url, "POSTGRES_RO_PASSWORD": password}

    # The last things each script does: 02 seeds orders, 03 ends by setting the
    # role's read-only session default, 04 comments orders.total_amount.
    state = psql(env, "-tA", "-c",
                 "SELECT to_regclass('public.orders') IS NOT NULL"
                 " AND EXISTS (SELECT 1 FROM public.orders), "
                 f"EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{RO_USER}'), "
                 "EXISTS (SELECT 1 FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole"
                 f" WHERE r.rolname = '{RO_USER}' AND 'default_transaction_read_only=on' = ANY (s.setconfig)), "
                 "to_regclass('public.orders') IS NOT NULL"
                 " AND col_description('public.orders'::regclass, (SELECT attnum FROM pg_attribute"
                 " WHERE attrelid = 'public.orders'::regclass AND attname = 'total_amount')) IS NOT NULL, "
                 "current_setting('server_version_num')::int / 10000")
    if state.returncode != 0:
        return fail(f"could not connect as the owner: {state.stderr.strip().splitlines()[-1:]}")
    seeded, role_exists, role_done, commented, major = state.stdout.strip().split("|")
    print(f"ok    connected to Postgres {major} as the owner")

    if seeded != "t":
        if role_exists == "t":
            return fail("the read-only role exists but no data does; inspect before re-running")
        pending = sorted(INIT_DIR.glob("*.sql"))
    elif role_done == "t" and commented == "t":
        pending = []
        print("ok    already initialised; skipping init")
    else:
        # Seeded, but 03 or 04 did not finish. The data is fine; the role is
        # rebuilt from scratch so no half-applied grant survives.
        if role_exists == "t":
            # DROP OWNED needs the role's privileges (PG16). The owner created
            # the role, so it may grant itself membership; the role, and with
            # it the membership, is gone two statements later.
            result = psql(env, "-c", f"GRANT {RO_USER} TO CURRENT_USER; DROP OWNED BY {RO_USER}; DROP ROLE {RO_USER};")
            if result.returncode != 0:
                return fail(f"dropping the half-made role: {result.stderr.strip()[-500:]}")
            print("ok    dropped the half-made read-only role to rebuild it")
        pending = [p for p in sorted(INIT_DIR.glob("*.sql")) if p.name >= "03"]

    for script in pending:
        result = psql(env, "-f", f"/init/{script.name}")
        if result.returncode != 0:
            return fail(f"{script.name}: {result.stderr.strip()[-500:]}")
        print(f"ok    applied {script.name}")

    write_env_value("DATABASE_URL_READONLY", readonly_url(owner_url, password))
    print(f"ok    wrote DATABASE_URL_READONLY to {ENV_FILE.name} (password percent-encoded)")

    verify = subprocess.run(
        ["uv", "run", "scripts/verify_db.py"], cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "DATABASE_URL": owner_url,
             "DATABASE_URL_READONLY": readonly_url(owner_url, password)},
    )
    # verify_db prints which user and database it connected to; those are parts
    # of the connection strings, so they stay out of this output.
    for line in verify.stdout.splitlines():
        if "connected as" not in line:
            print(line)
    if verify.returncode != 0:
        print(verify.stderr.strip()[-500:])
        return fail("verify_db.py failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
