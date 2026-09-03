#!/usr/bin/env python3
"""Verify the QueryGuard database: row counts, and proof the read-only role cannot write.

Usage:  uv run scripts/verify_db.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from psycopg import sql
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tables that must exist and be non-empty, in dependency order.
EXPECTED_TABLES = [
    "categories",
    "customers",
    "products",
    "orders",
    "order_items",
    "refunds",
]


def fail(message: str) -> None:
    print(f"  FAIL  {message}")


def ok(message: str) -> None:
    print(f"  ok    {message}")


def list_tables(conn: psycopg.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )
        return [row[0] for row in cur.fetchall()]


def row_counts(conn: psycopg.Connection, tables: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    with conn.cursor() as cur:
        for table in tables:
            # Table names come from information_schema, not user input, but quote anyway.
            cur.execute(
                sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table))
            )
            counts[table] = cur.fetchone()[0]
    return counts


def print_counts(title: str, counts: dict[str, int]) -> None:
    width = max((len(t) for t in counts), default=10)
    print(f"\n{title}")
    print(f"  {'table'.ljust(width)}  {'rows':>8}")
    print(f"  {'-' * width}  {'-' * 8}")
    for table, count in counts.items():
        print(f"  {table.ljust(width)}  {count:>8,}")


def expect_denied(conn: psycopg.Connection, label: str, statement: str) -> bool:
    """Run a write statement that must be rejected. Returns True if it was."""
    try:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute(statement)
            # Never reached unless the write succeeded; raising rolls it back.
            raise _Allowed()
    except _Allowed:
        fail(f"{label}: statement was ALLOWED — the read-only boundary is broken!")
        return False
    except psycopg.errors.InsufficientPrivilege as exc:
        ok(f"{label}: denied (SQLSTATE {exc.sqlstate}) — {str(exc).strip().splitlines()[0]}")
        return True
    except psycopg.errors.ReadOnlySqlTransaction as exc:
        ok(f"{label}: denied (SQLSTATE {exc.sqlstate}) — {str(exc).strip().splitlines()[0]}")
        return True
    except psycopg.Error as exc:
        fail(f"{label}: rejected for the WRONG reason (SQLSTATE {exc.sqlstate}) — {exc}")
        return False


class _Allowed(Exception):
    """Raised to roll back a write that unexpectedly succeeded."""


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")

    rw_url = os.getenv("DATABASE_URL")
    ro_url = os.getenv("DATABASE_URL_READONLY")
    missing = [n for n, v in (("DATABASE_URL", rw_url), ("DATABASE_URL_READONLY", ro_url)) if not v]
    if missing:
        print(f"ERROR: {', '.join(missing)} not set (expected in {REPO_ROOT / '.env'})")
        return 1

    failures = 0

    # ---------------------------------------------------------------- owner
    print("=" * 62)
    print("1. Owner connection (DATABASE_URL)")
    print("=" * 62)
    with psycopg.connect(rw_url) as rw:
        with rw.cursor() as cur:
            cur.execute("SELECT current_user, current_database(), version()")
            user, db, version = cur.fetchone()
        print(f"  connected as {user!r} to {db!r}")
        print(f"  {version.split(' on ')[0]}")

        owner_tables = list_tables(rw)
        owner_counts = row_counts(rw, owner_tables)
        print_counts("Row counts (owner):", owner_counts)

        for table in EXPECTED_TABLES:
            if table not in owner_counts:
                fail(f"expected table {table!r} is missing")
                failures += 1
            elif owner_counts[table] == 0:
                fail(f"table {table!r} is empty")
                failures += 1

    # ------------------------------------------------------------ read-only
    print()
    print("=" * 62)
    print("2. Read-only connection (DATABASE_URL_READONLY)")
    print("=" * 62)
    with psycopg.connect(ro_url) as ro:
        ro.autocommit = True  # each check manages its own transaction
        with ro.cursor() as cur:
            cur.execute("SELECT current_user, current_database()")
            user, db = cur.fetchone()
        print(f"  connected as {user!r} to {db!r}")

        with ro.cursor() as cur:
            cur.execute("SHOW default_transaction_read_only")
            print(f"  session default_transaction_read_only = {cur.fetchone()[0]}")
            # That session default is a safety net, not the boundary: it is a USERSET
            # GUC any role can flip. Turn it off so the write attempts below are
            # rejected by GRANTs alone (SQLSTATE 42501), which is the real proof.
            cur.execute("SET default_transaction_read_only = off")
        print("  (turned it off, so the checks below test GRANTs, not the safety net)")

        ro_tables = list_tables(ro)
        ro_counts = row_counts(ro, ro_tables)
        print_counts("Row counts (read-only user — SELECT works):", ro_counts)

        if ro_counts != owner_counts:
            fail("read-only user sees different data than the owner")
            failures += 1
        else:
            ok("read-only user sees exactly the same tables and counts as the owner")

        print("\n  Write attempts (all of these must be refused):")
        checks = [
            (
                "INSERT INTO customers",
                "INSERT INTO customers (first_name, last_name, email, country) "
                "VALUES ('Mallory', 'Injection', 'mallory.injection@example.com', 'USA')",
            ),
            ("UPDATE products", "UPDATE products SET price = 0"),
            ("DELETE FROM orders", "DELETE FROM orders WHERE true"),
            ("CREATE TABLE", "CREATE TABLE public.should_not_exist (id integer)"),
            ("DROP TABLE", "DROP TABLE public.refunds"),
        ]
        for label, statement in checks:
            if not expect_denied(ro, label, statement):
                failures += 1

    # ----------------------------------------------------------- conclusion
    print()
    print("=" * 62)
    if failures:
        print(f"FAIL — {failures} check(s) failed")
        return 1
    print("PASS — schema seeded and the read-only boundary holds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
