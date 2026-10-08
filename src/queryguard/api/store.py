"""Query history, feedback and the response cache, in SQLite (LocalState's backend).

Not in Postgres on purpose: the API reaches Postgres only as `queryguard_ro`,
whose inability to write is the project's first boundary. Giving the API a
writable Postgres identity for its own bookkeeping would weaken that, so the
app's state lives in a separate file (data/app.db, gitignored).

Clients are stored as a salted hash of their IP, never the IP itself. The salt
is generated once per database and kept in it, so history stays scoped to a
client across restarts.

Every method opens its own short-lived connection, so the store is safe to call
from the worker threads asyncio.to_thread uses.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS queries (
    query_id    TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    client      TEXT NOT NULL,
    question    TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    confidence  REAL,
    cached      INTEGER NOT NULL,
    cost_usd    REAL NOT NULL,
    elapsed_ms  INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    sql_source  TEXT NOT NULL DEFAULT 'model',
    mode        TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS queries_by_client ON queries (client, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    query_id   TEXT PRIMARY KEY REFERENCES queries (query_id),
    correct    INTEGER NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    result_json TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
            # Databases created before sql_source or mode existed.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
            if "sql_source" not in columns:
                conn.execute("ALTER TABLE queries ADD COLUMN sql_source TEXT NOT NULL DEFAULT 'model'")
            if "mode" not in columns:
                conn.execute("ALTER TABLE queries ADD COLUMN mode TEXT NOT NULL DEFAULT 'live'")
            conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('client_salt', ?)", (secrets.token_hex(16),)
            )
            self._salt = conn.execute("SELECT value FROM meta WHERE key = 'client_salt'").fetchone()[0]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def client_key(self, ip: str) -> str:
        return hashlib.sha256(f"{self._salt}:{ip}".encode()).hexdigest()[:16]

    # ---------------------------------------------------------------- history

    def record(
        self, *, query_id: str, client: str, question: str, outcome: str, confidence: float | None,
        cached: bool, cost_usd: float, elapsed_ms: int, result: dict[str, Any], sql_source: str = "model",
        mode: str = "live",
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """INSERT INTO queries (query_id, created_at, client, question, outcome, confidence, cached,
                                        cost_usd, elapsed_ms, result_json, sql_source, mode)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (query_id, _now(), client, question, outcome, confidence, int(cached), cost_usd,
                 elapsed_ms, json.dumps(result, default=str), sql_source, mode),
            )

    def history(self, client: str, limit: int) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT q.query_id, q.created_at, q.question, q.outcome, q.confidence, q.cached,
                          q.cost_usd, q.sql_source, q.mode, f.correct, f.note, f.created_at AS feedback_at
                   FROM queries AS q LEFT JOIN feedback AS f USING (query_id)
                   WHERE q.client = ? ORDER BY q.created_at DESC LIMIT ?""",
                (client, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_result(self, query_id: str, client: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT result_json FROM queries WHERE query_id = ? AND client = ?", (query_id, client)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def exists(self, query_id: str) -> bool:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT 1 FROM queries WHERE query_id = ?", (query_id,)).fetchone() is not None

    # --------------------------------------------------------------- feedback

    def save_feedback(self, query_id: str, correct: bool, note: str | None) -> str:
        """Latest feedback on a query wins. Returns its timestamp."""
        created_at = _now()
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """INSERT INTO feedback VALUES (?, ?, ?, ?)
                   ON CONFLICT (query_id) DO UPDATE SET
                       correct = excluded.correct, note = excluded.note, created_at = excluded.created_at""",
                (query_id, int(correct), note, created_at),
            )
        return created_at

    def incorrect_feedback(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """SELECT q.query_id, q.created_at, q.question, q.outcome, q.confidence, q.result_json,
                          f.note, f.created_at AS feedback_at
                   FROM feedback AS f JOIN queries AS q USING (query_id)
                   WHERE f.correct = 0 ORDER BY f.created_at"""
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ cache

    def cache_get(self, key: str) -> dict[str, Any] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT result_json FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, result: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(result, default=str), _now())
            )
