"""Shared state in Redis, for many instances: what production on Vercel uses.

Every key is under settings.state_prefix (e.g. "prod:"), so a preview
deployment pointed at the same database never sees production's counters.

Where a check and an update must not interleave with another instance -- the
rate limit, the spend reservation, the daily call cap -- the work is one Lua
script (state/lua/*.lua, commented line by line): Redis runs a script with no
other command in between. Writes that only need to land together use a
MULTI/EXEC pipeline (transaction=True), which is atomic but cannot branch on
what it reads.

Keys (all with the prefix; TTLs keep the free tier's 256 MB from filling):
    rate:<client>        sorted set of hit times                       24 h
    spend:<UTC date>     today's LLM spend, USD, INCRBYFLOAT            48 h
    reservations         sorted set id -> expiry ms                    ttl
    calls:<UTC date>     today's LLM call count                        48 h
    llmlog:<UTC date>    the last 1000 calls' cost fields (no prompt)  30 d
    cache:<key>          a cached QueryResult                           7 d
    q:<query_id>         one history row with its result               30 d
    hist:<client>        sorted set of the client's query ids          30 d
    fb:<query_id>        feedback on one query                         30 d
    fb:incorrect         sorted set of query ids marked incorrect      30 d
    salt                 the client-key salt, set once                 never
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from datetime import datetime, timezone
from importlib import resources
from typing import Any

import redis

from queryguard.api.settings import Settings
from queryguard.llm.client import CallGuard, RequestCapExceeded
from queryguard.state import AppState, Reservation

DAY_S = 86_400
SPEND_TTL_S = 2 * DAY_S
LOG_TTL_S = 30 * DAY_S
CACHE_TTL_S = 7 * DAY_S
HISTORY_TTL_S = 30 * DAY_S
HISTORY_KEEP = 100      # per client; matches the API's MAX_HISTORY
LOG_KEEP = 1000         # calls per day in llmlog
# The log keeps what spend accounting needs. The prompt hash and question text stay out.
_LOG_FIELDS = ("timestamp", "model", "input_tokens", "output_tokens", "cache_creation_input_tokens",
               "cache_read_input_tokens", "estimated_cost_usd", "latency_ms", "usage_unknown", "error")


def connect(url: str, max_connections: int = 50) -> redis.Redis:
    """A client whose pool makes callers wait for a free connection.

    The default pool raises MaxConnectionsError once every connection is in
    use; under a burst, a request should wait a moment instead of failing.
    """
    pool = redis.BlockingConnectionPool.from_url(
        url, max_connections=max_connections, timeout=10, decode_responses=True,
        socket_timeout=5, socket_connect_timeout=5, health_check_interval=30,
    )
    return redis.Redis(connection_pool=pool)


def _script(name: str) -> str:
    return resources.files("queryguard.state").joinpath("lua", name).read_text(encoding="utf-8")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisCallGuard(CallGuard):
    """The daily call cap and the spend ledger, shared by every instance."""

    def __init__(self, state: RedisState) -> None:
        self._state = state

    def before_call(self) -> None:
        s = self._state
        if not s._take_call(keys=[s._k(f"calls:{_today()}")], args=[s.settings.daily_call_cap]):
            raise RequestCapExceeded(
                f"the daily cap of {s.settings.daily_call_cap} LLM calls is reached; it resets at 00:00 UTC"
            )

    def after_call(self, entry: dict[str, Any], log: Any = None) -> None:
        s, day = self._state, _today()
        record = json.dumps({k: entry[k] for k in _LOG_FIELDS if k in entry}, sort_keys=True)
        with s.r.pipeline(transaction=True) as pipe:   # MULTI ... EXEC: all four land together
            pipe.incrbyfloat(s._k(f"spend:{day}"), float(entry.get("estimated_cost_usd", 0.0)))
            pipe.expire(s._k(f"spend:{day}"), SPEND_TTL_S)
            pipe.lpush(s._k(f"llmlog:{day}"), record)
            pipe.ltrim(s._k(f"llmlog:{day}"), 0, LOG_KEEP - 1)
            pipe.expire(s._k(f"llmlog:{day}"), LOG_TTL_S)
            pipe.execute()


class RedisState(AppState):
    def __init__(self, settings: Settings, client: redis.Redis | None = None) -> None:
        if client is None and not settings.redis_url:
            raise ValueError("RedisState needs settings.redis_url (QUERYGUARD_REDIS_URL)")
        self.settings = settings
        self.r = client or connect(settings.redis_url)
        self._prefix = settings.state_prefix
        # register_script sends EVALSHA and falls back to EVAL (loading the
        # script) the first time a server has not seen it.
        self._rate = self.r.register_script(_script("rate_limit.lua"))
        self._reserve = self.r.register_script(_script("reserve.lua"))
        self._take_call = self.r.register_script(_script("take_call.lua"))
        self._guard = RedisCallGuard(self)
        self._salt: str | None = None

    def _k(self, name: str) -> str:
        return f"{self._prefix}{name}"

    # ------------------------------------------------------------- identity

    def client_key(self, ip: str) -> str:
        if self._salt is None:
            # SET NX: the first instance ever to get here writes the salt; every
            # other instance (and every later run) reads that same value.
            self.r.set(self._k("salt"), secrets.token_hex(16), nx=True)
            self._salt = self.r.get(self._k("salt"))
        return hashlib.sha256(f"{self._salt}:{ip}".encode()).hexdigest()[:16]

    # ------------------------------------------------------------ admission

    def rate_hit(self, client: str) -> float | None:
        s = self.settings
        wait_ms = self._rate(keys=[self._k(f"rate:{client}")],
                             args=[s.rate_per_minute, s.rate_per_day, uuid.uuid4().hex])
        return None if int(wait_ms) < 0 else int(wait_ms) / 1000

    def try_reserve(self) -> tuple[Reservation | None, float]:
        s = self.settings
        reservation = Reservation(uuid.uuid4().hex)
        admitted, spent = self._reserve(
            keys=[self._k(f"spend:{_today()}"), self._k("reservations")],
            args=[s.daily_spend_usd, s.question_reserve_usd, int(s.reservation_ttl_s * 1000), reservation.id],
        )
        return (reservation if int(admitted) == 1 else None), float(spent)

    def release(self, reservation: Reservation) -> None:
        self.r.zrem(self._k("reservations"), reservation.id)

    def spent_today(self) -> float:
        return float(self.r.get(self._k(f"spend:{_today()}")) or 0.0)

    def call_guard(self) -> CallGuard:
        return self._guard

    # ---------------------------------------------------------------- cache

    def cache_get(self, key: str) -> dict[str, Any] | None:
        raw = self.r.get(self._k(f"cache:{key}"))
        return json.loads(raw) if raw else None

    def cache_put(self, key: str, result: dict[str, Any]) -> None:
        self.r.set(self._k(f"cache:{key}"), json.dumps(result, default=str), ex=CACHE_TTL_S)

    # -------------------------------------------------------------- history

    def record(
        self, *, query_id: str, client: str, question: str, outcome: str, confidence: float | None,
        cached: bool, cost_usd: float, elapsed_ms: int, result: dict[str, Any], sql_source: str = "model",
    ) -> None:
        row = {"query_id": query_id, "created_at": _now_iso(), "client": client, "question": question,
               "outcome": outcome, "confidence": confidence, "cached": cached, "cost_usd": cost_usd,
               "elapsed_ms": elapsed_ms, "sql_source": sql_source,
               "result_json": json.dumps(result, default=str)}
        hist = self._k(f"hist:{client}")
        with self.r.pipeline(transaction=True) as pipe:
            pipe.set(self._k(f"q:{query_id}"), json.dumps(row), ex=HISTORY_TTL_S)
            # Score in microseconds keeps rows in the order they were asked,
            # even when two land in the same millisecond.
            pipe.zadd(hist, {query_id: time.time_ns() // 1000})
            pipe.zremrangebyrank(hist, 0, -(HISTORY_KEEP + 1))   # keep only the newest 100
            pipe.expire(hist, HISTORY_TTL_S)
            pipe.execute()

    def _rows(self, query_ids: list[str]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
        """(row, feedback) for each id still stored, in the given order."""
        if not query_ids:
            return []
        raw_rows = self.r.mget([self._k(f"q:{q}") for q in query_ids])
        raw_fb = self.r.mget([self._k(f"fb:{q}") for q in query_ids])
        return [(json.loads(row), json.loads(fb) if fb else None)
                for row, fb in zip(raw_rows, raw_fb) if row]

    def history(self, client: str, limit: int) -> list[dict[str, Any]]:
        ids = self.r.zrevrange(self._k(f"hist:{client}"), 0, limit - 1)
        out = []
        for row, fb in self._rows(ids):
            out.append({
                **{k: row[k] for k in ("query_id", "created_at", "question", "outcome", "confidence",
                                       "cached", "cost_usd", "sql_source")},
                "correct": fb["correct"] if fb else None,
                "note": fb["note"] if fb else None,
                "feedback_at": fb["created_at"] if fb else None,
            })
        return out

    def get_result(self, query_id: str, client: str) -> dict[str, Any] | None:
        raw = self.r.get(self._k(f"q:{query_id}"))
        if not raw:
            return None
        row = json.loads(raw)
        return json.loads(row["result_json"]) if row["client"] == client else None

    def exists(self, query_id: str) -> bool:
        return bool(self.r.exists(self._k(f"q:{query_id}")))

    # ------------------------------------------------------------- feedback

    def save_feedback(self, query_id: str, correct: bool, note: str | None) -> str:
        created_at = _now_iso()
        incorrect = self._k("fb:incorrect")
        with self.r.pipeline(transaction=True) as pipe:
            pipe.set(self._k(f"fb:{query_id}"),
                     json.dumps({"correct": correct, "note": note, "created_at": created_at}), ex=HISTORY_TTL_S)
            if correct:
                pipe.zrem(incorrect, query_id)                       # latest feedback wins
            else:
                pipe.zadd(incorrect, {query_id: time.time_ns() // 1000})
            pipe.expire(incorrect, HISTORY_TTL_S)
            pipe.execute()
        return created_at

    def incorrect_feedback(self) -> list[dict[str, Any]]:
        ids = self.r.zrange(self._k("fb:incorrect"), 0, -1)        # oldest first
        return [
            {"query_id": row["query_id"], "created_at": row["created_at"], "question": row["question"],
             "outcome": row["outcome"], "confidence": row["confidence"], "result_json": row["result_json"],
             "note": fb["note"], "feedback_at": fb["created_at"]}
            for row, fb in self._rows(ids) if fb and not fb["correct"]
        ]
