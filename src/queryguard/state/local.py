"""One-process state: what `make dev`, the tests and a single container use.

Rate limits and reservations live in memory and reset when the process
restarts; spend is read from the JSONL call log and history from SQLite, so
both survive a restart on the same disk. None of it is shared between
processes -- that is RedisState's job.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from queryguard.api.limits import RateLimiter, spent_today_usd
from queryguard.api.settings import Settings
from queryguard.api.store import Store
from queryguard.llm.client import PROCESS_GUARD, CallGuard
from queryguard.state import AppState, Reservation


class LocalState(AppState):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._store = Store(settings.db_path)
        self._limiter = RateLimiter(settings.rate_per_minute, settings.rate_per_day)
        self._reserved: set[str] = set()
        self._lock = threading.Lock()

    def client_key(self, ip: str) -> str:
        return self._store.client_key(ip)

    # ------------------------------------------------------------ admission

    def rate_hit(self, client: str) -> float | None:
        return self._limiter.hit(client)

    def try_reserve(self) -> tuple[Reservation | None, float]:
        # Today's spend only learns a call's cost after the call, so questions
        # already running are counted at the reserve each until they finish.
        with self._lock:
            spent = spent_today_usd()
            held = (len(self._reserved) + 1) * self.settings.question_reserve_usd
            if spent + held > self.settings.daily_spend_usd:
                return None, spent
            reservation = Reservation(uuid.uuid4().hex)
            self._reserved.add(reservation.id)
            return reservation, spent

    def release(self, reservation: Reservation) -> None:
        with self._lock:
            self._reserved.discard(reservation.id)

    def spent_today(self) -> float:
        return spent_today_usd()

    def call_guard(self) -> CallGuard:
        return PROCESS_GUARD

    # ---------------------------------------------------- cache and history

    def cache_get(self, key: str) -> dict[str, Any] | None:
        return self._store.cache_get(key)

    def cache_put(self, key: str, result: dict[str, Any]) -> None:
        self._store.cache_put(key, result)

    def record(self, **fields: Any) -> None:
        self._store.record(**fields)

    def history(self, client: str, limit: int) -> list[dict[str, Any]]:
        return self._store.history(client, limit)

    def get_result(self, query_id: str, client: str) -> dict[str, Any] | None:
        return self._store.get_result(query_id, client)

    def exists(self, query_id: str) -> bool:
        return self._store.exists(query_id)

    def save_feedback(self, query_id: str, correct: bool, note: str | None) -> str:
        return self._store.save_feedback(query_id, correct, note)

    def incorrect_feedback(self) -> list[dict[str, Any]]:
        return self._store.incorrect_feedback()
