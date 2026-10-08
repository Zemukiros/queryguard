"""Per-client rate limiting and the global daily spend ceiling.

Both are checked before a question reaches the pipeline, so a refused request
costs nothing. A cache hit is checked before either and consumes neither: it
makes no API call, so there is nothing to protect.

These are the one-process versions, used by queryguard.state.LocalState
(which also holds the in-flight spend reservations). The rate limiter is in
memory and resets when the process restarts: fine for one instance, wrong for
several -- RedisState shares both across instances.
"""

from __future__ import annotations

import bisect
import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from queryguard.llm.client import log_path

MINUTE_S = 60.0
DAY_S = 86_400.0


class RateLimiter:
    """Rolling one-minute and 24-hour windows per client key."""

    def __init__(self, per_minute: int, per_day: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record a request. None if allowed, else seconds until one would be."""
        with self._lock:
            now = self._clock()
            hits = self._hits[key]
            while hits and now - hits[0] >= DAY_S:
                hits.popleft()
            # hits is in time order, so the last minute is a suffix of it.
            in_last_minute = len(hits) - bisect.bisect_right(hits, now - MINUTE_S)
            if in_last_minute >= self.per_minute:
                return MINUTE_S - (now - hits[len(hits) - in_last_minute])
            if len(hits) >= self.per_day:
                return DAY_S - (now - hits[0])
            hits.append(now)
            return None


def spent_today_usd(path: Path | None = None, *, now: datetime | None = None) -> float:
    """Sum of estimated_cost_usd over today's (UTC) entries in the LLM call log."""
    target = path or log_path()
    if not target.is_file():
        return 0.0
    today = (now or datetime.now(timezone.utc)).date().isoformat()
    total = 0.0
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            # Timestamps are written as UTC ISO strings, so the date is the prefix.
            if str(entry.get("timestamp", ""))[:10] == today:
                total += float(entry.get("estimated_cost_usd", 0.0))
        except (ValueError, TypeError):
            continue  # a partially written last line
    return total


def seconds_until_utc_midnight(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((midnight - now).total_seconds()))
