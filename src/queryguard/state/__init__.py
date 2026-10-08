"""The API's own state, behind one interface: limits, spend, cache, history.

Everything here is state the API keeps *about* requests -- never the queried
database, which the API reaches only as `queryguard_ro`. Two implementations:

    LocalState  (local.py)  one process: in-memory rate limits and reservations,
                            spend from the JSONL call log, history in SQLite.
                            Dev, tests, `make dev`.
    RedisState  (redis.py)  many processes: everything in Redis, with the limits
                            enforced by atomic Lua scripts. Production on
                            serverless, where one visitor's requests can land on
                            different instances and an instance can vanish.

`make_state(settings)` picks RedisState when QUERYGUARD_REDIS_URL is set.

What the interface covers (the stateful items a multi-instance deploy must share):
  1. per-client rate-limit windows         rate_hit
  2. today's spend                          spent_today, and the call guard's after_call
  3. in-flight spend reservations           try_reserve / release
  4. the response cache                     cache_get / cache_put
  5. history, feedback, the client salt     record / history / get_result / exists /
                                            save_feedback / incorrect_feedback / client_key
  6. the daily cap on LLM calls             call_guard().before_call
  7. the LLM call log                       call_guard().after_call
The schema cache (8) and where all of this lives (9) are deployment concerns,
not methods: see docs/DEPLOYMENT.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from queryguard.config import REPO_ROOT

if TYPE_CHECKING:
    from queryguard.api.settings import Settings
    from queryguard.llm.client import CallGuard

FEEDBACK_CANDIDATES = REPO_ROOT / "evals" / "feedback_candidates.yaml"


@dataclass(frozen=True)
class Reservation:
    """A held slot against the daily spend ceiling. Release it exactly once."""

    id: str


class AppState(ABC):
    """See the module docstring. Every method may block; call from a worker thread."""

    # ------------------------------------------------------------- identity

    @abstractmethod
    def client_key(self, ip: str) -> str:
        """A salted hash of the client's IP. The IP itself is never stored."""

    # ------------------------------------------------------------ admission

    @abstractmethod
    def rate_hit(self, client: str) -> float | None:
        """Record a question. None if allowed, else seconds until one would be."""

    @abstractmethod
    def try_reserve(self) -> tuple[Reservation | None, float]:
        """(reservation or None, spent today). Admits only if today's spend plus
        every live reservation, plus this one, stays within the ceiling."""

    @abstractmethod
    def release(self, reservation: Reservation) -> None:
        """Give a reservation back. Releasing twice is harmless."""

    @abstractmethod
    def spent_today(self) -> float:
        """Today's (UTC) LLM spend in USD."""

    @abstractmethod
    def call_guard(self) -> CallGuard:
        """Admits each LLM call (the daily call cap) and records it (spend, log)."""

    # ---------------------------------------------------------------- cache

    @abstractmethod
    def cache_get(self, key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def cache_put(self, key: str, result: dict[str, Any]) -> None: ...

    # -------------------------------------------------------------- history

    @abstractmethod
    def record(
        self, *, query_id: str, client: str, question: str, outcome: str, confidence: float | None,
        cached: bool, cost_usd: float, elapsed_ms: int, result: dict[str, Any], sql_source: str = "model",
    ) -> None: ...

    @abstractmethod
    def history(self, client: str, limit: int) -> list[dict[str, Any]]:
        """Newest first. Each row: query_id, created_at, question, outcome, confidence,
        cached, cost_usd, sql_source, and correct / note / feedback_at (None without feedback)."""

    @abstractmethod
    def get_result(self, query_id: str, client: str) -> dict[str, Any] | None:
        """The stored result, only for the client that asked."""

    @abstractmethod
    def exists(self, query_id: str) -> bool: ...

    # ------------------------------------------------------------- feedback

    @abstractmethod
    def save_feedback(self, query_id: str, correct: bool, note: str | None) -> str:
        """Latest feedback on a query wins. Returns its timestamp."""

    @abstractmethod
    def incorrect_feedback(self) -> list[dict[str, Any]]:
        """Oldest first. Each row: query_id, created_at, question, outcome, confidence,
        result_json (a JSON string), note, feedback_at."""

    def export_feedback_candidates(self, path: Path = FEEDBACK_CANDIDATES) -> tuple[Path, int]:
        """Write answers marked incorrect as golden-set candidates. Returns (path, count).

        Each candidate carries what the pipeline did and blank golden fields:
        a person writes the golden SQL and category, then moves the entry into
        golden.yaml. Nothing here is a golden case until that review.
        """
        import json

        candidates = []
        for row in self.incorrect_feedback():
            result = json.loads(row["result_json"])
            candidates.append({
                "query_id": row["query_id"],
                "asked_at": row["created_at"],
                "question": row["question"],
                "outcome": row["outcome"],
                "confidence": row["confidence"],
                "sql": result.get("sql"),
                "executed_sql": result.get("executed_sql"),
                "feedback_note": row["note"],
                "feedback_at": row["feedback_at"],
                "category": None,
                "golden_sql": None,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Answers users marked incorrect, exported from the API's feedback store.\n"
            "# Review each: write golden_sql and category, then move it into golden.yaml.\n"
        )
        body = yaml.safe_dump({"candidates": candidates}, sort_keys=False, allow_unicode=True, width=100)
        path.write_text(header + body, encoding="utf-8")
        return path, len(candidates)


def make_state(settings: Settings) -> AppState:
    """RedisState when settings.redis_url is set, else LocalState."""
    from queryguard.state.local import LocalState

    return LocalState(settings)
