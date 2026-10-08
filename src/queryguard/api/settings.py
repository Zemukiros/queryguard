"""API configuration, read from the environment once at startup.

Every public-demo cost control is here, so a deployment changes behaviour by
setting variables, never by editing code:

    QUERYGUARD_RATE_PER_MINUTE       per-client questions per rolling minute (10)
    QUERYGUARD_RATE_PER_DAY          per-client questions per rolling 24 h (50)
    QUERYGUARD_DAILY_SPEND_USD       global ceiling on today's (UTC) LLM spend (1.00)
    QUERYGUARD_QUESTION_RESERVE_USD  held per in-flight question against the ceiling (0.05)
    QUERYGUARD_FRONTEND_ORIGIN       comma-separated CORS origins (http://localhost:3000)
    QUERYGUARD_APP_DB                SQLite file for history, feedback and cache (data/app.db)
    QUERYGUARD_TRUST_PROXY           1 = client IP from the last X-Forwarded-For hop (0)
    QUERYGUARD_FAKE_LLM              1 = answer from llm/fake.py, never the API; $0 (0)
    QUERYGUARD_LIVE                  0 = kill switch: every question runs in demo mode (1)
    QUERYGUARD_REDIS_URL             app state in Redis (queryguard.state.redis), shared by every
                                     instance; unset = in-process state (queryguard.state.local)
    QUERYGUARD_STATE_PREFIX          prefix for every Redis key, so deployments can share one
                                     database without seeing each other's state (qg:)
    QUERYGUARD_DAILY_CALL_CAP        LLM calls per UTC day across all instances (Redis state) (200)
    QUERYGUARD_RESERVATION_TTL_S     a spend reservation expires after this, so one held by a
                                     crashed instance frees itself (300, the platform time limit)

    QUERYGUARD_FAKE_LLM_LOG          where simulated calls are logged with local state
                                     (logs/fake_llm_calls.jsonl); with Redis state they are not logged

Modes. A question runs live (the real model) unless one of these sends it to
demo mode (llm/fake.py: $0, answers the eval set's questions), in this order:
QUERYGUARD_FAKE_LLM=1 (a demo-only deployment), QUERYGUARD_LIVE=0 (the kill
switch), today's spend ceiling, today's call cap. The answer and /healthz say
which mode and why. Simulated calls never touch the real ledger or the caps.

The reserve is about twice the most expensive question in the live eval run
($0.0255), so questions already running cannot carry spend past the ceiling.

With local state, spend is read from the LLM call log (QUERYGUARD_LLM_LOG,
default logs/llm_calls.jsonl) -- including CLI runs on the same host, so the
ceiling errs towards stopping early -- and the daily call cap is counted in
memory. With Redis state both are shared by every instance. The API's own
calls are capped by QUERYGUARD_DAILY_CALL_CAP; QUERYGUARD_MAX_REQUESTS
(llm/client.py) now guards only the CLI and the evals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from queryguard.config import REPO_ROOT, load_env


def _env(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    rate_per_minute: int = 10
    rate_per_day: int = 50
    daily_spend_usd: float = 1.00
    question_reserve_usd: float = 0.05
    frontend_origins: tuple[str, ...] = ("http://localhost:3000",)
    db_path: Path = field(default_factory=lambda: REPO_ROOT / "data" / "app.db")
    trust_proxy: bool = False
    fake_llm: bool = False
    live: bool = True
    redis_url: str | None = None
    state_prefix: str = "qg:"
    daily_call_cap: int = 200
    reservation_ttl_s: float = 300.0

    @classmethod
    def from_env(cls) -> Settings:
        load_env()
        return cls(
            rate_per_minute=int(_env("QUERYGUARD_RATE_PER_MINUTE", "10")),
            rate_per_day=int(_env("QUERYGUARD_RATE_PER_DAY", "50")),
            daily_spend_usd=float(_env("QUERYGUARD_DAILY_SPEND_USD", "1.00")),
            question_reserve_usd=float(_env("QUERYGUARD_QUESTION_RESERVE_USD", "0.05")),
            frontend_origins=tuple(
                o.strip() for o in _env("QUERYGUARD_FRONTEND_ORIGIN", "http://localhost:3000").split(",")
                if o.strip()
            ),
            # A relative path is relative to the repo, not to wherever the server was started.
            db_path=REPO_ROOT / _env("QUERYGUARD_APP_DB", "data/app.db"),
            trust_proxy=_env("QUERYGUARD_TRUST_PROXY", "0") == "1",
            fake_llm=_env("QUERYGUARD_FAKE_LLM", "0") == "1",
            live=_env("QUERYGUARD_LIVE", "1") != "0",
            redis_url=_env("QUERYGUARD_REDIS_URL", "") or None,
            state_prefix=_env("QUERYGUARD_STATE_PREFIX", "qg:"),
            daily_call_cap=int(_env("QUERYGUARD_DAILY_CALL_CAP", "200")),
            reservation_ttl_s=float(_env("QUERYGUARD_RESERVATION_TTL_S", "300")),
        )
