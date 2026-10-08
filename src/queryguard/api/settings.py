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

Fake mode also points the LLM call log at logs/fake_llm_calls.jsonl (unless
QUERYGUARD_LLM_LOG is set), so the real ledger -- and the spend ceiling read
from it -- never sees fake calls, and lifts the process request cap, which
exists to protect money that fake mode does not spend.

The reserve is about twice the most expensive question in the live eval run
($0.0255), so questions already running cannot carry spend past the ceiling.

Spend is read from the LLM call log (QUERYGUARD_LLM_LOG, default
logs/llm_calls.jsonl), which every API call appends to -- including CLI runs on
the same host, so the ceiling errs towards stopping early.

QUERYGUARD_MAX_REQUESTS (llm/client.py) still applies: it is a process-lifetime
runaway guard, 50 by default, i.e. about 12 questions per server start. A
long-running server should raise it; the daily ceiling is the real control.
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
        )
