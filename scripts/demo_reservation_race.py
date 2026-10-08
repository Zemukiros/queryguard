"""Why the spend reservation is a Lua script: run the race both ways.

    docker compose up -d redis && uv run python scripts/demo_reservation_race.py

200 threads each try to reserve $0.05 against a $1.00 ceiling, so exactly 20
should get through. The naive version reads (GET, ZCARD) and then writes
(ZADD) as separate commands; the gap between them is any network round trip,
and every thread that reads before the others write sees room that is not
there. The Lua version (state/lua/reserve.lua) does the read and the write as
one step Redis never interleaves.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

from queryguard.api.settings import Settings
from queryguard.state.redis import RedisState, connect

URL = os.getenv("QUERYGUARD_TEST_REDIS_URL", "redis://localhost:6379/15")
THREADS, CEILING, RESERVE = 200, 1.00, 0.05


def race(reserve) -> int:
    barrier = threading.Barrier(THREADS)
    admitted: list[int] = []

    def worker() -> None:
        barrier.wait()
        if reserve():
            admitted.append(1)

    threads = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return len(admitted)


def main() -> int:
    r = connect(URL)
    prefix = f"race:{uuid.uuid4().hex[:6]}:"

    def naive() -> bool:
        spent = float(r.get(prefix + "spend") or 0)
        held = r.zcard(prefix + "naive")
        time.sleep(0.001)  # stands in for the round trip between the read and the write
        if spent + (held + 1) * RESERVE <= CEILING:
            r.zadd(prefix + "naive", {uuid.uuid4().hex: 1})
            return True
        return False

    state = RedisState(Settings(db_path=Path("/dev/null"), daily_spend_usd=CEILING, question_reserve_usd=RESERVE,
                                state_prefix=prefix), client=r)
    try:
        print(f"ceiling allows {int(round(CEILING / RESERVE))} of {THREADS}")
        print(f"naive GET-then-ZADD admitted {race(naive)}")
        print(f"Lua reserve.lua admitted     {race(lambda: state.try_reserve()[0] is not None)}")
    finally:
        keys = list(r.scan_iter(prefix + "*"))
        if keys:
            r.delete(*keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())
