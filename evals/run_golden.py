"""Execute every golden_sql and record its row count and result hash.

    uv run python -m evals.run_golden

Fails (exit 1) if any golden query is blocked, errors, hits the row cap, or
returns nothing -- zero rows, or one row of zero/NULL aggregates, which is how
count() and sum() say "nothing matched" -- without `empty_ok` and a reason in
the YAML -- a golden
answer of "nothing" is usually a broken filter, and an eval built on it
rewards models for writing the same broken filter. Also fails if a
non-SQL entry is missing its expected_outcome.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from evals.common import GOLDEN_RESULTS, load_golden, result_hash, run_guarded

EXPECTED_OUTCOMES = {"clarification", "refusal_or_clarification"}


def _matched_nothing(frame) -> bool:
    """Zero rows, or the one-row shape an aggregate over nothing takes (count 0, sum NULL)."""
    if len(frame) == 0:
        return True
    if len(frame) != 1:
        return False
    values = frame.iloc[0].tolist()
    return all(v is None or v != v or (isinstance(v, (int, float, Decimal)) and v == 0) for v in values)


def main() -> int:
    entries = load_golden()
    problems: list[str] = []
    results: dict[str, dict] = {}

    ids = [e["id"] for e in entries]
    for duplicate in [i for i, n in Counter(ids).items() if n > 1]:
        problems.append(f"{duplicate}: duplicate id")

    for entry in entries:
        if "golden_sql" not in entry:
            if entry.get("expected_outcome") not in EXPECTED_OUTCOMES:
                problems.append(f"{entry['id']}: no golden_sql and no valid expected_outcome")
            results[entry["id"]] = {"expected_outcome": entry["expected_outcome"]}
            continue

        run = run_guarded(entry["golden_sql"])
        if not run.ok:
            problems.append(f"{entry['id']}: {run.error}")
            continue
        frame = run.execution.rows
        if run.execution.truncated:
            problems.append(f"{entry['id']}: truncated at the row cap; the full answer was never seen")
            continue
        if _matched_nothing(frame) and not entry.get("empty_ok"):
            problems.append(
                f"{entry['id']}: returned no rows, or one row of zero/NULL aggregates, "
                "and is not marked empty_ok"
            )
        ordered = bool(entry.get("ordered", False))
        results[entry["id"]] = {
            "row_count": len(frame),
            "columns": frame.shape[1],
            "ordered": ordered,
            "result_sha256": result_hash(frame, ordered=ordered),
        }

    by_category = Counter(e["category"] for e in entries)
    print(f"{len(entries)} questions: " + ", ".join(f"{c} {n}" for c, n in by_category.items()))
    for entry in entries:
        r = results.get(entry["id"], {})
        shown = r.get("row_count", r.get("expected_outcome", "FAILED"))
        print(f"  {entry['id']:<10} {shown}")

    if problems:
        print("\nREJECTED:\n  " + "\n  ".join(problems))
        return 1

    GOLDEN_RESULTS.write_text(
        json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "results": results},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {GOLDEN_RESULTS.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
