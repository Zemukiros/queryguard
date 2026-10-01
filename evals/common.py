"""Shared by run_golden.py and mutations.py: load the set, run SQL, hash results.

SQL runs exactly the way the pipeline runs it -- guardrail first, then the
read-only executor -- so a golden query that the guardrail would refuse, or
that hits the row cap, is caught here rather than discovered in an eval run.

The result hash is over a canonical form, not the raw rows: numbers as floats
rounded to 6 decimal places (so Decimal('3.10') and 3.1 hash alike), timestamps
as UTC ISO strings, and rows sorted unless the entry is `ordered`. Column names
are left out, matching how agreement compares results: an alias is not part of
the answer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from queryguard.executor import ExecutionResult, execute
from queryguard.guardrails import check

EVALS_DIR = Path(__file__).resolve().parent

# Eval runs execute hundreds of queries, many deliberately wrong. They get their
# own log so the production execution log stays a record of real traffic.
os.environ.setdefault(
    "QUERYGUARD_EXECUTOR_LOG", str(EVALS_DIR.parent / "logs" / "eval_executions.jsonl")
)
GOLDEN_YAML = EVALS_DIR / "golden.yaml"
GOLDEN_RESULTS = EVALS_DIR / "golden_results.json"
MUTATION_RESULTS = EVALS_DIR / "mutation_results.json"


def load_golden(path: Path = GOLDEN_YAML) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


@dataclass(frozen=True)
class Run:
    """One guarded execution. `error` is set when nothing usable came back."""

    sql: str
    execution: ExecutionResult | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_guarded(sql: str) -> Run:
    guardrail = check(sql)
    if not guardrail.allowed:
        return Run(sql, None, f"blocked by {guardrail.rule}: {guardrail.reason}")
    execution = execute(guardrail.sql_to_execute)
    if not execution.ok:
        return Run(sql, execution, f"{execution.outcome}: {execution.reason or execution.error_message}")
    # Truncation is not an error here: a golden query must not truncate (the
    # runner rejects it), but a mutation that does has plainly changed the answer.
    return Run(sql, execution)


def _canonical_cell(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, Decimal, np.integer, np.floating)):
        return round(float(value), 6)
    if isinstance(value, (datetime, date)):
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return stamp.isoformat()
    return str(value)


def canonical_rows(frame: pd.DataFrame, *, ordered: bool) -> list[list[Any]]:
    rows = [[_canonical_cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    if not ordered:
        rows.sort(key=lambda r: json.dumps(r, default=str))
    return rows


def result_hash(frame: pd.DataFrame, *, ordered: bool) -> str:
    payload = json.dumps(
        {"columns": frame.shape[1], "rows": canonical_rows(frame, ordered=ordered)}, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
