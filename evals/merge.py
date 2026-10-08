"""Merge a targeted re-run into a full run, for calibration, offline.

    uv run python -m evals.merge --base live-2026-10-01 --rerun refund-fix-2026-10-08 --run-id merged-2026-10-08

A fix is re-measured on the items it is about, not on all 194: re-running the
lot costs ~$2 and changes nothing the fix touched. The merged set is the base
run's corrected rows with every re-run item replaced by its newer row, each
tagged with `source_run`. Run evals.recompute on both inputs first, so every
row carries the current offline-derivable labels and sanity flags.

Writes evals/results/<run_id>.corrected.jsonl and <run_id>.llm_calls.jsonl, the
two ledgers concatenated: every call that produced a row in either input,
including the base calls whose rows were superseded. No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys

from evals.run_eval import RESULTS_DIR


def _read(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge(base: list[dict], rerun: list[dict], base_id: str, rerun_id: str) -> list[dict]:
    newer = {r["id"]: {**r, "source_run": rerun_id} for r in rerun}
    unknown = sorted(set(newer) - {r["id"] for r in base})
    if unknown:
        raise SystemExit(f"re-run items not in {base_id}: {', '.join(unknown)}")
    return [newer.get(r["id"], {**r, "source_run": base_id}) for r in base]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.merge")
    parser.add_argument("--base", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    base = _read(RESULTS_DIR / f"{args.base}.corrected.jsonl")
    rerun = _read(RESULTS_DIR / f"{args.rerun}.corrected.jsonl")
    merged = merge(base, rerun, args.base, args.rerun)

    out = RESULTS_DIR / f"{args.run_id}.corrected.jsonl"
    out.write_text("".join(json.dumps(r, default=str, sort_keys=True) + "\n" for r in merged), encoding="utf-8")
    ledger = RESULTS_DIR / f"{args.run_id}.llm_calls.jsonl"
    ledger.write_text("".join(
        (RESULTS_DIR / f"{run}.llm_calls.jsonl").read_text(encoding="utf-8") for run in (args.base, args.rerun)
    ), encoding="utf-8")
    print(f"wrote {out.name} ({len(merged)} rows, {len(rerun)} from {args.rerun}) and {ledger.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
