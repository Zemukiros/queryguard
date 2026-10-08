"""Re-derive agreement, labels and v0 confidence for an eval run, offline.

    uv run python -m evals.recompute --run-id live-2026-10-01

When a comparison rule changes, the API calls do not need repeating: every row
logs the first SQL and (where agreement ran) the second, so both are
re-executed against the database and compared under the current rules. Free --
no API calls.

Reads evals/results/<run_id>.jsonl, which is never modified, and writes
evals/results/<run_id>.corrected.jsonl. A corrected row keeps its original
values under `corrections` for every field that changed.

What is recomputed:
- agreement outcome and explanation, via evaluate_second_sql (same guardrail,
  executor and compare_results as the live pipeline);
- the feature vector's `agreement` field, and the v0 confidence from it;
- labels of answerable generated items, via compare_to_golden.
Alignment, sanity flags and everything else are left as they were: their
rules did not change, and alignment cannot be recomputed without the API.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from evals.common import run_guarded
from evals.run_eval import GENERATED, GOLDEN, MUTATION, RESULTS_DIR, build_items, compare_to_golden, golden_frames
from queryguard.validation.agreement import AGREE, evaluate_second_sql
from queryguard.validation.confidence import V0_WEIGHTS, Features, score

DETECTORS = ("sanity", "alignment", "agreement")


def recompute_row(row: dict, item, frames: dict) -> dict:
    row = json.loads(json.dumps(row))  # deep copy
    corrections: dict = {}
    agreement = (row.get("detectors") or {}).get("agreement")
    first = None

    if agreement and agreement.get("outcome") is not None and agreement.get("second_sql"):
        first = run_guarded(row["sql"])
        if first.execution is not None and first.execution.ok:
            new = evaluate_second_sql(row["question"], agreement["second_sql"], first.execution)
            if (new.outcome, new.explanation) != (agreement["outcome"], agreement["explanation"]):
                corrections["agreement"] = {"outcome": agreement["outcome"], "explanation": agreement["explanation"]}
                agreement.update(outcome=new.outcome, explanation=new.explanation, flagged=new.outcome != AGREE)
                row["features"]["agreement"] = new.outcome

    if row["population"] == GENERATED and item.expected_outcome is None and row["outcome"] == "executed":
        first = first or run_guarded(row["sql"])
        verdict, explanation = compare_to_golden(item, frames[item.golden_id], first.execution.rows)
        label = "correct" if verdict == AGREE else "wrong"
        reason = f"{verdict}: {explanation}"
        if label != row["label"]:
            corrections["label"] = {"label": row["label"], "label_reason": row["label_reason"]}
        row["label"], row["label_reason"] = label, reason

    if row.get("features"):
        confidence, breakdown = score(Features(**row["features"]), V0_WEIGHTS)
        if abs(confidence - row["confidence"]) > 1e-12:
            corrections["confidence"] = row["confidence"]
        row["confidence"], row["confidence_breakdown"] = confidence, breakdown

    row["corrections"] = corrections
    return row


def _auc(pos: list[float], neg: list[float]) -> float:
    pairs = [(p > n) + 0.5 * (p == n) for p in pos for n in neg]
    return sum(pairs) / len(pairs) if pairs else float("nan")


def summarise(raw: list[dict], corrected: list[dict]) -> None:
    def golden_false_flags(rows: list[dict], detector: str) -> int:
        return sum(r["detectors"][detector]["flagged"] for r in rows if r["population"] == GOLDEN)

    print("False flags on the 40 golden SQLs (before -> after):")
    for d in DETECTORS:
        print(f"  {d:<10} {golden_false_flags(raw, d):>2} -> {golden_false_flags(corrected, d):>2}")
    any_before = sum(any(r["detectors"][d]["flagged"] for d in DETECTORS) for r in raw if r["population"] == GOLDEN)
    any_after = sum(any(r["detectors"][d]["flagged"] for d in DETECTORS) for r in corrected if r["population"] == GOLDEN)
    print(f"  {'any':<10} {any_before:>2} -> {any_after:>2}")

    changed = Counter(k for r in corrected for k in r["corrections"])
    print(f"\nrows changed: {sum(bool(r['corrections']) for r in corrected)} of {len(corrected)} ({dict(changed)})")
    for r in corrected:
        if "label" in r["corrections"]:
            print(f"  relabelled {r['id']}: {r['corrections']['label']['label']} -> {r['label']}")

    print(f"\n{'population':<11}{'items':>6}{'correct':>9}{'wrong':>7}{'calls':>7}{'cost $':>9}")
    for pop in (GENERATED, MUTATION, GOLDEN):
        group = [r for r in corrected if r["population"] == pop]
        labels = Counter(r["label"] for r in group)
        print(f"{pop:<11}{len(group):>6}{labels['correct']:>9}{labels['wrong']:>7}"
              f"{sum(r['n_calls'] for r in group):>7}{sum(r['cost_usd'] for r in group):>9.4f}")

    scored = [r for r in corrected if r["confidence"] is not None]
    for name, rows in (("all scored", scored),
                       ("mutation vs golden", [r for r in scored if r["population"] != GENERATED]),
                       ("generated", [r for r in scored if r["population"] == GENERATED])):
        pos = [r["confidence"] for r in rows if r["label"] == "correct"]
        neg = [r["confidence"] for r in rows if r["label"] == "wrong"]
        print(f"v0 AUC {name:<19} {_auc(pos, neg):.3f}  (correct {len(pos)}, wrong {len(neg)})")

    wrong = [r for r in scored if r["label"] == "wrong"]
    right = [r for r in scored if r["label"] == "correct"]
    print("\ndetector     catches wrong     false-flags correct")
    for d in (*DETECTORS, "any"):
        flag = (lambda r: any(r["detectors"][x]["flagged"] for x in DETECTORS)) if d == "any" else (lambda r, d=d: r["detectors"][d]["flagged"])
        fw, fc = sum(map(flag, wrong)), sum(map(flag, right))
        print(f"  {d:<10} {fw:>3}/{len(wrong)} ({fw / len(wrong):.0%})    {fc:>3}/{len(right)} ({fc / len(right):.0%})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.recompute")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    source = RESULTS_DIR / f"{args.run_id}.jsonl"
    target = RESULTS_DIR / f"{args.run_id}.corrected.jsonl"
    raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = {i.id: i for i in build_items()}
    frames = golden_frames()

    corrected = [recompute_row(row, items[row["id"]], frames) for row in raw]
    target.write_text("".join(json.dumps(r, default=str, sort_keys=True) + "\n" for r in corrected), encoding="utf-8")
    print(f"wrote {target.relative_to(RESULTS_DIR.parent.parent)} ({len(corrected)} rows)\n")
    summarise(raw, corrected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
