"""Fit the confidence score to a labelled eval run, offline.

    uv run python -m evals.calibrate --run-id live-2026-10-01

No API calls and no database: everything comes from
evals/results/<run_id>.corrected.jsonl.

Fitting set: every row with a feature vector, minus generated items whose
golden entry is ambiguous or unanswerable. Those are labelled on whether the
pipeline declined (or answered with self-confidence < 0.5), not on whether the
SQL was right, so their label does not mean "this answer is correct". They
stay in the accuracy report.

Validation is grouped 5-fold cross-validation by golden question id: a
question's golden SQL, its mutations and its generated answer always land in
the same fold, so the model never scores a mutation of a question it saw the
golden version of. Every reported metric for a fitted model is out-of-fold.
The v0 hand-set weights were never fitted, so they are scored on the same rows
as they stand.

Two variants are fitted, with and without self-confidence. On the mutation and
golden rows self-confidence is an injected constant (INJECTED_SELF_CONFIDENCE);
only generated rows carry a real value, and those are nearly all correct, so
"self-confidence is not 0.9" can stand in for "this is a generated row".
Grouping by question does not block that shortcut -- it works across
populations, not questions -- so the comparison is tilted toward the variant
with self-confidence. It is chosen only if it beats the variant without by more
than MIN_BRIER_GAIN; otherwise the variant without wins. The chosen variant is
refitted on the whole set and written to
src/queryguard/validation/calibration.json, which confidence.py loads at import.

A feature that is zero on every fitting row (it never fired in the run) has no
evidence either way, and the fit gives it weight 0. That would silently switch
the signal off at runtime, so such a feature keeps its v0 weight instead and is
listed under `unobserved_features`. This changes no fitted coefficient and no
out-of-fold score: the feature is zero on every row they are computed from.

Writes:
  src/queryguard/validation/calibration.json   weights + intercept (runtime)
  evals/results/<run_id>.calibration.json       every metric below, plus OOF scores
  docs/calibration.png                          reliability diagram
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from evals.run_eval import GENERATED, GOLDEN, INJECTED_SELF_CONFIDENCE, MUTATION, RESULTS_DIR
from queryguard.config import REPO_ROOT
from queryguard.validation.confidence import CALIBRATION_PATH, V0_WEIGHTS, Features, encode, score

FEATURES = tuple(name for name in V0_WEIGHTS if name != "bias")
N_SPLITS = 5
N_BINS = 10
# Fixed before looking at any result; with ~180 rows there is no data to spare
# for an inner search, and tuning C on the outer folds would leak.
C = 1.0
THRESHOLD = 0.5
# Smallest out-of-fold Brier improvement that justifies keeping self-confidence
# (see the docstring). Set after seeing the live-2026-10-01 margin (0.0001), so
# it is a judgement, not a pre-registered rule: about a tenth of the Brier score.
MIN_BRIER_GAIN = 0.005
NOT_FITTED_CATEGORIES = ("ambiguous", "unanswerable")
DETECTORS = ("alignment", "agreement", "sanity")
PLOT_PATH = REPO_ROOT / "docs" / "calibration.png"


# -------------------------------------------------------------------- data


def load_rows(run_id: str) -> list[dict]:
    path = RESULTS_DIR / f"{run_id}.corrected.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fitting_rows(rows: list[dict]) -> list[dict]:
    return [
        r for r in rows
        if r["features"] is not None
        and not (r["population"] == GENERATED and r["category"] in NOT_FITTED_CATEGORIES)
    ]


def design_matrix(rows: list[dict], features: tuple[str, ...]) -> np.ndarray:
    encoded = [encode(Features(**r["features"])) for r in rows]
    return np.array([[e[name] for name in features] for e in encoded])


# ----------------------------------------------------------------- metrics


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: equal-width bins, weighted by bin size."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.sum() * abs(y[mask].mean() - p[mask].mean())
    return total / len(y)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            out.append({"bin": b, "n": int(mask.sum()), "mean_score": float(p[mask].mean()),
                        "observed": float(y[mask].mean())})
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    wrong, right = y == 0, y == 1
    return {
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "wrong_below_threshold": float((p[wrong] < THRESHOLD).mean()),
        "correct_false_flag_rate": float((p[right] < THRESHOLD).mean()),
        "n": int(len(y)), "n_correct": int(right.sum()), "n_wrong": int(wrong.sum()),
    }


# ---------------------------------------------------------------- fitting


def model() -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=10_000)


def out_of_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[float]]:
    oof = np.empty(len(y))
    fold_brier = []
    for train, test in GroupKFold(n_splits=N_SPLITS).split(X, y, groups):
        fitted = model().fit(X[train], y[train])
        oof[test] = fitted.predict_proba(X[test])[:, 1]
        fold_brier.append(float(brier_score_loss(y[test], oof[test])))
    return oof, fold_brier


def fit_variant(rows: list[dict], y: np.ndarray, groups: np.ndarray, features: tuple[str, ...]) -> dict:
    X = design_matrix(rows, features)
    oof, fold_brier = out_of_fold(X, y, groups)
    final = model().fit(X, y)
    weights = {name: 0.0 for name in FEATURES}
    weights.update({name: float(w) for name, w in zip(features, final.coef_[0])})
    unobserved = [name for name, column in zip(features, X.T) if not column.any()]
    weights.update({name: V0_WEIGHTS[name] for name in unobserved})
    return {
        "unobserved_features": unobserved,
        "features": list(features),
        "oof": oof,
        "fold_brier": fold_brier,
        "metrics": metrics(y, oof),
        "intercept": float(final.intercept_[0]),
        "weights": weights,
    }


# ---------------------------------------------------------------- reports


def detector_table(fit: list[dict], calibrated: np.ndarray, v0: np.ndarray) -> list[dict]:
    """Per mutation type: share caught by each detector alone, and by score < 0.5."""
    index = {r["id"]: i for i, r in enumerate(fit)}
    mutations: dict[str, list[dict]] = defaultdict(list)
    for r in fit:
        if r["population"] == MUTATION:
            mutations[r["mutation"]].append(r)
    groups = sorted(mutations.items(), key=lambda kv: -len(kv[1]))
    groups.append(("all mutations", [r for r in fit if r["population"] == MUTATION]))
    groups.append(("generated, wrong", [r for r in fit if r["population"] == GENERATED and r["label"] == "wrong"]))
    groups.append(("correct answers (false flags)", [r for r in fit if r["label"] == "correct"]))

    table = []
    for name, members in groups:
        if not members:  # e.g. no wrong generated answers left after a fix
            continue
        row = {"group": name, "n": len(members)}
        for d in DETECTORS:
            row[d] = sum(r["detectors"][d]["flagged"] for r in members) / len(members)
        row["any_detector"] = sum(any(r["detectors"][d]["flagged"] for d in DETECTORS) for r in members) / len(members)
        row["calibrated_below_0.5"] = float(np.mean([calibrated[index[r["id"]]] < THRESHOLD for r in members]))
        row["v0_below_0.5"] = float(np.mean([v0[index[r["id"]]] < THRESHOLD for r in members]))
        table.append(row)
    return table


def accuracy_report(rows: list[dict]) -> dict:
    generated = [r for r in rows if r["population"] == GENERATED]
    per_category = {}
    for cat in dict.fromkeys(r["category"] for r in generated):
        members = [r for r in generated if r["category"] == cat]
        per_category[cat] = {
            "n": len(members),
            "correct": sum(r["label"] == "correct" for r in members),
            "outcomes": dict(Counter(r["outcome"] for r in members)),
            "wrong_ids": [r["golden_id"] for r in members if r["label"] != "correct"],
        }
    declines = [
        {"id": r["golden_id"], "category": r["category"], "outcome": r["outcome"],
         "label": r["label"], "reason": r["label_reason"]}
        for r in generated if r["category"] in NOT_FITTED_CATEGORIES
    ]
    return {"per_category": per_category, "declines": declines}


def cost_report(run_id: str, rows: list[dict]) -> dict:
    ledger_path = RESULTS_DIR / f"{run_id}.llm_calls.jsonl"
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_pop = {}
    for pop in (GENERATED, MUTATION, GOLDEN):
        members = [r for r in rows if r["population"] == pop]
        by_pop[pop] = {
            "items": len(members),
            "calls": sum(r["n_calls"] for r in members),
            "cost_usd": sum(r["cost_usd"] for r in members),
            "median_cost_usd": statistics.median(r["cost_usd"] for r in members),
            "median_latency_ms": statistics.median(r["latency_ms"] for r in members),
            "p90_latency_ms": float(np.percentile([r["latency_ms"] for r in members], 90)),
        }
    return {
        "ledger_calls": len(ledger),
        "ledger_cost_usd": sum(r["estimated_cost_usd"] for r in ledger),
        "row_cost_usd": sum(r["cost_usd"] for r in rows),
        "median_latency_ms_all": statistics.median(r["latency_ms"] for r in rows),
        "by_population": by_pop,
    }


def plot(y: np.ndarray, v0: np.ndarray, calibrated: np.ndarray, label: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [("v0 hand-set", v0, "#2a78d6", "o"), (label, calibrated, "#eb6834", "s")]

    fig, (ax, hist) = plt.subplots(
        2, 1, figsize=(7, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        facecolor=surface,
    )
    for a in (ax, hist):
        a.set_facecolor(surface)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(muted)
        a.tick_params(colors=muted, labelsize=9)
        a.grid(color=grid, linewidth=0.8)
        a.set_axisbelow(True)

    ax.plot([0, 1], [0, 1], linestyle="--", color=muted, linewidth=1.2, label="perfect calibration")
    edges = np.linspace(0, 1, N_BINS + 1)
    width = 1 / N_BINS / 2 - 0.006
    for i, (name, p, color, marker) in enumerate(series):
        pts = reliability(y, p)
        xs, ys = [b["mean_score"] for b in pts], [b["observed"] for b in pts]
        m = metrics(y, p)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=8,
                markeredgecolor=surface, markeredgewidth=1.5,
                label=f"{name}: Brier {m['brier']:.3f}, ECE {m['ece']:.3f}")
        counts, _ = np.histogram(p, bins=edges)
        hist.bar(edges[:-1] + 0.003 + i * (width + 0.002), counts, width=width, align="edge",
                 color=color, linewidth=0)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("observed share correct", color=ink, fontsize=10)
    ax.set_title(f"Confidence reliability, {N_BINS} bins, n={len(y)}", color=ink, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink, loc="upper left")
    hist.set_ylabel("answers", color=ink, fontsize=10)
    hist.set_xlabel("confidence score", color=ink, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=surface)
    plt.close(fig)


# ------------------------------------------------------------------- main


def _md_metrics(results: dict[str, dict]) -> str:
    lines = ["| scorer | Brier ↓ | ECE ↓ | AUROC ↑ | wrong < 0.5 ↑ | correct < 0.5 (false flags) ↓ |",
             "|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['brier']:.3f} | {m['ece']:.3f} | {m['auroc']:.3f} | "
                     f"{m['wrong_below_threshold']:.1%} | {m['correct_false_flag_rate']:.1%} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.calibrate")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    rows = load_rows(args.run_id)
    fit = fitting_rows(rows)
    y = np.array([r["label"] == "correct" for r in fit], dtype=int)
    groups = np.array([r["golden_id"] for r in fit])
    v0 = np.array([score(Features(**r["features"]), V0_WEIGHTS)[0] for r in fit])

    variants = {
        "with_self_confidence": fit_variant(fit, y, groups, FEATURES),
        "without_self_confidence": fit_variant(fit, y, groups, tuple(f for f in FEATURES if f != "self_confidence")),
    }
    gain = variants["without_self_confidence"]["metrics"]["brier"] - variants["with_self_confidence"]["metrics"]["brier"]
    chosen_name = "with_self_confidence" if gain > MIN_BRIER_GAIN else "without_self_confidence"
    chosen = variants[chosen_name]
    version = f"calibrated-{args.run_id}"

    calibration = {
        "version": version,
        "fitted_on": f"evals/results/{args.run_id}.corrected.jsonl",
        "fitted_by": "evals/calibrate.py",
        "model": f"sklearn LogisticRegression, L2, C={C}",
        "variant": chosen_name,
        "unobserved_features": chosen["unobserved_features"],
        "n_rows": int(len(y)),
        "n_correct": int(y.sum()),
        "intercept": chosen["intercept"],
        "weights": chosen["weights"],
        "oof_metrics": chosen["metrics"],
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    detectors = detector_table(fit, chosen["oof"], v0)
    excluded = [r for r in rows if r["features"] is not None and r not in fit]
    report = {
        "run_id": args.run_id,
        "chosen_variant": chosen_name,
        "brier_gain_with_self_confidence": gain,
        "min_brier_gain": MIN_BRIER_GAIN,
        "cv": {"splitter": "GroupKFold", "n_splits": N_SPLITS, "groups": "golden_id",
               "n_groups": int(len(set(groups)))},
        "self_confidence": {
            "injected_value": INJECTED_SELF_CONFIDENCE,
            "by_population": {
                pop: dict(Counter(r["features"]["self_confidence"] for r in fit if r["population"] == pop))
                for pop in (GENERATED, MUTATION, GOLDEN)
            },
        },
        "metrics": {"v0": metrics(y, v0), **{k: v["metrics"] for k, v in variants.items()}},
        "fold_brier": {k: v["fold_brier"] for k, v in variants.items()},
        "weights": {k: {"intercept": v["intercept"], **v["weights"]} for k, v in variants.items()},
        "reliability": {"v0": reliability(y, v0), chosen_name: reliability(y, chosen["oof"])},
        "detectors": detectors,
        "accuracy": accuracy_report(rows),
        "cost": cost_report(args.run_id, rows),
        "excluded_from_fit": [
            {"id": r["id"], "label": r["label"], "reason": r["label_reason"],
             "v0": r["confidence"], "calibrated": score(Features(**r["features"]),
                                                        {"bias": chosen["intercept"], **chosen["weights"]})[0]}
            for r in excluded
        ],
        "oof_scores": {r["id"]: {"label": r["label"], "v0": float(v0[i]),
                                 **{k: float(v["oof"][i]) for k, v in variants.items()}}
                       for i, r in enumerate(fit)},
    }
    out = RESULTS_DIR / f"{args.run_id}.calibration.json"
    out.write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    plot(y, v0, chosen["oof"], "calibrated, out-of-fold", PLOT_PATH)

    print(f"fit rows {len(y)} ({y.sum()} correct, {len(y) - y.sum()} wrong), {len(set(groups))} groups\n")
    print(_md_metrics({"v0 hand-set": report["metrics"]["v0"],
                       **{f"calibrated, {k.replace('_', ' ')} (OOF)": v["metrics"] for k, v in variants.items()}}))
    print(f"\nfold Brier: { {k: [round(b, 3) for b in v['fold_brier']] for k, v in variants.items()} }")
    print(f"Brier gain from self-confidence {gain:.5f} (needs > {MIN_BRIER_GAIN}); chosen: {chosen_name}\n")
    for k, v in variants.items():
        print(k, f"intercept {v['intercept']:+.3f}", {n: round(w, 3) for n, w in v["weights"].items()})
    print("\n| group | n | back-translation | agreement | sanity | any | calibrated < 0.5 | v0 < 0.5 |\n|---|---|---|---|---|---|---|---|")
    for d in detectors:
        print(f"| {d['group']} | {d['n']} | {d['alignment']:.0%} | {d['agreement']:.0%} | {d['sanity']:.0%} | "
              f"{d['any_detector']:.0%} | {d['calibrated_below_0.5']:.0%} | {d['v0_below_0.5']:.0%} |")
    print("\nexcluded from fit:", json.dumps(report["excluded_from_fit"], indent=1))
    print("\naccuracy:", json.dumps(report["accuracy"], indent=1))
    print("\ncost:", json.dumps(report["cost"], indent=1))
    print(f"\nwrote {CALIBRATION_PATH.relative_to(REPO_ROOT)}, {out.relative_to(REPO_ROOT)}, {PLOT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
