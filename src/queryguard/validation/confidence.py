"""One number for "how much should I trust this answer", and the data to fit it.

Each run produces a feature vector from every signal the pipeline has: the
model's own confidence, how well the back-translated question aligns with the
asked one, the sanity flags, whether a second query agreed, whether the
guardrail had to rewrite the SQL, and how many rows came back.

The score is a logistic model -- a weighted sum of the encoded features passed
through a sigmoid. Its weights come from `calibration.json` beside this module,
a logistic regression fitted to a labelled eval run by `evals/calibrate.py`
(see docs/EVAL_RESULTS.md for how it was validated). The file is loaded once at
import; if it is missing, the hand-set V0_WEIGHTS below are used instead and
SCORER_VERSION says so. A file that exists but does not match the feature set
is an error, not a silent fallback.

V0_WEIGHTS were chosen to rank obviously good and obviously bad runs sensibly,
not fitted to anything. They are kept as the fallback and as the baseline the
calibration is measured against.

Every run is appended to `logs/confidence_features.jsonl`, with the raw
features, the encoded vector the weights apply to, the score, and a `label`
left null for a human (or an eval suite) to fill in. That file is the training
set for the next calibration. A run that is not logged cannot be learned from,
so logging is not optional and does not depend on whether validation succeeded.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from queryguard.config import REPO_ROOT
from queryguard.llm.client import append_log

V0_VERSION = "v0-hand-set"
CALIBRATION_PATH = Path(__file__).with_name("calibration.json")

AGREEMENT_NOT_RUN = "not_run"

# Row-count buckets. Edges are about what the count means, not its size: none,
# a scalar answer, a short list, a long one, and one that hit the cap.
ROW_BUCKETS = ("0", "1", "2-10", "11-100", "101-999", "capped")

# Hand-set fallback, logit-space; see the module docstring. Each encoded feature is multiplied by its weight and summed with the bias.
V0_WEIGHTS: dict[str, float] = {
    "bias": -1.0,
    # The model's 0-1 self-estimate. Informative, but models are overconfident,
    # so it is weighted below the external checks.
    "self_confidence": 1.5,
    # alignment - 0.5: a perfect match adds 1.5, a 0.0 match subtracts 1.5.
    "alignment_centered": 3.0,
    # Back-translation failed: no evidence either way, but a small penalty for
    # an unchecked answer.
    "alignment_missing": -0.3,
    "discrepancy_count": -0.5,
    "sanity_fail": -2.0,
    "sanity_warn": -0.7,
    "sanity_info": -0.1,
    "agreement_agree": 1.0,
    "agreement_disagree": -2.0,
    "agreement_incomparable": -0.3,
    # A LIMIT added on the way through is routine; it is recorded because the
    # calibration may well find it is not.
    "guardrail_rewrote": -0.1,
    "rows_empty": -0.5,
    "rows_capped": -0.3,
}

_MAX_DISCREPANCIES = 4  # beyond this the judge is restating, not finding more


def load_weights(path: Path = CALIBRATION_PATH) -> tuple[dict[str, float], str]:
    """(weights incl. bias, scorer version) from a calibration file, or v0 if there is none."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(V0_WEIGHTS), V0_VERSION
    weights = {"bias": float(data["intercept"]), **{k: float(v) for k, v in data["weights"].items()}}
    if set(weights) != set(V0_WEIGHTS):
        raise ValueError(f"{path}: weights {sorted(weights)} do not match features {sorted(V0_WEIGHTS)}")
    return weights, data["version"]


WEIGHTS, SCORER_VERSION = load_weights()


@dataclass(frozen=True)
class Features:
    """Raw per-run signals, before encoding."""

    executed: bool
    self_confidence: float
    alignment: float | None
    discrepancy_count: int
    sanity_fail: int
    sanity_warn: int
    sanity_info: int
    agreement: str  # agree / disagree / incomparable / not_run
    guardrail_rewrote: bool
    row_count_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def row_count_bucket(row_count: int, truncated: bool) -> str:
    if truncated:
        return "capped"
    if row_count <= 1:
        return str(max(row_count, 0))
    if row_count <= 10:
        return "2-10"
    if row_count <= 100:
        return "11-100"
    return "101-999"


def build_features(
    *,
    executed: bool,
    self_confidence: float,
    alignment: float | None,
    discrepancies: tuple[str, ...] | list[str],
    sanity: tuple[Any, ...] | list[Any],
    agreement: str | None,
    guardrail_rewrote: bool,
    row_count: int,
    truncated: bool,
) -> Features:
    severities = [flag.severity for flag in sanity]
    return Features(
        executed=executed,
        self_confidence=float(self_confidence),
        alignment=None if alignment is None else float(alignment),
        discrepancy_count=len(discrepancies),
        sanity_fail=severities.count("fail"),
        sanity_warn=severities.count("warn"),
        sanity_info=severities.count("info"),
        agreement=agreement or AGREEMENT_NOT_RUN,
        guardrail_rewrote=guardrail_rewrote,
        row_count_bucket=row_count_bucket(row_count, truncated),
    )


def encode(features: Features) -> dict[str, float]:
    """The numeric vector the weights apply to. Keys match V0_WEIGHTS minus bias."""
    aligned = features.alignment is not None
    return {
        "self_confidence": features.self_confidence,
        "alignment_centered": (features.alignment - 0.5) if aligned else 0.0,
        "alignment_missing": 0.0 if aligned else 1.0,
        "discrepancy_count": float(min(features.discrepancy_count, _MAX_DISCREPANCIES)),
        "sanity_fail": float(features.sanity_fail),
        "sanity_warn": float(features.sanity_warn),
        "sanity_info": float(features.sanity_info),
        "agreement_agree": float(features.agreement == "agree"),
        "agreement_disagree": float(features.agreement == "disagree"),
        "agreement_incomparable": float(features.agreement == "incomparable"),
        "guardrail_rewrote": float(features.guardrail_rewrote),
        "rows_empty": float(features.row_count_bucket == "0"),
        "rows_capped": float(features.row_count_bucket == "capped"),
    }


def score(features: Features, weights: dict[str, float] | None = None) -> tuple[float, dict[str, float]]:
    """(confidence 0-1, logit contribution per feature).

    A query that did not execute scores 0 with an empty breakdown: there is no
    answer to be confident in, and no weight was involved in saying so.
    """
    if not features.executed:
        return 0.0, {}
    weights = weights or WEIGHTS
    breakdown = {"bias": weights["bias"]}
    for name, value in encode(features).items():
        contribution = weights[name] * value
        if contribution:
            breakdown[name] = round(contribution, 4)
    logit = sum(breakdown.values())
    return 1.0 / (1.0 + math.exp(-logit)), breakdown


# ----------------------------------------------------------------- the log


def features_log_path() -> Path:
    override = os.getenv("QUERYGUARD_CONFIDENCE_LOG")
    return Path(override) if override else REPO_ROOT / "logs" / "confidence_features.jsonl"


def log_features(
    *,
    question: str,
    sql: str,
    features: Features,
    confidence: float,
    breakdown: dict[str, float],
    path: Path | None = None,
) -> Path:
    """Append one training row. The question and SQL are kept so a run can be labelled."""
    return append_log(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scorer_version": SCORER_VERSION,
            "question": question,
            "sql": sql,
            "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            "features": features.to_dict(),
            "encoded": encode(features),
            "confidence": confidence,
            "breakdown": breakdown,
            "label": None,
        },
        path or features_log_path(),
    )
