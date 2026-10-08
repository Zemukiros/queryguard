"""Run the golden set through the pipeline and record calibration data.

    uv run python -m evals.run_eval                       # dry run, fake client, no spend
    uv run python -m evals.run_eval --live --run-id ID    # real API; re-run the same ID to resume
    uv run python -m evals.run_eval --live --run-id ID --category refund_trap --population generated --population mutation
    uv run python -m evals.run_eval --live --run-id ID --reuse-second-sql BASE   # see "Reusing second queries"

Three populations, interleaved so a run stopped early still has all three:

  generated  all 50 golden questions through run_question (the full pipeline).
             correct = result matches the golden result (compare_results, the
             generated result may add columns but must contain every golden
             one); for ambiguous items, a clarification; for unanswerable
             items, a refusal (CannotAnswer), a clarification, or
             self-confidence below REFUSAL_CONFIDENCE.
  mutation   the 104 known-wrong mutation SQLs through run_answer.  label: wrong
  golden     the 40 golden SQLs through run_answer.                label: correct

The last two have no model-written confidence. Both get the same injected
INJECTED_SELF_CONFIDENCE and are tagged `self_confidence_source: injected`, so
self-confidence carries no label information within them; calibration should
treat it accordingly.

Two golden options loosen the comparison where the golden query is more
specific than the question: `compare_columns` (only these golden columns must
appear -- e.g. not the arbitrary row id chosen to identify a line item) and
`null_label_ok` (a NULL group may come back under a consistent label, e.g.
COALESCE(approved_by, 'auto-approved')).

Reusing second queries (--reuse-second-sql BASE)
  The mutation and golden SQL never changes, and neither, for a given SQL, does
  the second query much matter to a judge or prompt change -- yet it is the
  most expensive step. With this flag those items re-run back-translation and
  the judge only, and copy the second query (and everything not derived from
  an API call) from BASE's corrected rows; evals.recompute then re-executes
  that second query and re-derives agreement, sanity and labels offline.
  Generated items always run the full pipeline. Each row records
  `prompt_version` (queryguard.prompt_version) and, when reused,
  `second_sql_from`.

Spending guards
- Every API call is logged to evals/results/<run_id>.llm_calls.jsonl (the
  client's own log, redirected). That file is the ledger: cumulative calls and
  cost are read from it, so they survive crashes and resumes.
- Before an item starts, its worst case (max calls for its population x the
  highest observed cost of each step) is reserved. An item that could push the
  ledger plus in-flight reservations past --max-calls or --max-cost is not
  started, and the run stops cleanly. Each call also re-checks the ledger.
- Finished items are appended to evals/results/<run_id>.jsonl and skipped on
  re-run, so a crash never re-spends finished work.
- 429s and 5xx: the SDK retries with backoff (max_retries=5). An item that still
  fails transiently -- raised, or swallowed by a validation step -- is not
  recorded; it is re-queued after a global backoff, up to MAX_ATTEMPTS.

The dry run answers every call from a fake whose usage is the mean token count
of that step in logs/llm_calls.jsonl, so its cost total is the projection.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evals.common import EVALS_DIR, GOLDEN_RESULTS, MUTATION_RESULTS, load_golden, result_hash, run_guarded
from queryguard.generate import Ambiguity, CannotAnswer, ClarificationNeeded, GeneratedSQL, Interpretation
from queryguard.llm.client import DEFAULT_MODEL, LLMClient, RequestCapExceeded, estimate_cost_usd
from queryguard.pipeline import PipelineResult, run_answer, run_question
from queryguard.prompt_version import prompt_version
from queryguard.schema.introspect import load_schema
from queryguard.validation.agreement import AGREE, compare_results
from queryguard.validation.backtranslate import (
    VALIDATION_MODEL,
    AlignmentJudgement,
    BackTranslation,
    back_translate,
    judge_alignment,
)
from queryguard.validation.confidence import V0_WEIGHTS, Features, score

RESULTS_DIR = EVALS_DIR / "results"
CALL_LOG = EVALS_DIR.parent / "logs" / "llm_calls.jsonl"

GENERATED, MUTATION, GOLDEN = "generated", "mutation", "golden"
MAX_CALLS_PER_ITEM = {GENERATED: 4, MUTATION: 3, GOLDEN: 3}
REJUDGE_STEPS = ("back_translate", "judge")

DEFAULT_MAX_CALLS = 600
DEFAULT_MAX_COST_USD = 4.00
DEFAULT_CONCURRENCY = 4
MAX_ATTEMPTS = 3
BACKOFF_START_S = 15.0
BACKOFF_MAX_S = 120.0

INJECTED_SELF_CONFIDENCE = 0.9
REFUSAL_CONFIDENCE = 0.5
ALIGNMENT_FLAG_BELOW = 0.7

STEPS = ("generate", "back_translate", "judge", "second_sql")
STEP_MODEL = {
    "generate": DEFAULT_MODEL,
    "second_sql": DEFAULT_MODEL,
    "back_translate": VALIDATION_MODEL,
    "judge": VALIDATION_MODEL,
}
# Exception class names that mean "try again later", not "this item is bad".
TRANSIENT = {
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
    "ServiceUnavailableError", "OverloadedError",
}


# ---------------------------------------------------------------------- items


@dataclass(frozen=True)
class Item:
    id: str
    population: str
    category: str
    question: str
    golden_id: str
    ordered: bool = False
    compare_columns: tuple[str, ...] | None = None
    null_label_ok: bool = False
    sql: str | None = None  # mutation and golden populations
    mutation: str | None = None
    expected_outcome: str | None = None  # ambiguous / unanswerable generated items


def select_items(
    items: list[Item], categories: list[str] | None = None, populations: list[str] | None = None
) -> list[Item]:
    """A targeted re-run's subset, in build_items order. None keeps everything."""
    return [
        i for i in items
        if (not categories or i.category in categories) and (not populations or i.population in populations)
    ]


def build_items() -> list[Item]:
    """All 194 items, populations interleaved round-robin."""
    golden = load_golden()
    by_id = {e["id"]: e for e in golden}
    generated = [
        Item(f"gen:{e['id']}", GENERATED, e["category"], e["question"], e["id"],
             ordered=bool(e.get("ordered")), expected_outcome=e.get("expected_outcome"),
             compare_columns=tuple(e["compare_columns"]) if e.get("compare_columns") else None,
             null_label_ok=bool(e.get("null_label_ok")))
        for e in golden
    ]
    mutations = [
        Item(f"mut:{m['id']}", MUTATION, m["category"], m["question"], m["golden_id"],
             ordered=bool(by_id[m["golden_id"]].get("ordered")), sql=m["sql"], mutation=m["mutation"])
        for m in json.loads(MUTATION_RESULTS.read_text(encoding="utf-8"))["kept"]
    ]
    positives = [
        Item(f"gold:{e['id']}", GOLDEN, e["category"], e["question"], e["id"],
             ordered=bool(e.get("ordered")), sql=e["golden_sql"])
        for e in golden if "golden_sql" in e
    ]
    queues = [deque(generated), deque(mutations), deque(positives)]
    ordered_items: list[Item] = []
    while any(queues):
        for q in queues:
            if q:
                ordered_items.append(q.popleft())
    return ordered_items


def golden_frames() -> dict:
    """Golden results, re-executed and checked against golden_results.json."""
    stored = json.loads(GOLDEN_RESULTS.read_text(encoding="utf-8"))["results"]
    frames = {}
    for e in load_golden():
        if "golden_sql" not in e:
            continue
        run = run_guarded(e["golden_sql"])
        if not run.ok or result_hash(run.execution.rows, ordered=bool(e.get("ordered"))) != stored[e["id"]]["result_sha256"]:
            raise SystemExit(f"{e['id']}: golden result changed; re-run evals.run_golden")
        frames[e["id"]] = run.execution.rows
    return frames


# --------------------------------------------------------------- token profile


def token_profile(path: Path = CALL_LOG) -> dict[str, dict]:
    """Per-step mean usage and max cost from real calls.

    The log has no step field, so steps are told apart by model and size:
    Haiku back-translation carries the schema (>2000 input tokens), the judge
    does not; Sonnet's second opinion sends the first query (>100 uncached
    input tokens), generation sends only the question.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def step(r: dict) -> str:
        if r["model"] == VALIDATION_MODEL:
            return "back_translate" if r["input_tokens"] > 2000 else "judge"
        return "second_sql" if r["input_tokens"] > 100 else "generate"

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[step(r)].append(r)
    missing = [s for s in STEPS if not grouped[s]]
    if missing:
        raise SystemExit(f"no logged calls for {missing} in {path}; cannot project costs")

    profile = {}
    for name in STEPS:
        group = grouped[name]
        mean = {
            field_: round(statistics.mean(r[field_] for r in group))
            for field_ in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        }
        costs = [r["estimated_cost_usd"] for r in group]
        profile[name] = {
            "samples": len(group),
            "mean_usage": mean,
            "mean_cost": estimate_cost_usd(STEP_MODEL[name], SimpleNamespace(**mean)),
            "max_cost": max(costs),
        }
    return profile


# --------------------------------------------------------------------- budget


class BudgetExhausted(RequestCapExceeded):
    """The eval's call or cost cap is reached. No API call was made."""


class Budget:
    """Ledger-backed caps, with reservations for items in flight."""

    def __init__(
        self, ledger: Path, max_calls: int, max_cost: float, profile: dict, reuse_second_sql: bool = False
    ) -> None:
        self.ledger = ledger
        self.reuse_second_sql = reuse_second_sql
        self.max_calls = max_calls
        self.max_cost = max_cost
        self.step_max_cost = {s: profile[s]["max_cost"] for s in STEPS}
        self._lock = threading.Lock()
        self._reserved_calls = 0
        self._reserved_cost = 0.0

    def spent(self) -> tuple[int, float]:
        if not self.ledger.is_file():
            return 0, 0.0
        calls, cost = 0, 0.0
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                calls += 1
                cost += float(json.loads(line).get("estimated_cost_usd", 0.0))
        return calls, cost

    def worst_case(self, population: str) -> tuple[int, float]:
        if population == GENERATED:
            steps = STEPS
        else:
            steps = REJUDGE_STEPS if self.reuse_second_sql else STEPS[1:]
        return len(steps), sum(self.step_max_cost[s] for s in steps)

    def reserve(self, population: str) -> tuple[int, float] | None:
        """Hold an item's worst case, or None if it could breach a cap."""
        calls, cost = self.worst_case(population)
        with self._lock:
            spent_calls, spent_cost = self.spent()
            if spent_calls + self._reserved_calls + calls > self.max_calls:
                return None
            if spent_cost + self._reserved_cost + cost > self.max_cost:
                return None
            self._reserved_calls += calls
            self._reserved_cost += cost
            return calls, cost

    def release(self, reservation: tuple[int, float]) -> None:
        with self._lock:
            self._reserved_calls -= reservation[0]
            self._reserved_cost -= reservation[1]

    def check_call(self) -> None:
        """Last line of defence, before every request."""
        calls, cost = self.spent()
        if calls >= self.max_calls or cost >= self.max_cost:
            raise BudgetExhausted(f"eval budget reached: {calls} calls, ${cost:.4f}")


# --------------------------------------------------------------------- client


def _step_of(output_format: Any, user_message: str) -> str:
    if output_format is BackTranslation:
        return "back_translate"
    if output_format is AlignmentJudgement:
        return "judge"
    return "second_sql" if "Earlier query:" in user_message else "generate"


class RecordingClient(LLMClient):
    """LLMClient that checks the eval budget first and records every call."""

    def __init__(self, sdk: Any, model: str, budget: Budget, record: list[dict]) -> None:
        super().__init__(sdk_client=sdk, model=model)
        self._budget = budget
        self._record = record

    def complete(self, system_blocks, user_message, output_format, **kwargs):
        self._budget.check_call()
        result = super().complete(system_blocks, user_message, output_format, **kwargs)
        usage = result.usage
        step = _step_of(output_format, user_message)
        record = {}
        if step == "back_translate":  # the judge's input; the pipeline result keeps only the question
            record["output"] = {"question": result.parsed.question, "details": list(result.parsed.details)}
        self._record.append({
            **record,
            "step": step,
            "model": self.model,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        })
        return result


def _answer(sql: str, confidence: float = INJECTED_SELF_CONFIDENCE) -> GeneratedSQL:
    return GeneratedSQL(
        sql=sql, explanation="Injected by the eval harness.", confidence=confidence,
        tables_used=[], columns_used=[], assumptions=[],
        ambiguity=Ambiguity(is_ambiguous=False, interpretations=[]),
    )


class FakeSDK:
    """Dry-run stand-in for anthropic.Anthropic: plausible answers, mean real usage.

    Generation returns the golden SQL (so the full validation path runs), an
    ambiguous answer for ambiguous items, and a low-confidence count for
    unanswerable ones -- the worst case, since it then runs every check.
    """

    def __init__(self, profile: dict) -> None:
        self.messages = self
        self._usage = {s: profile[s]["mean_usage"] for s in STEPS}
        self._by_question = {e["question"]: e for e in load_golden()}

    def parse(self, *, output_format, messages, **kwargs):
        content = messages[0]["content"]
        step = _step_of(output_format, content)
        if step == "back_translate":
            parsed = BackTranslation(question="(dry run)", details=[])
        elif step == "judge":
            parsed = AlignmentJudgement(alignment=1.0, discrepancies=[])
        elif step == "second_sql":
            parsed = _answer(content.split("Earlier query:\n", 1)[1])
        else:
            entry = self._by_question[content.removeprefix("Q: ")]
            if entry.get("expected_outcome") == "clarification":
                parsed = GeneratedSQL(
                    sql="", explanation="(dry run)", confidence=0.0, tables_used=[], columns_used=[],
                    assumptions=[], ambiguity=Ambiguity(is_ambiguous=True, interpretations=[
                        Interpretation(label="a", sql="SELECT 1", explanation="a"),
                        Interpretation(label="b", sql="SELECT 2", explanation="b"),
                    ]),
                )
            elif "golden_sql" in entry:
                parsed = _answer(entry["golden_sql"])
            else:
                parsed = _answer("SELECT count(*) AS n FROM orders AS o", confidence=0.3)
        return SimpleNamespace(parsed_output=parsed, usage=SimpleNamespace(**self._usage[step]))


# ------------------------------------------------------------------- labelling


def _outcome_kind(outcome: Any) -> str:
    if isinstance(outcome, ClarificationNeeded):
        return "clarification"
    if isinstance(outcome, CannotAnswer):
        return "cannot_answer"
    if not outcome.guardrail.allowed:
        return "blocked"
    if not outcome.ok:
        return "failed"
    return "truncated" if outcome.execution.truncated else "executed"


def label_generated(item: Item, outcome: Any, frames: dict) -> tuple[str, str]:
    """(label, why) for a generated item."""
    kind = _outcome_kind(outcome)
    if item.expected_outcome == "clarification":
        return ("correct", "asked for clarification") if kind == "clarification" else ("wrong", f"{kind} instead of asking")
    if item.expected_outcome == "refusal_or_clarification":
        if kind == "cannot_answer":
            return "correct", f"refused: {outcome.explanation}"
        if kind == "clarification":
            return "correct", "asked for clarification"
        confidence = outcome.answer.confidence
        if confidence < REFUSAL_CONFIDENCE:
            return "correct", f"answered with self-confidence {confidence:.2f} (< {REFUSAL_CONFIDENCE}), counted as refusal"
        return "wrong", f"{kind} with self-confidence {confidence:.2f} for an unanswerable question"
    if kind != "executed":
        return "wrong", kind
    verdict, explanation = compare_to_golden(item, frames[item.golden_id], outcome.execution.rows)
    return ("correct" if verdict == AGREE else "wrong"), f"{verdict}: {explanation}"


def _label_nulls(golden, generated):
    """Fill golden NULLs in a text column with the one label the generated result adds.

    COALESCE(approved_by, 'auto-approved') answers "count auto-approved
    refunds as their own group" exactly; only the label differs from NULL.
    """
    golden = golden.copy()
    generated_text = {
        v for col in generated.columns for v in generated[col].tolist() if isinstance(v, str)
    }
    golden_text = {
        v for col in golden.columns for v in golden[col].tolist() if isinstance(v, str)
    }
    labels = generated_text - golden_text
    if len(labels) != 1:
        return golden
    (label,) = labels
    for col in golden.columns:
        values = golden[col].tolist()
        if any(isinstance(v, str) for v in values) and golden[col].isna().any():
            golden[col] = golden[col].astype(object).where(golden[col].notna(), label)
    return golden


def compare_to_golden(item: Item, golden, generated) -> tuple[str, str]:
    """compare_results, with the generated side allowed to add columns."""
    if item.compare_columns:
        golden = golden[list(item.compare_columns)]
    if item.null_label_ok:
        golden = _label_nulls(golden, generated)
    return compare_results(golden, generated, ordered=item.ordered, extra_columns="second")


def _detectors(outcome: Any) -> dict:
    if not isinstance(outcome, PipelineResult):
        return {}
    severities = Counter(f.severity for f in outcome.sanity)
    agreement = outcome.agreement
    return {
        "sanity": {
            "fail": severities["fail"], "warn": severities["warn"], "info": severities["info"],
            "checks": [f"{f.severity}:{f.check}" for f in outcome.sanity],
            "flagged": severities["fail"] + severities["warn"] > 0,
        },
        "alignment": {
            "score": outcome.alignment,
            "discrepancies": list(outcome.discrepancies),
            "back_translation": outcome.back_translation,
            "flagged": outcome.alignment is not None and outcome.alignment < ALIGNMENT_FLAG_BELOW,
        },
        "agreement": {
            "outcome": agreement.outcome if agreement else None,
            "explanation": agreement.explanation if agreement else None,
            "second_sql": agreement.second_sql if agreement else None,
            "flagged": bool(agreement) and agreement.outcome != AGREE,
        },
    }


def _transient_errors(outcome: Any) -> list[str]:
    errors = getattr(outcome, "validation_errors", ()) or ()
    return [e for e in errors if any(f" {name}:" in e or e.startswith(f"{name}:") for name in TRANSIENT)]


# ------------------------------------------------------------------- one item


@dataclass
class Context:
    run_id: str
    dry_run: bool
    sdk: Any
    budget: Budget
    schema: Any
    frames: dict
    results_path: Path
    write_lock: threading.Lock = field(default_factory=threading.Lock)
    prompt_version: str = ""
    reuse: dict[str, dict] | None = None  # id -> base corrected row, with --reuse-second-sql


class TransientFailure(Exception):
    pass


def rejudge_item(item: Item, ctx: Context) -> dict:
    """Back-translation and judge only; the second query comes from the base row."""
    base = json.loads(json.dumps(ctx.reuse[item.id]))
    if base["sql"].strip() != item.sql.strip():
        raise ValueError(f"{item.id}: base row ran different SQL")
    calls: list[dict] = []
    validation_client = RecordingClient(ctx.sdk, VALIDATION_MODEL, ctx.budget, calls)
    started = time.perf_counter()
    alignment, discrepancies, back_translation, errors = None, [], None, []
    try:
        translated, _ = back_translate(item.sql, client=validation_client, schema=ctx.schema)
        back_translation = translated.question
        judgement, _ = judge_alignment(item.question, translated, client=validation_client)
        alignment, discrepancies = judgement.alignment, list(judgement.discrepancies)
    except BudgetExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - the pipeline records a failed validation step the same way
        if type(exc).__name__ in TRANSIENT:
            raise TransientFailure(f"{type(exc).__name__}: {exc}") from exc
        errors.append(f"alignment: {type(exc).__name__}: {exc}")

    features = {**base["features"], "alignment": alignment, "discrepancy_count": len(discrepancies)}
    confidence, breakdown = score(Features(**features), V0_WEIGHTS)
    base["detectors"]["alignment"] = {
        "score": alignment, "discrepancies": discrepancies, "back_translation": back_translation,
        "flagged": alignment is not None and alignment < ALIGNMENT_FLAG_BELOW,
    }
    for stale in ("corrections", "source_run"):
        base.pop(stale, None)
    base.update(
        run_id=ctx.run_id, dry_run=ctx.dry_run, finished_at=datetime.now(timezone.utc).isoformat(),
        features=features, confidence=confidence, confidence_breakdown=breakdown,
        validation_errors=errors, calls=calls, n_calls=len(calls),
        cost_usd=round(sum(c["cost_usd"] for c in calls), 6),
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_version=ctx.prompt_version,
        second_sql_from=ctx.reuse[item.id].get("source_run") or ctx.reuse[item.id]["run_id"],
    )
    return base


def run_item(item: Item, ctx: Context) -> dict:
    if ctx.reuse is not None and item.population != GENERATED:
        return rejudge_item(item, ctx)
    calls: list[dict] = []
    client = RecordingClient(ctx.sdk, DEFAULT_MODEL, ctx.budget, calls)
    validation_client = RecordingClient(ctx.sdk, VALIDATION_MODEL, ctx.budget, calls)
    started = time.perf_counter()
    try:
        if item.population == GENERATED:
            outcome = run_question(item.question, client=client, validation_client=validation_client, schema=ctx.schema)
        else:
            outcome = run_answer(item.question, _answer(item.sql), client=client,
                                 validation_client=validation_client, schema=ctx.schema)
    except BudgetExhausted:
        raise
    except Exception as exc:  # noqa: BLE001 - classified below
        if type(exc).__name__ in TRANSIENT:
            raise TransientFailure(f"{type(exc).__name__}: {exc}") from exc
        raise
    latency_ms = int((time.perf_counter() - started) * 1000)

    transient = _transient_errors(outcome)
    if transient:
        raise TransientFailure("; ".join(transient))

    if item.population == GENERATED:
        label, why = label_generated(item, outcome, ctx.frames)
    else:
        label, why = ("wrong" if item.population == MUTATION else "correct"), f"known {item.population}"

    pipeline = outcome if isinstance(outcome, PipelineResult) else None
    return {
        "id": item.id,
        "run_id": ctx.run_id,
        "dry_run": ctx.dry_run,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "population": item.population,
        "category": item.category,
        "golden_id": item.golden_id,
        "mutation": item.mutation,
        "question": item.question,
        "sql": pipeline.answer.sql if pipeline else None,
        "outcome": _outcome_kind(outcome),
        "label": label,
        "label_reason": why,
        "self_confidence_source": "model" if item.population == GENERATED else "injected",
        "features": pipeline.features.to_dict() if pipeline and pipeline.features else None,
        "confidence": pipeline.confidence if pipeline else None,
        "confidence_breakdown": pipeline.confidence_breakdown if pipeline else None,
        "detectors": _detectors(outcome),
        "validation_errors": list(pipeline.validation_errors) if pipeline else [],
        "calls": calls,
        "n_calls": len(calls),
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
        "latency_ms": latency_ms,
        "prompt_version": ctx.prompt_version,
    }


# ----------------------------------------------------------------------- driver


def finished_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def run(items: list[Item], ctx: Context, concurrency: int) -> str:
    """Run every pending item. Returns why it stopped."""
    done = finished_ids(ctx.results_path)
    queue = deque(i for i in items if i.id not in done)
    attempts: Counter = Counter()
    backoff = BACKOFF_START_S
    pause_until = 0.0
    stop_reason = "all items finished"

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        in_flight: dict = {}
        while queue or in_flight:
            while queue and len(in_flight) < concurrency and time.monotonic() >= pause_until and stop_reason == "all items finished":
                item = queue[0]
                reservation = ctx.budget.reserve(item.population)
                if reservation is None:
                    calls, cost = ctx.budget.spent()
                    stop_reason = (
                        f"budget: starting {item.id} could exceed {ctx.budget.max_calls} calls / "
                        f"${ctx.budget.max_cost:.2f} (spent {calls} calls, ${cost:.4f})"
                    )
                    break
                queue.popleft()
                in_flight[pool.submit(run_item, item, ctx)] = (item, reservation)
            if not in_flight:
                if stop_reason != "all items finished":
                    break
                time.sleep(max(0.0, pause_until - time.monotonic()))
                continue

            finished, _ = wait(in_flight, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in finished:
                item, reservation = in_flight.pop(future)
                ctx.budget.release(reservation)
                try:
                    row = future.result()
                except TransientFailure as exc:
                    attempts[item.id] += 1
                    if attempts[item.id] < MAX_ATTEMPTS:
                        queue.append(item)
                    print(f"  transient on {item.id} (attempt {attempts[item.id]}): {exc}; backing off {backoff:.0f}s")
                    pause_until = time.monotonic() + backoff
                    backoff = min(backoff * 2, BACKOFF_MAX_S)
                    continue
                except BudgetExhausted as exc:
                    stop_reason = str(exc)
                    continue
                except Exception as exc:  # noqa: BLE001 - one bad item must not sink the run
                    print(f"  ERROR {item.id}: {type(exc).__name__}: {exc}")
                    continue
                backoff = BACKOFF_START_S
                with ctx.write_lock:
                    with ctx.results_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
                    total = len(finished_ids(ctx.results_path))
                if total % 20 == 0:
                    calls, cost = ctx.budget.spent()
                    print(f"  {total} finished, {calls} calls, ${cost:.4f}")
    return stop_reason


# ----------------------------------------------------------------------- report


def report(results_path: Path, profile: dict, budget: Budget, dry_run: bool) -> None:
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_pop: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pop[r["population"]].append(r)

    title = "PROJECTION (dry run: mean real tokens per step)" if dry_run else "RESULTS"
    print(f"\n{title}")
    cost_header = "expected $" if dry_run else "cost $"
    header = f"{'population':<11}{'items':>6}{'gen':>6}{'bt':>6}{'judge':>7}{'2nd':>6}{'calls':>7}{cost_header:>12}{'high $':>9}{'correct':>9}{'wrong':>7}"
    print(header)
    total_calls = total_cost = total_high = 0.0
    for pop in (GENERATED, MUTATION, GOLDEN):
        group = by_pop.get(pop, [])
        steps = Counter(c["step"] for r in group for c in r["calls"])
        calls = sum(steps.values())
        cost = sum(r["cost_usd"] for r in group)
        high = sum(profile[s]["max_cost"] * n for s, n in steps.items())
        labels = Counter(r["label"] for r in group)
        print(f"{pop:<11}{len(group):>6}{steps['generate']:>6}{steps['back_translate']:>6}{steps['judge']:>7}"
              f"{steps['second_sql']:>6}{calls:>7}{cost:>12.4f}{high:>9.4f}{labels['correct']:>9}{labels['wrong']:>7}")
        total_calls += calls
        total_cost += cost
        total_high += high
    print(f"{'total':<11}{len(rows):>6}{'':>25}{int(total_calls):>7}{total_cost:>12.4f}{total_high:>9.4f}")
    print(f"\ncaps: {budget.max_calls} calls, ${budget.max_cost:.2f}")
    if dry_run:
        print("\nper-step token profile from logs/llm_calls.jsonl:")
        for s in STEPS:
            p = profile[s]
            print(f"  {s:<15} n={p['samples']:<3} mean usage {p['mean_usage']}  mean ${p['mean_cost']:.4f}  max ${p['max_cost']:.4f}")


def final_status(stop_reason: str, unfinished: list[str]) -> tuple[str, int]:
    """(stop reason, exit code). Exit 0 only when every item has a row.

    The queue can run dry with items missing -- errors, or transient failures
    past MAX_ATTEMPTS. The first live run printed "all items finished" and
    exited 0 in exactly that state.
    """
    if not unfinished:
        return stop_reason, 0
    if stop_reason == "all items finished":
        stop_reason = "queue exhausted with items unfinished (errors or retries exhausted)"
    return stop_reason, 3


class _DropHaikuCacheFloorWarning(logging.Filter):
    """The back-translation prefix is below Haiku's cache floor by design, so
    that warning would fire on every item. Sonnet cache misses still show."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (VALIDATION_MODEL in record.getMessage() and "cache_control" in record.getMessage())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run_eval")
    parser.add_argument("--live", action="store_true", help="use the real API (spends money)")
    parser.add_argument("--run-id", help="required with --live; re-use to resume")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, help="only the first N items (smoke tests)")
    parser.add_argument("--category", action="append", help="only this golden category (repeatable)")
    parser.add_argument("--population", action="append", choices=(GENERATED, MUTATION, GOLDEN),
                        help="only this population (repeatable)")
    parser.add_argument("--reuse-second-sql", metavar="BASE",
                        help="mutation/golden items: re-run back-translation and judge only, second query from BASE")
    args = parser.parse_args(argv)

    if args.live and not args.run_id:
        parser.error("--live needs an explicit --run-id, so a resume is deliberate")
    run_id = args.run_id or "dryrun"
    dry_run = not args.live
    if dry_run and args.run_id and not args.run_id.startswith("dryrun"):
        parser.error("dry-run ids must start with 'dryrun' so they cannot be mistaken for real data")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f"{run_id}.jsonl"
    ledger = RESULTS_DIR / f"{run_id}.llm_calls.jsonl"
    confidence_log = RESULTS_DIR / f"{run_id}.confidence.jsonl"
    if dry_run:
        for path in (results_path, ledger, confidence_log):
            path.unlink(missing_ok=True)  # a dry run always starts fresh

    # Redirect the client's call log (the ledger) and the confidence log into
    # the run directory: eval traffic is neither production traffic nor
    # unlabelled calibration rows.
    os.environ["QUERYGUARD_LLM_LOG"] = str(ledger)
    os.environ["QUERYGUARD_CONFIDENCE_LOG"] = str(confidence_log)
    os.environ["QUERYGUARD_MAX_REQUESTS"] = str(args.max_calls)

    logging.getLogger("queryguard.llm.client").addFilter(_DropHaikuCacheFloorWarning())
    profile = token_profile()
    budget = Budget(ledger, args.max_calls, args.max_cost, profile, reuse_second_sql=bool(args.reuse_second_sql))
    if dry_run:
        sdk = FakeSDK(profile)
    else:
        import anthropic

        from queryguard.config import load_env

        load_env()
        sdk = anthropic.Anthropic(max_retries=5)  # SDK backoff on 429/5xx, honouring retry-after

    items = select_items(build_items(), args.category, args.population)
    if args.category and not items:
        parser.error(f"no items in categories {args.category}")
    items = items[: args.limit] if args.limit else items
    schema = load_schema()
    reuse = None
    if args.reuse_second_sql:
        base_path = RESULTS_DIR / f"{args.reuse_second_sql}.corrected.jsonl"
        reuse = {r["id"]: r for r in map(json.loads, base_path.read_text(encoding="utf-8").splitlines()) if r}
        missing = [i.id for i in items if i.population != GENERATED and i.id not in reuse]
        if missing:
            parser.error(f"{len(missing)} items have no row in {base_path.name}, e.g. {missing[0]}")
    ctx = Context(run_id, dry_run, sdk, budget, schema, golden_frames(), results_path,
                  prompt_version=prompt_version(schema), reuse=reuse)
    print(f"{'DRY RUN' if dry_run else 'LIVE'} {run_id} ({ctx.prompt_version}): {len(items)} items, "
          f"{len(finished_ids(results_path))} already finished"
          + (f"; second queries reused from {args.reuse_second_sql}" if reuse else ""))
    stop_reason = run(items, ctx, args.concurrency)
    unfinished = sorted({i.id for i in items} - finished_ids(results_path))
    stop_reason, exit_code = final_status(stop_reason, unfinished)
    print(f"\nstopped: {stop_reason}")
    if unfinished:
        print(f"unfinished ({len(unfinished)}): {', '.join(unfinished)} -- re-run the same --run-id to retry them")
    report(results_path, profile, budget, dry_run)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
