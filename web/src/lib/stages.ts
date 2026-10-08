/**
 * The pipeline's stages as the timeline shows them. Order and meaning follow
 * src/queryguard/events.py: every stage is reported when it finishes, a
 * clarification or refusal ends after `generating`, a guardrail rejection
 * skips execution and validation, and validation runs only when the query
 * executed. /v1/run streams start at `guardrails` (the SQL is not generated).
 *
 * A stage that finished is not automatically a pass: a rejection is
 * "blocked", and a check whose signal lowers trust is "warn". Only a stage
 * that ran and found nothing wrong gets "done".
 */
import type { PayloadOf, QueryResult, Stage, StageEvent } from "../api/client";
import { formatMs } from "./format";

export const TIMELINE: readonly Stage[] = [
  "generating", "clarification", "guardrails", "executing", "sanity",
  "backtranslate", "agreement", "confidence", "done",
];

export const STAGE_LABEL: Record<Stage, string> = {
  generating: "Generate SQL",
  clarification: "Clarify",
  guardrails: "Guardrail",
  executing: "Execute read-only",
  sanity: "Sanity checks",
  backtranslate: "Back-translate + judge",
  agreement: "Second query",
  confidence: "Confidence",
  done: "Result",
  error: "Error",
};

/** blocked: the stage stopped the answer. warn: it ran and its signal lowers trust. */
export type RowState = "pending" | "active" | "done" | "warn" | "blocked" | "skipped" | "failed";
export type Mode = "query" | "run";

type Seen = Partial<Record<Stage, StageEvent>>;
type Verdict = "done" | "warn" | "blocked" | "skipped";

const ALIGNMENT_FLAG = 0.7; // the eval's flag threshold (evals/run_eval.py)

function payload<S extends Stage>(seen: Seen, stage: S): PayloadOf<S> | undefined {
  return seen[stage]?.payload as PayloadOf<S> | undefined;
}

/** What a finished stage found, from its event or (replayed) from the final result. */
export function verdict(stage: Stage, event: StageEvent | undefined, r: QueryResult | null): Verdict {
  const p = event?.payload;
  switch (stage) {
    case "guardrails":
      return (p?.stage === "guardrails" ? !p.allowed : r?.outcome === "blocked") ? "blocked" : "done";
    case "executing":
      if (p?.stage === "executing") return p.outcome === "ok" ? "done" : "blocked";
      return r?.outcome === "failed" || r?.outcome === "refused" ? "blocked" : "done";
    case "sanity": {
      const flags = p?.stage === "sanity" ? p.flags : (r?.sanity ?? []);
      return flags.some((f) => f.severity !== "info") ? "warn" : "done";
    }
    case "backtranslate": {
      if (p?.stage === "backtranslate") return p.error || (p.alignment !== null && p.alignment < ALIGNMENT_FLAG) ? "warn" : "done";
      return r?.alignment !== null && r?.alignment !== undefined && r.alignment < ALIGNMENT_FLAG ? "warn" : "done";
    }
    case "agreement":
      if (p?.stage === "agreement") return !p.ran ? "skipped" : p.error || p.outcome !== "agree" ? "warn" : "done";
      return r?.agreement && r.agreement !== "agree" ? "warn" : "done";
    case "confidence": {
      const logit = p?.stage === "confidence" ? p.logit : (r?.confidence_logit ?? null);
      const band = p?.stage === "confidence" ? p.band : r?.confidence_band;
      if (logit === null) return "skipped"; // nothing executed: there is no score to give
      return band === "low" ? "warn" : "done";
    }
    case "done": {
      const final = p?.stage === "done" ? p : r;
      const outcome = final?.outcome;
      if (outcome === "blocked" || outcome === "failed" || outcome === "refused") return "blocked";
      return outcome === "answered" && final?.confidence_band === "low" ? "warn" : "done";
    }
    default:
      return "done";
  }
}

/** Stages that cannot happen on the path taken so far. */
export function skippedStages(seen: Seen, mode: Mode, finished: boolean): Set<Stage> {
  const skipped = new Set<Stage>();
  const generating = payload(seen, "generating");
  const guardrails = payload(seen, "guardrails");
  const executing = payload(seen, "executing");

  if (mode === "run") {
    skipped.add("generating");
    skipped.add("clarification");
  } else if (generating && generating.kind !== "clarification") {
    skipped.add("clarification");
  }
  if (generating && generating.kind !== "sql") {
    for (const s of ["guardrails", "executing", "sanity", "backtranslate", "agreement", "confidence"] as const) skipped.add(s);
  }
  if (guardrails && !guardrails.allowed) {
    for (const s of ["executing", "sanity", "backtranslate", "agreement"] as const) skipped.add(s);
  }
  if (executing && executing.outcome !== "ok") {
    skipped.add("backtranslate");
    skipped.add("agreement");
  }
  if (finished) for (const s of TIMELINE) if (!seen[s]) skipped.add(s);
  return skipped;
}

export interface TimelineRow {
  stage: Stage;
  state: RowState;
  event: StageEvent | undefined;
  summary: string;
}

export function timelineRows(args: {
  seen: Seen;
  mode: Mode;
  running: boolean;
  failedStage: Stage | null;
  result: QueryResult | null;
  replayed?: boolean;
}): TimelineRow[] {
  const { seen, mode, running, failedStage, result, replayed = false } = args;
  // Nothing has happened yet (idle, or refused at the door with a 429/503): every stage is still to come.
  const untouched = !running && !result && failedStage === null && Object.keys(seen).length === 0;
  if (untouched) return TIMELINE.map((stage) => ({ stage, state: "pending", event: undefined, summary: "" }));
  const finished = !running;
  const skipped = skippedStages(seen, mode, finished && !result);
  let activeAssigned = false;
  let afterFailure = false;

  return TIMELINE.map((stage) => {
    const event = seen[stage];
    let state: RowState;
    if (stage === failedStage) {
      state = "failed";
      afterFailure = true;
    } else if (event) state = verdict(stage, event, null);
    else if (afterFailure || skipped.has(stage)) state = "skipped";
    else if (result) state = ranInResult(stage, result) ? verdict(stage, undefined, result) : "skipped";
    else if (running && !activeAssigned) {
      state = "active";
      activeAssigned = true;
    } else state = finished ? "skipped" : "pending";
    // A replayed run's events (a cache hit's lone `done`) carry no real timings.
    return { stage, state, event: replayed ? undefined : event, summary: summarize(stage, event, result) };
  });
}

/** For a cached or reopened result, which stages it shows ran (timings were not replayed). */
function ranInResult(stage: Stage, r: QueryResult): boolean {
  switch (stage) {
    case "generating": return r.sql_source === "model";
    case "clarification": return r.outcome === "clarification";
    case "guardrails": return r.outcome !== "clarification" && r.outcome !== "cannot_answer";
    case "executing":
    case "sanity": return r.executed_sql !== null;
    case "backtranslate": return r.back_translation !== null || r.alignment !== null;
    case "agreement": return r.agreement !== null;
    case "confidence": return r.confidence !== null && r.outcome !== "clarification" && r.outcome !== "cannot_answer";
    case "done": return true;
    default: return false;
  }
}

const plural = (n: number, word: string) => `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;

export function summarize(stage: Stage, event: StageEvent | undefined, result: QueryResult | null): string {
  const p = event?.payload;
  switch (stage) {
    case "generating":
      if (p?.stage === "generating") return p.kind === "sql" ? `SQL written · self-confidence ${p.self_confidence.toFixed(2)}` : p.kind === "clarification" ? "more than one reading" : "cannot be answered from this schema";
      if (result?.sql_source === "user") return "not used: your SQL";
      if (result?.sql_source === "reading") return "not used: a chosen reading";
      return "";
    case "clarification":
      if (p?.stage === "clarification") return `${plural(p.interpretations.length, "reading")}: pick one`;
      return result?.outcome === "clarification" ? plural(result.interpretations.length, "reading") : "";
    case "guardrails":
      if (p?.stage === "guardrails") return p.allowed ? (p.rewritten_sql ? "allowed · LIMIT added" : "allowed") : `rejected · ${p.rule ?? "rule"}`;
      if (result?.outcome === "blocked") return `rejected · ${result.guardrail_rule ?? "rule"}`;
      return "";
    case "executing":
      if (p?.stage === "executing") return p.outcome === "ok" ? `${plural(p.row_count, "row")} · ${formatMs(p.execution_ms)}${p.truncated ? " · truncated" : ""}` : `${p.outcome}: ${p.reason ?? p.error_class ?? "error"}`;
      return result?.executed_sql ? plural(result.row_count, "row") : "";
    case "sanity": {
      const flags = p?.stage === "sanity" ? p.flags : result?.executed_sql ? result.sanity : null;
      if (!flags) return "";
      return flags.length === 0 ? "no flags" : plural(flags.length, "flag");
    }
    case "backtranslate":
      if (p?.stage === "backtranslate") return p.error ? "could not run: signal missing" : p.alignment !== null ? `alignment ${p.alignment.toFixed(2)}` : "";
      return result?.alignment !== null && result?.alignment !== undefined ? `alignment ${result.alignment.toFixed(2)}` : "";
    case "agreement":
      if (p?.stage === "agreement") return p.ran ? (p.error ? "could not run: signal missing" : (p.outcome ?? "")) : "not needed: trivial query";
      return result?.agreement ?? "";
    case "confidence":
      if (p?.stage === "confidence") return p.logit === null ? "n/a: nothing executed" : `${p.confidence.toFixed(2)} · ${p.band}`;
      if (result?.confidence_logit === null) return "n/a: nothing executed";
      return result?.confidence !== null && result?.confidence !== undefined ? result.confidence.toFixed(2) : "";
    case "done": {
      const final = p?.stage === "done" ? p : result;
      if (!final) return "";
      return final.outcome === "answered" && final.confidence_band === "low" ? "answered · low confidence" : OUTCOME_TEXT[final.outcome];
    }
    default:
      return "";
  }
}

export const OUTCOME_TEXT: Record<QueryResult["outcome"], string> = {
  answered: "answered",
  clarification: "needs clarification",
  cannot_answer: "cannot be answered",
  blocked: "blocked: nothing ran",
  refused: "refused before running",
  failed: "the database rejected it",
};

/** One-line verdict for the timeline header. */
export function headline(rows: TimelineRow[], result: QueryResult | null): { text: string; tone: "ok" | "warn" | "fail" | "info" | "neutral" } | null {
  const failed = rows.find((r) => r.state === "failed");
  if (failed) return { text: `Stopped at ${STAGE_LABEL[failed.stage]}`, tone: "fail" };
  if (!result) return null;
  const blocked = rows.find((r) => r.state === "blocked");
  if (blocked) return { text: `Blocked at ${STAGE_LABEL[blocked.stage]}`, tone: "fail" };
  if (result.outcome === "clarification") return { text: "Needs clarification", tone: "info" };
  if (result.outcome === "cannot_answer") return { text: "Cannot be answered", tone: "neutral" };
  // Count the checks that raised a concern, not the score or result they fed into.
  const CHECKS: readonly Stage[] = ["sanity", "backtranslate", "agreement"];
  const warnings = rows.filter((r) => r.state === "warn" && CHECKS.includes(r.stage)).length;
  if (warnings > 0) return { text: `Answered · ${plural(warnings, "check")} flagged`, tone: "warn" };
  if (result.confidence_band === "low") return { text: "Answered · low confidence", tone: "warn" };
  return { text: "Answered · all checks passed", tone: "ok" };
}
