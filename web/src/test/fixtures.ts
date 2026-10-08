/** Typed builders for StageEvents and QueryResults, shaped exactly like the generated API types. */
import type { PayloadOf, QueryResult, Stage, StageEvent } from "../api/client";

let clock = 0;
export function resetClock() {
  clock = 0;
}

export function event<S extends Stage>(payload: PayloadOf<S>, duration = 10): StageEvent {
  clock += duration;
  return { stage: payload.stage, elapsed_ms: clock, duration_ms: duration, payload };
}

export const generating = (over: Partial<PayloadOf<"generating">> = {}): PayloadOf<"generating"> => ({
  stage: "generating", kind: "sql", sql: "SELECT count(*) AS n FROM orders", explanation: "Counts orders.",
  self_confidence: 0.9, assumptions: [], tables_used: ["orders"], cost_usd: 0, latency_ms: 10, ...over,
});

export const guardrails = (over: Partial<PayloadOf<"guardrails">> = {}): PayloadOf<"guardrails"> => ({
  stage: "guardrails", allowed: true, rule: null, reason: null,
  rewritten_sql: "SELECT count(*) AS n FROM orders\nLIMIT 1001", sql_to_execute: "SELECT count(*) AS n FROM orders\nLIMIT 1001", ...over,
});

export const executing = (over: Partial<PayloadOf<"executing">> = {}): PayloadOf<"executing"> => ({
  stage: "executing", outcome: "ok", row_count: 1, truncated: false, execution_ms: 3, columns: [{ name: "n", dtype: "int64" }],
  estimated_rows: 1, reason: null, error_class: null, error_message: null, sqlstate: null, ...over,
});

export const sanity = (): PayloadOf<"sanity"> => ({ stage: "sanity", flags: [] });

export const backtranslate = (): PayloadOf<"backtranslate"> => ({
  stage: "backtranslate", back_translation: "How many orders are there?", alignment: 1, discrepancies: [], error: null,
});

export const agreement = (): PayloadOf<"agreement"> => ({
  stage: "agreement", ran: true, outcome: "agree", explanation: "1 rows x 1 columns match", second_sql: "SELECT count(1) FROM orders",
  skipped_reason: null, error: null,
});

export const confidence = (over: Partial<PayloadOf<"confidence">> = {}): PayloadOf<"confidence"> => ({
  stage: "confidence", confidence: 0.95, band: "high", logit: 2.94, breakdown: {}, scorer_version: "calibrated-test",
  contributions: [
    { feature: "bias", label: "Baseline", value: 1, weight: -0.4, contribution: -0.4 },
    { feature: "agreement_agree", label: "Second query agreed", value: 1, weight: 2.07, contribution: 2.07 },
    { feature: "alignment_centered", label: "Back-translation alignment", value: 0.5, weight: 2.02, contribution: 1.01 },
    { feature: "sanity_warn", label: "Sanity warnings", value: 0, weight: -0.66, contribution: 0 },
  ],
  ...over,
});

export const result = (over: Partial<QueryResult> = {}): QueryResult => ({
  stage: "done", query_id: "q1", question: "How many orders are there?", outcome: "answered", cached: false, sql_source: "model", mode: "live", mode_reason: null,
  sql: "SELECT count(*) AS n FROM orders", executed_sql: "SELECT count(*) AS n FROM orders\nLIMIT 1001",
  explanation: "Counts orders.", assumptions: [], columns: ["n"], rows: [[5000]], row_count: 1, truncated: false, execution_ms: 3,
  guardrail_rule: null, guardrail_reason: null, execution_error: null, sanity: [],
  back_translation: "How many orders are there?", alignment: 1, discrepancies: [], agreement: "agree",
  agreement_explanation: "1 rows x 1 columns match", second_sql: "SELECT count(1) FROM orders", validation_errors: [],
  confidence: 0.95, confidence_band: "high", confidence_logit: 2.94, contributions: confidence().contributions,
  confidence_breakdown: {}, scorer_version: "calibrated-test", interpretations: [], cannot_answer_reason: null,
  n_calls: 4, cost_usd: 0, elapsed_ms: 900, ...over,
});

export const blockedResult = (): QueryResult => result({
  outcome: "blocked", sql: "DROP TABLE orders;", executed_sql: null, sql_source: "user", columns: [], rows: [], row_count: 0,
  execution_ms: null, guardrail_rule: "statement_type", guardrail_reason: "only SELECT and WITH ... SELECT may be executed, got DROP",
  back_translation: null, alignment: null, agreement: null, agreement_explanation: null, second_sql: null,
  confidence: 0, confidence_band: "low", confidence_logit: null, contributions: [], n_calls: 0,
});

export const clarificationResult = (): QueryResult => result({
  outcome: "clarification", sql: null, executed_sql: null, columns: [], rows: [], row_count: 0, execution_ms: null,
  back_translation: null, alignment: null, agreement: null, agreement_explanation: null, second_sql: null,
  confidence: null, confidence_band: null, confidence_logit: null, contributions: [], n_calls: 1, scorer_version: null,
  interpretations: [
    { label: "by_total_spend", sql: "SELECT customer_id FROM orders GROUP BY 1 ORDER BY sum(total_amount) DESC LIMIT 10", explanation: "By spend." },
    { label: "by_order_count", sql: "SELECT customer_id FROM orders GROUP BY 1 ORDER BY count(*) DESC LIMIT 10", explanation: "By orders." },
  ],
});

/** A normal question's full event sequence, in order. */
export function normalRun(): StageEvent[] {
  resetClock();
  return [
    event(generating()), event(guardrails()), event(executing()), event(sanity()),
    event(backtranslate()), event(agreement()), event(confidence()), event(result()),
  ];
}
