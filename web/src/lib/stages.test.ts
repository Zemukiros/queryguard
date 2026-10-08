import { describe, expect, it } from "vitest";

import type { Stage, StageEvent } from "../api/client";
import { blockedResult, event, generating, guardrails, normalRun, resetClock, result } from "../test/fixtures";
import { headline, TIMELINE, timelineRows } from "./stages";

const seenOf = (events: StageEvent[]) => Object.fromEntries(events.map((e) => [e.stage, e])) as Partial<Record<Stage, StageEvent>>;
const states = (rows: ReturnType<typeof timelineRows>) => Object.fromEntries(rows.map((r) => [r.stage, r.state]));

describe("timelineRows", () => {
  it("lights stages in pipeline order as events arrive, with one active row", () => {
    const events = normalRun();
    for (let n = 1; n < events.length; n++) {
      const rows = timelineRows({ seen: seenOf(events.slice(0, n)), mode: "query", running: true, failedStage: null, result: null });
      expect(rows.map((r) => r.stage)).toEqual(TIMELINE);
      expect(rows.filter((r) => r.state === "active")).toHaveLength(1);
      const active = rows.findIndex((r) => r.state === "active");
      expect(rows.slice(0, active).every((r) => r.state === "done" || r.state === "skipped")).toBe(true);
      expect(rows.slice(active + 1).every((r) => r.state === "pending")).toBe(true);
    }
    const final = timelineRows({ seen: seenOf(events), mode: "query", running: false, failedStage: null, result: null });
    expect(states(final)).toMatchObject({ generating: "done", clarification: "skipped", agreement: "done", done: "done" });
    expect(headline(final, result())).toEqual({ text: "Answered · all checks passed", tone: "ok" });
  });

  it("marks a guardrail rejection as blocked and skips execution and validation", () => {
    resetClock();
    const seen = seenOf([event(generating()), event(guardrails({ allowed: false, rule: "statement_type", reason: "got DROP", rewritten_sql: null, sql_to_execute: null }))]);
    const rows = timelineRows({ seen, mode: "query", running: true, failedStage: null, result: null });
    expect(states(rows)).toMatchObject({
      guardrails: "blocked", executing: "skipped", sanity: "skipped", backtranslate: "skipped", agreement: "skipped", confidence: "active",
    });
    expect(rows.find((r) => r.stage === "guardrails")?.summary).toBe("rejected · statement_type");
    expect(headline(rows, blockedResult())?.text).toBe("Blocked at Guardrail");
  });

  it("starts at the guardrail for user SQL", () => {
    const rows = timelineRows({ seen: {}, mode: "run", running: true, failedStage: null, result: null });
    expect(states(rows)).toMatchObject({ generating: "skipped", clarification: "skipped", guardrails: "active" });
  });

  it("marks the failed stage and skips everything after it", () => {
    resetClock();
    const rows = timelineRows({ seen: seenOf([event(generating())]), mode: "query", running: false, failedStage: "executing", result: null });
    expect(states(rows)).toMatchObject({ generating: "done", executing: "failed", sanity: "skipped", done: "skipped" });
  });

  it("derives a replayed result's path without timings", () => {
    const rows = timelineRows({ seen: {}, mode: "run", running: false, failedStage: null, result: blockedResult(), replayed: true });
    expect(states(rows)).toMatchObject({ guardrails: "blocked", executing: "skipped", confidence: "skipped", done: "blocked" });
    expect(rows.every((r) => r.event === undefined)).toBe(true);
  });

  it("shows every stage as still to come before anything has run", () => {
    const rows = timelineRows({ seen: {}, mode: "query", running: false, failedStage: null, result: null });
    expect(rows.every((r) => r.state === "pending" && r.summary === "")).toBe(true);
  });

  it("does not give a low-confidence answer a pass", () => {
    const events = normalRun();
    const low = result({ confidence: 0.04, confidence_band: "low" });
    events[events.length - 1] = event(low);
    const seen = Object.fromEntries(events.map((e) => [e.stage, e])) as Partial<Record<Stage, StageEvent>>;
    const rows = timelineRows({ seen, mode: "query", running: false, failedStage: null, result: null });
    expect(rows.find((r) => r.stage === "done")).toMatchObject({ state: "warn", summary: "answered · low confidence" });
    expect(headline(rows, low)).toEqual({ text: "Answered · low confidence", tone: "warn" });
  });
});
