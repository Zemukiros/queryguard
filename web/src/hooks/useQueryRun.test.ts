import { describe, expect, it } from "vitest";

import { event, normalRun, resetClock, result } from "../test/fixtures";
import { initialRunState, runReducer } from "./useQueryRun";

describe("runReducer", () => {
  it("folds a stream into state and finishes on done", () => {
    let state = runReducer(initialRunState, { type: "start", mode: "query", question: "q", sql: null });
    expect(state).toMatchObject({ status: "running", runId: 1 });
    for (const e of normalRun()) state = runReducer(state, { type: "event", event: e });
    expect(state.status).toBe("done");
    expect(state.order).toHaveLength(8);
    expect(state.result?.outcome).toBe("answered");
    expect(state.replayed).toBeNull();
  });

  it("records a stage error as a failure naming the stage", () => {
    resetClock();
    let state = runReducer(initialRunState, { type: "start", mode: "query", question: "q", sql: null });
    state = runReducer(state, { type: "event", event: event({ stage: "error", failed_stage: "generating", error_type: "APIError", message: "try again" }) });
    expect(state).toMatchObject({ status: "failed", failure: { kind: "stage", stage: "generating", message: "try again" } });
  });

  it("marks cache hits and history loads as replayed, and a new start resets", () => {
    resetClock();
    let state = runReducer(initialRunState, { type: "start", mode: "query", question: "q", sql: null });
    state = runReducer(state, { type: "event", event: event(result({ cached: true })) });
    expect(state.replayed).toBe("cache");
    state = runReducer(state, { type: "load", result: result({ sql_source: "user" }) });
    expect(state).toMatchObject({ replayed: "history", mode: "run", runId: 2 });
    state = runReducer(state, { type: "start", mode: "run", question: "q", sql: "SELECT 1" });
    expect(state).toMatchObject({ status: "running", result: null, submittedSql: "SELECT 1", runId: 3 });
  });
});
