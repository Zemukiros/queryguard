/**
 * One question at a time: start a stream, fold its StageEvents into state,
 * abort the previous stream when a new one starts.
 */
import { useCallback, useEffect, useReducer, useRef } from "react";

import { api, ApiError, NetworkError, type PayloadOf, type QueryResult, type Stage, type StageEvent } from "../api/client";
import { streamQuery, streamRun } from "../api/sse";
import type { Mode } from "../lib/stages";

export type Failure =
  | { kind: "http"; status: number; message: string; retryAfter: number | null }
  | { kind: "network"; message: string }
  | { kind: "stage"; stage: Stage; errorType: string; message: string };

export interface RunState {
  /** Increments on every start or load, so the UI can tell runs apart. */
  runId: number;
  status: "idle" | "running" | "done" | "failed";
  mode: Mode;
  question: string;
  seen: Partial<Record<Stage, StageEvent>>;
  order: Stage[];
  result: QueryResult | null;
  failure: Failure | null;
  /** The SQL sent to /v1/run, shown while its stream is in flight. */
  submittedSql: string | null;
  /** Date.now() when the run started, for the live timer. */
  startedAt: number | null;
  /** Where a result came from when its stages were not streamed live. */
  replayed: "cache" | "history" | null;
}

export const initialRunState: RunState = {
  runId: 0, status: "idle", mode: "query", question: "", seen: {}, order: [], result: null, failure: null,
  submittedSql: null, startedAt: null, replayed: null,
};

type Action =
  | { type: "start"; mode: Mode; question: string; sql: string | null; at?: number }
  | { type: "event"; event: StageEvent }
  | { type: "fail"; failure: Failure }
  | { type: "load"; result: QueryResult };

export function runReducer(state: RunState, action: Action): RunState {
  switch (action.type) {
    case "start":
      return { ...initialRunState, runId: state.runId + 1, status: "running", mode: action.mode, question: action.question, submittedSql: action.sql, startedAt: action.at ?? null };
    case "event": {
      const { event } = action;
      const seen = { ...state.seen, [event.stage]: event };
      const order = [...state.order, event.stage];
      if (event.payload.stage === "done") {
        const result = event.payload;
        return { ...state, seen, order, status: "done", result, replayed: result.cached ? "cache" : null };
      }
      if (event.payload.stage === "error") {
        const p: PayloadOf<"error"> = event.payload;
        return { ...state, seen, order, status: "failed", failure: { kind: "stage", stage: p.failed_stage, errorType: p.error_type, message: p.message } };
      }
      return { ...state, seen, order };
    }
    case "fail":
      return { ...state, status: "failed", failure: action.failure };
    case "load":
      return {
        ...initialRunState, runId: state.runId + 1, status: "done", mode: action.result.sql_source === "user" ? "run" : "query",
        question: action.result.question, result: action.result, replayed: "history",
      };
  }
}

function toFailure(error: unknown): Failure {
  if (error instanceof ApiError) return { kind: "http", status: error.status, message: error.message, retryAfter: error.retryAfter };
  if (error instanceof NetworkError) return { kind: "network", message: error.message };
  return { kind: "network", message: error instanceof Error ? error.message : String(error) };
}

export function useQueryRun(onFinished?: () => void) {
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  const controller = useRef<AbortController | null>(null);
  const finished = useRef(onFinished);
  useEffect(() => {
    finished.current = onFinished;
  }, [onFinished]);
  useEffect(() => () => controller.current?.abort(), []);

  const consume = useCallback(async (mode: Mode, question: string, sql: string | null, events: (signal: AbortSignal) => AsyncGenerator<StageEvent>) => {
    controller.current?.abort();
    const mine = new AbortController();
    controller.current = mine;
    dispatch({ type: "start", mode, question, sql, at: Date.now() });
    try {
      for await (const event of events(mine.signal)) {
        if (mine.signal.aborted) return;
        dispatch({ type: "event", event });
      }
    } catch (error) {
      if (mine.signal.aborted || (error instanceof DOMException && error.name === "AbortError")) return;
      dispatch({ type: "fail", failure: toFailure(error) });
    } finally {
      if (controller.current === mine) {
        controller.current = null;
        finished.current?.();
      }
    }
  }, []);

  const ask = useCallback((question: string) => consume("query", question, null, (signal) => streamQuery({ question }, signal)), [consume]);
  const run = useCallback((question: string, sql: string, source: "user" | "reading" = "user") =>
    consume("run", question, sql, (signal) => streamRun({ question, sql, source }, signal)), [consume]);
  /** Reopen a past result from history. */
  const open = useCallback(async (queryId: string) => {
    controller.current?.abort();
    try {
      dispatch({ type: "load", result: await api.historyItem(queryId) });
    } catch (error) {
      dispatch({ type: "fail", failure: toFailure(error) });
    }
  }, []);

  return { state, ask, run, open };
}
