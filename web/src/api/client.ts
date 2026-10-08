/**
 * Typed access to the QueryGuard API. Every type comes from schema.ts, which
 * is generated from the FastAPI OpenAPI schema (`npm run gen:api`); nothing
 * here restates a server shape by hand.
 */
import type { components } from "./schema";

export type Schemas = components["schemas"];
export type QueryResult = Schemas["QueryResult"];
export type StageEvent = Schemas["StageEvent"];
export type Stage = StageEvent["stage"];
export type HistoryItem = Schemas["HistoryItem"];
export type Health = Schemas["Health"];
export type SchemaResponse = Schemas["SchemaResponse"];
export type FeedbackRequest = Schemas["FeedbackRequest"];
export type FeedbackResponse = Schemas["FeedbackResponse"];
export type QueryRequest = Schemas["QueryRequest"];
export type RunRequest = Schemas["RunRequest"];

/** Narrow a StageEvent's payload by its stage. */
export type PayloadOf<S extends Stage> = Extract<StageEvent["payload"], { stage: S }>;

/** A non-2xx response. `retryAfter` is set for 429 and 503. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The server could not be reached at all. */
export class NetworkError extends Error {
  constructor(cause: unknown) {
    super("The QueryGuard API could not be reached.", { cause });
    this.name = "NetworkError";
  }
}

function detailText(body: unknown, fallback: string): string {
  if (body && typeof body === "object") {
    const record = body as Record<string, unknown>;
    if (typeof record.detail === "string") return record.detail;
    if (typeof record.message === "string") return record.message;
    if (Array.isArray(record.detail)) return "The request was not valid.";
  }
  return fallback;
}

export async function toApiError(response: Response): Promise<ApiError> {
  const raw = response.headers.get("retry-after");
  const retryAfter = raw !== null && /^\d+$/.test(raw) ? Number(raw) : null;
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // not JSON; keep the status text
  }
  return new ApiError(response.status, detailText(body, response.statusText || `HTTP ${response.status}`), retryAfter);
}

export async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new NetworkError(error);
  }
  if (!response.ok) throw await toApiError(response);
  return response;
}

async function getJson<T>(path: string): Promise<T> {
  return (await (await apiFetch(path)).json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return (await response.json()) as T;
}

export const api = {
  health: () => getJson<Health>("/healthz"),
  schema: () => getJson<SchemaResponse>("/v1/schema"),
  history: (limit = 30) => getJson<HistoryItem[]>(`/v1/history?limit=${limit}`),
  historyItem: (queryId: string) => getJson<QueryResult>(`/v1/history/${encodeURIComponent(queryId)}`),
  feedback: (body: FeedbackRequest) => postJson<FeedbackResponse>("/v1/feedback", body),
};
