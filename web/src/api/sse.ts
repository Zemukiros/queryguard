/**
 * Server-Sent Events over a POST. EventSource can only GET, so the stream is
 * read with fetch + ReadableStream and parsed here.
 *
 * The parser follows the SSE spec's line rules: CRLF, LF and lone CR all end
 * a line (a CR at the end of a chunk waits for the next chunk in case an LF
 * follows), lines starting with ":" are comments, a single space after the
 * colon is dropped, repeated `data:` lines join with "\n", and a blank line
 * dispatches. One deliberate leniency: `flush()` dispatches a final event
 * that arrived without its closing blank line instead of dropping it.
 */
import { apiFetch, type QueryRequest, type RunRequest, type StageEvent } from "./client";

export interface SseMessage {
  event: string;
  data: string;
}

export class SseParser {
  private buffer = "";
  private event = "";
  private data: string[] = [];

  push(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const out: SseMessage[] = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer[i];
      if (ch !== "\n" && ch !== "\r") continue;
      if (ch === "\r" && i === this.buffer.length - 1) break; // maybe half of CRLF
      const line = this.buffer.slice(start, i);
      if (ch === "\r" && this.buffer[i + 1] === "\n") i++;
      start = i + 1;
      const message = this.line(line);
      if (message) out.push(message);
    }
    this.buffer = this.buffer.slice(start);
    return out;
  }

  flush(): SseMessage[] {
    const out = this.buffer ? this.push("\n") : [];
    const last = this.dispatch();
    return last ? [...out, last] : out;
  }

  private line(line: string): SseMessage | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.event = value;
    else if (field === "data") this.data.push(value);
    return null;
  }

  private dispatch(): SseMessage | null {
    if (this.data.length === 0) {
      this.event = "";
      return null;
    }
    const message = { event: this.event || "message", data: this.data.join("\n") };
    this.event = "";
    this.data = [];
    return message;
  }
}

/** Yield each StageEvent of a streamed POST. Throws ApiError / NetworkError / AbortError. */
export async function* streamEvents(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<StageEvent> {
  const response = await apiFetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal: signal ?? null,
  });
  if (!response.body) throw new Error("the response has no body to stream");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      const messages = done ? [...parser.push(decoder.decode()), ...parser.flush()] : parser.push(decoder.decode(value, { stream: true }));
      for (const message of messages) yield JSON.parse(message.data) as StageEvent;
      if (done) return;
    }
  } finally {
    // Stopping early (a new question) must also close the connection.
    await reader.cancel().catch(() => undefined);
  }
}

export const streamQuery = (body: QueryRequest, signal?: AbortSignal) => streamEvents("/v1/query/stream", body, signal);
export const streamRun = (body: RunRequest, signal?: AbortSignal) => streamEvents("/v1/run/stream", body, signal);
