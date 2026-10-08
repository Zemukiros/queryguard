import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "./client";
import { SseParser, streamEvents } from "./sse";

describe("SseParser", () => {
  it("parses events split across arbitrary chunk boundaries", () => {
    const wire = "event: generating\ndata: {\"a\":1}\n\nevent: done\ndata: {\"b\":2}\n\n";
    for (let size = 1; size <= wire.length; size++) {
      const parser = new SseParser();
      const got = [];
      for (let i = 0; i < wire.length; i += size) got.push(...parser.push(wire.slice(i, i + size)));
      expect(got, `chunk size ${size}`).toEqual([
        { event: "generating", data: "{\"a\":1}" },
        { event: "done", data: "{\"b\":2}" },
      ]);
    }
  });

  it("accepts CRLF and lone CR line endings, including a CRLF split between chunks", () => {
    const parser = new SseParser();
    expect(parser.push("event: a\r")).toEqual([]);
    expect(parser.push("\ndata: 1\r\n\r")).toEqual([]);
    // The final CR could be half of a CRLF, so event b waits for the next chunk (or flush).
    expect(parser.push("\nevent: b\rdata: 2\r\r")).toEqual([{ event: "a", data: "1" }]);
    expect(parser.flush()).toEqual([{ event: "b", data: "2" }]);
  });

  it("joins multi-line data, ignores comments, and strips only one leading space", () => {
    const parser = new SseParser();
    expect(parser.push(": keep-alive\ndata: line one\ndata:  two spaces\ndata\n\n")).toEqual([
      { event: "message", data: "line one\n two spaces\n" },
    ]);
  });

  it("dispatches a trailing event that has no closing blank line on flush", () => {
    const parser = new SseParser();
    expect(parser.push("event: done\ndata: {}")).toEqual([]);
    expect(parser.flush()).toEqual([{ event: "done", data: "{}" }]);
  });

  it("drops an event with no data", () => {
    const parser = new SseParser();
    expect(parser.push("event: lonely\n\n")).toEqual([]);
  });
});

async function drain(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const out: unknown[] = [];
  for await (const item of stream) out.push(item);
  return out;
}

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

describe("streamEvents", () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it("yields parsed StageEvents from a POSTed stream", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(new Response(streamOf([
      "event: guardrails\ndata: {\"stage\":\"guardrails\",\"elapsed_ms\":1}\n",
      "\nevent: done\ndata: {\"stage\":\"done\",\"elapsed_ms\":2}\n\n",
    ]), { status: 200, headers: { "content-type": "text/event-stream" } })));
    vi.stubGlobal("fetch", fetchMock);

    const stages = [];
    for await (const e of streamEvents("/v1/run/stream", { question: "q", sql: "SELECT 1" })) stages.push(e.stage);

    expect(stages).toEqual(["guardrails", "done"]);
    const [path, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(path).toBe("/v1/run/stream");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ question: "q", sql: "SELECT 1" });
  });

  it("turns 429 into an ApiError carrying Retry-After", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify({ detail: "rate limit" }), {
      status: 429, headers: { "retry-after": "42", "content-type": "application/json" },
    }))));
    const consume = () => drain(streamEvents("/v1/query/stream", {}));
    await expect(consume()).rejects.toMatchObject({ status: 429, retryAfter: 42, message: "rate limit" });
    await expect(consume()).rejects.toBeInstanceOf(ApiError);
  });

  it("stops when aborted", async () => {
    const controller = new AbortController();
    vi.stubGlobal("fetch", vi.fn((_: string, init: RequestInit) => new Promise((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => { reject(new DOMException("aborted", "AbortError")); });
    })));
    const consume = drain(streamEvents("/v1/query/stream", {}, controller.signal));
    controller.abort();
    await expect(consume).rejects.toMatchObject({ name: "AbortError" });
  });
});
