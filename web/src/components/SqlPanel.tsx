import { lazy, Suspense } from "react";

import type { PayloadOf, QueryResult, StageEvent } from "../api/client";
import { sentence } from "../lib/format";
import { Play, Shield } from "./icons";
import { SqlDiff } from "./SqlDiff";
import { Button, Chip, Mono, Panel, type Tone } from "./ui";

// CodeMirror is most of the bundle; the page paints first and the editor follows.
const SqlEditor = lazy(() => import("./SqlEditor").then((m) => ({ default: m.SqlEditor })));

const SOURCE: Record<QueryResult["sql_source"], { tone: Tone; text: string }> = {
  model: { tone: "neutral", text: "written by the model" },
  user: { tone: "accent", text: "your SQL" },
  reading: { tone: "info", text: "chosen reading" },
};


export function GuardrailRejection({ rule, reason }: { rule: string | null; reason: string | null }) {
  return (
    <div role="alert" data-testid="guardrail-rejection" className="mb-3 rounded-md border border-fail/40 bg-fail-soft px-3 py-2.5">
      <p className="flex items-center gap-2 text-[13px] font-semibold text-fail"><Shield size={15} /> Blocked by the guardrail</p>
      <p className="mt-1 text-[12.5px] text-ink-2">Rule: <Mono className="rounded bg-surface px-1 py-px text-fail">{rule ?? "unknown"}</Mono></p>
      {reason && <p className="mt-1 text-[13px] text-ink">{sentence(reason)}</p>}
      <p className="mt-1 text-[12px] text-ink-2">Nothing was executed. Even if it had been, the read-only database role cannot write.</p>
    </div>
  );
}

/** What the guardrail changed. A plain appended row cap gets one line; anything else a diff. */
function Rewrite({ wrote, ran }: { wrote: string; ran: string }) {
  const base = wrote.trim().replace(/;$/, "");
  if (ran.startsWith(base)) {
    const appended = ran.slice(base.length).trim();
    return (
      <details className="mt-3 text-[12.5px] text-ink-2">
        <summary className="cursor-pointer">
          The guardrail appended <Mono className="rounded bg-surface-2 px-1 py-px text-ink">{appended}</Mono>, a row cap, before running it.
        </summary>
        <div className="mt-1.5"><SqlDiff before={wrote} after={ran} beforeLabel="as written" afterLabel="as run" /></div>
      </details>
    );
  }
  return (
    <div className="mt-3">
      <p className="mb-1.5 text-[12px] font-medium text-ink-2">The guardrail rewrote it before running:</p>
      <SqlDiff before={wrote} after={ran} beforeLabel="as written" afterLabel="as run" />
    </div>
  );
}

export function SqlPanel({ result, guardrailEvent, draft, onDraftChange, onRun, running }: {
  result: QueryResult | null;
  guardrailEvent: StageEvent | undefined;
  draft: string;
  onDraftChange: (sql: string) => void;
  onRun: (sql: string) => void;
  running: boolean;
}) {
  const guardrail = guardrailEvent?.payload as PayloadOf<"guardrails"> | undefined;
  const rejected = guardrail ? !guardrail.allowed : result?.outcome === "blocked";
  const rule = guardrail?.rule ?? result?.guardrail_rule ?? null;
  const reason = guardrail?.reason ?? result?.guardrail_reason ?? null;
  const ran = result?.executed_sql ?? null;
  const wrote = result?.sql ?? null;
  const source = result?.sql ? SOURCE[result.sql_source] : null;

  return (
    <Panel id="sql" title="SQL" aside={source && <Chip tone={source.tone}>{source.text}</Chip>}>
      {rejected && <GuardrailRejection rule={rule} reason={reason} />}
      <Suspense fallback={<pre className="min-h-[38px] rounded-md bg-surface-2 px-3 py-2 font-mono text-[12.5px] text-ink-2">{draft || " "}</pre>}>
        <SqlEditor value={draft} onChange={onDraftChange} label="SQL editor" testId="sql-editor"
          placeholder="Write or paste SQL here, then run it through the same checks." />
      </Suspense>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[12px] text-ink-2">Your SQL goes through the same guardrail, read-only executor and checks as the model's.</p>
        <Button variant="primary" onClick={() => { onRun(draft); }} disabled={running || !draft.trim()} data-testid="run-sql">
          <Play size={13} /> Run my SQL
        </Button>
      </div>
      {wrote && ran && ran.trim() !== wrote.trim() && <Rewrite wrote={wrote} ran={ran} />}
    </Panel>
  );
}
