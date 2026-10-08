import { useEffect, useState } from "react";

import { cx } from "../lib/cx";
import { formatMs } from "../lib/format";
import { headline, STAGE_LABEL, type RowState, type TimelineRow } from "../lib/stages";
import type { QueryResult } from "../api/client";
import { Alert, Check, Cross, Minus, Ring, Shield } from "./icons";
import { Chip, Panel } from "./ui";

const STATE_TEXT: Record<RowState, string> = {
  pending: "waiting", active: "running", done: "passed", warn: "ran, with a warning", blocked: "stopped the answer",
  skipped: "skipped", failed: "failed",
};

function StateIcon({ state, muted }: { state: RowState; muted: boolean }) {
  const base = "relative z-10 grid size-[22px] place-items-center rounded-full";
  switch (state) {
    case "done": return <span className={cx(base, muted ? "bg-surface-3 text-ink-2" : "bg-ok-soft text-ok")}><Check size={13} /></span>;
    case "warn": return <span className={cx(base, "bg-warn-soft text-warn")}><Alert size={12} /></span>;
    case "blocked": return <span className={cx(base, "bg-fail-soft text-fail")}><Shield size={12} /></span>;
    case "failed": return <span className={cx(base, "bg-fail-soft text-fail")}><Cross size={13} /></span>;
    case "active": return <span className={cx(base, "bg-accent-fill text-accent-fill-ink")}><span className="qg-spin block size-2.5 rounded-full border-2 border-current border-r-transparent" /></span>;
    case "skipped": return <span className={cx(base, "bg-surface text-ink-3")}><Minus size={13} /></span>;
    case "pending": return <span className={cx(base, "bg-surface text-ink-3")}><Ring size={13} /></span>;
  }
}

function useElapsed(running: boolean, startedAt: number | null): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(() => { setNow(Date.now()); }, 100);
    return () => { clearInterval(timer); };
  }, [running]);
  return running && startedAt !== null ? Math.max(0, now - startedAt) : null;
}

/**
 * One row per stage, in pipeline order, lit as each StageEvent arrives. Every
 * run keeps the same nine rows, so runs can be compared at a glance; a stage
 * that could not happen on this path says "skipped" instead of disappearing.
 */
export function PipelineTimeline({ rows, replayed, running, startedAt, result }: {
  rows: TimelineRow[]; replayed: "cache" | "history" | null; running: boolean; startedAt: number | null; result: QueryResult | null;
}) {
  const live = useElapsed(running, startedAt);
  const verdict = headline(rows, result);
  const total = rows.find((r) => r.stage === "done")?.event?.elapsed_ms;

  let aside: React.ReactNode = null;
  if (running) aside = <Chip tone="accent"><span className="qg-pulse inline-block size-1.5 rounded-full bg-accent" /> Running · <span className="font-mono tabular-nums">{formatMs(live)}</span></Chip>;
  else if (verdict) aside = (
    <span className="flex flex-wrap items-center justify-end gap-1.5">
      {replayed && <Chip>{replayed === "cache" ? "Cached" : "From history"}</Chip>}
      <Chip tone={verdict.tone}>{verdict.text}</Chip>
      {total !== undefined && !replayed && <span className="font-mono text-[11px] text-ink-2">{formatMs(total)}</span>}
    </span>
  );

  return (
    <Panel id="timeline" title="Pipeline" aside={aside} className="border-line-strong">
      <ol className="relative" data-testid="timeline" aria-live="polite">
        {rows.map((row, i) => (
          <li key={row.stage} data-stage={row.stage} data-state={row.state}
            className={cx("relative grid grid-cols-[22px_minmax(0,1fr)_auto] items-center gap-x-3 rounded-md px-1.5 py-[5px]",
              row.state === "active" && "bg-accent-soft")}>
            {i < rows.length - 1 && <span aria-hidden="true" className="absolute top-[24px] left-[16px] h-[calc(100%-8px)] w-0.5 bg-rail" />}
            <StateIcon state={row.state} muted={replayed !== null} />
            <div className="min-w-0 truncate">
              <span className={cx("text-[13px] font-medium",
                row.state === "pending" || row.state === "skipped" ? "text-ink-3" : row.state === "blocked" || row.state === "failed" ? "text-fail" : "text-ink")}>
                {STAGE_LABEL[row.stage]}
              </span>
              <span className="sr-only">: {STATE_TEXT[row.state]}.</span>
              {row.summary ? <span className={cx("ml-2 text-[12px]", row.state === "warn" ? "text-warn" : "text-ink-2")}>{row.summary}</span>
                : row.state === "skipped" ? <span className="ml-2 text-[12px] text-ink-3">skipped</span> : null}
            </div>
            <span className="text-right font-mono text-[11px] tabular-nums text-ink-2"
              title={row.event ? `finished ${formatMs(row.event.elapsed_ms)} after the question was asked` : undefined}>
              {row.event ? formatMs(row.event.duration_ms) : ""}
            </span>
          </li>
        ))}
      </ol>
    </Panel>
  );
}
