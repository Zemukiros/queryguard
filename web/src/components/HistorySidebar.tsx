import { useQuery } from "@tanstack/react-query";

import { api, type HistoryItem } from "../api/client";
import { timeAgo } from "../lib/format";
import { Chip, type Tone } from "./ui";
import { cx } from "../lib/cx";

const OUTCOME: Record<string, Tone> = {
  answered: "ok", clarification: "info", cannot_answer: "neutral", blocked: "fail",
  refused: "warn", failed: "fail", error: "fail",
};

export function HistorySidebar({ activeId, onOpen }: { activeId: string | null; onOpen: (item: HistoryItem) => void }) {
  const history = useQuery({ queryKey: ["history"], queryFn: () => api.history(40) });
  return (
    <nav aria-label="Your recent questions" className="flex min-h-0 flex-col">
      <h2 className="px-1 pb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-ink-2">Your history</h2>
      {history.isPending && <p className="px-1 text-[12px] text-ink-3">Loading…</p>}
      {history.isError && <p className="px-1 text-[12px] text-ink-3">History is unavailable.</p>}
      {history.data?.length === 0 && <p className="px-1 text-[12px] text-ink-3">Questions you ask appear here. Only you can see them.</p>}
      <ul className="min-h-0 space-y-0.5 overflow-y-auto" data-testid="history">
        {history.data?.map((item) => (
          <li key={item.query_id}>
            <button type="button" onClick={() => { onOpen(item); }} disabled={item.outcome === "error"}
              className={cx("w-full rounded-md px-2 py-1.5 text-left hover:bg-surface-2 disabled:cursor-default disabled:hover:bg-transparent",
                activeId === item.query_id && "bg-surface-2")}>
              <span className="line-clamp-2 text-[12.5px] text-ink">{item.question}</span>
              <span className="mt-0.5 flex items-center gap-2 text-[11px] text-ink-2">
                <span className="flex min-w-0 flex-1 items-center gap-1.5 overflow-hidden whitespace-nowrap">
                  <Chip tone={OUTCOME[item.outcome] ?? "neutral"}>{item.outcome.replace("_", " ")}</Chip>
                  {item.confidence !== null && item.outcome === "answered" && <span className="font-mono">{item.confidence.toFixed(2)}</span>}
                  {item.sql_source === "user" && <span>your SQL</span>}
                  {item.sql_source === "reading" && <span>chosen reading</span>}
                  {item.cached && <span>cached</span>}
                  {item.feedback && <span>{item.feedback.correct ? "👍" : "👎"}<span className="sr-only">{item.feedback.correct ? "marked correct" : "marked wrong"}</span></span>}
                </span>
                <span className="shrink-0">{timeAgo(item.created_at)}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
}
