import { useMemo, useState } from "react";

import type { QueryResult } from "../api/client";
import { formatCell, formatMs } from "../lib/format";
import { Alert, Sort } from "./icons";
import { Panel } from "./ui";

type Direction = "asc" | "desc";

function compare(a: unknown, b: unknown): number {
  if (a === b) return 0;
  if (a === null || a === undefined) return 1; // NULLs last, either direction
  if (b === null || b === undefined) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return formatCell(a).localeCompare(formatCell(b), "en-US", { numeric: true });
}

/** Numbers align right, except identifiers (customer_id): those are labels, not quantities. */
function numericColumns(result: QueryResult): boolean[] {
  return result.columns.map((name, i) => {
    if (/(^|_)id$/i.test(name)) return false;
    const values = result.rows.map((row) => row[i]).filter((v) => v !== null && v !== undefined);
    return values.length > 0 && values.every((v) => typeof v === "number");
  });
}

const SCROLL_AFTER = 14;

export function ResultsTable({ result }: { result: QueryResult }) {
  const numeric = useMemo(() => numericColumns(result), [result]);
  const [sort, setSort] = useState<{ column: number; dir: Direction } | null>(null);
  const rows = useMemo(() => {
    if (!sort) return result.rows;
    const sign = sort.dir === "asc" ? 1 : -1;
    return [...result.rows].sort((x, y) => {
      const a = x[sort.column], b = y[sort.column];
      if (a === null || a === undefined || b === null || b === undefined) return compare(a, b);
      return sign * compare(a, b);
    });
  }, [result.rows, sort]);

  const toggle = (column: number) => {
    setSort((s) => (s?.column !== column ? { column, dir: "asc" } : s.dir === "asc" ? { column, dir: "desc" } : null));
  };

  return (
    <Panel id="results" title="Results"
      aside={<span className="font-mono text-[11px] text-ink-3" data-testid="row-count">
        {result.row_count.toLocaleString("en-US")} row{result.row_count === 1 ? "" : "s"} · {formatMs(result.execution_ms)}
      </span>}>
      {result.truncated && (
        <p role="status" className="mb-2 flex items-center gap-2 rounded-md bg-warn-soft px-3 py-2 text-[12.5px] text-warn">
          <Alert size={14} /> Truncated: showing the first {result.row_count.toLocaleString("en-US")} rows. The guardrail caps every result.
        </p>
      )}
      {result.columns.length === 0 ? (
        <p className="text-[13px] text-ink-3">No columns.</p>
      ) : (
        <div className="max-h-[402px] overflow-auto rounded-md border border-line">
          <table className="w-full border-collapse text-left text-[12.5px]" data-testid="results-table">
            <thead className="sticky top-0 z-10 bg-surface-2">
              <tr>
                {result.columns.map((name, i) => {
                  const dir = sort?.column === i ? sort.dir : null;
                  return (
                    <th key={`${name}-${i}`} scope="col" aria-sort={dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none"}
                      className="border-b border-line px-0 py-0 font-semibold text-ink-2">
                      <button type="button" onClick={() => { toggle(i); }}
                        className={`flex w-full items-center gap-1 px-2.5 py-1.5 font-mono text-[11.5px] hover:text-ink ${numeric[i] ? "flex-row-reverse text-right" : "text-left"}`}>
                        {name} <Sort dir={dir} size={12} className="text-ink-3" />
                      </button>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr><td colSpan={result.columns.length} className="px-2.5 py-3 text-ink-3">The query ran and matched no rows.</td></tr>
              ) : rows.map((row, r) => (
                <tr key={r} className="odd:bg-surface even:bg-surface-2/50">
                  {row.map((cell, c) => (
                    <td key={c} className={`border-b border-line px-2.5 py-1 font-mono text-[12px] ${numeric[c] ? "text-right tabular-nums" : ""} ${cell === null ? "text-ink-3 italic" : "text-ink"}`}>
                      {formatCell(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.rows.length > SCROLL_AFTER && (
        <p className="mt-1.5 text-right text-[11px] text-ink-2">{result.rows.length.toLocaleString("en-US")} rows · scroll the table for the rest</p>
      )}
    </Panel>
  );
}
