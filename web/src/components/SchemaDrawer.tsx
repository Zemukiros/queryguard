import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";

import { api } from "../api/client";
import { Cross } from "./icons";
import { Button, Chip } from "./ui";

export function SchemaDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const schema = useQuery({ queryKey: ["schema"], queryFn: api.schema, enabled: open, staleTime: Infinity });
  const [filter, setFilter] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const el = dialog.current;
    if (!el) return;
    if (open && !el.open) el.showModal();
    if (!open && el.open) el.close();
  }, [open]);

  const tables = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const all = schema.data?.tables ?? [];
    if (!needle) return all;
    return all.filter((t) => t.name.includes(needle) || (t.columns ?? []).some((c) => c.name.includes(needle)));
  }, [schema.data, filter]);

  return (
    <dialog ref={dialog} onClose={onClose} aria-labelledby="schema-title"
      className="m-0 ml-auto h-dvh max-h-dvh w-full max-w-[520px] border-l border-line bg-surface p-0 text-ink backdrop:bg-scrim">
      <div className="flex h-full flex-col">
        <header className="flex items-center justify-between gap-2 border-b border-line px-4 py-3">
          <div>
            <h2 id="schema-title" className="font-serif text-[20px] font-normal">Database schema</h2>
            <p className="text-[12px] text-ink-2">Exactly what the model sees. Read-only.</p>
          </div>
          <Button variant="ghost" onClick={onClose} aria-label="Close schema"><Cross size={16} /></Button>
        </header>
        <div className="border-b border-line px-4 py-2">
          <label className="sr-only" htmlFor="schema-filter">Filter tables and columns</label>
          <input id="schema-filter" value={filter} onChange={(e) => { setFilter(e.target.value); }} placeholder="Filter tables and columns"
            className="w-full rounded-md border border-line-strong bg-surface px-2.5 py-1.5 text-[13px] placeholder:text-ink-3" />
        </div>
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-3">
          {schema.isPending && <p className="text-[13px] text-ink-3">Loading…</p>}
          {schema.isError && <p className="text-[13px] text-fail">The schema could not be loaded.</p>}
          {tables.map((table) => (
            <section key={table.name}>
              <h3 className="flex items-baseline gap-2 font-mono text-[13px] font-semibold">
                {table.name}
                <span className="font-sans text-[11px] font-normal text-ink-3">{table.row_count.toLocaleString("en-US")} rows</span>
              </h3>
              {table.comment && <p className="mt-0.5 text-[12px] text-ink-2">{table.comment}</p>}
              <table className="mt-1.5 w-full table-fixed text-left text-[12px]">
                <colgroup><col className="w-[38%]" /><col className="w-[26%]" /><col /></colgroup>
                <tbody>
                  {(table.columns ?? []).map((col) => (
                    <tr key={col.name} className="border-t border-line align-top">
                      <td className="py-1 pr-2">
                        <span className="break-all font-mono text-ink">{col.name}</span>
                        {col.is_primary_key && <Chip tone="accent" className="ml-1.5">PK</Chip>}
                      </td>
                      <td className="py-1 pr-2">
                        <span className="whitespace-nowrap font-mono text-[11px] text-ink-2">{col.sql_type}</span>
                        {!col.nullable && <span className="block text-[10px] font-medium uppercase tracking-wide text-ink-3">not null</span>}
                      </td>
                      <td className="py-1 text-ink-2">
                        {col.references && <Chip className="mr-1">→ {col.references}</Chip>}
                        {col.comment}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          ))}
        </div>
      </div>
    </dialog>
  );
}
