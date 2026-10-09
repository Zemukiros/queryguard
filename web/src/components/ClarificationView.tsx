import type { QueryResult } from "../api/client";
import { Play } from "./icons";
import { Button, Panel } from "./ui";

/** The question has more than one defensible reading; each is runnable SQL. Picking one runs it. */
export function ClarificationView({ result, onPick, running }: {
  result: QueryResult; onPick: (sql: string) => void; running: boolean;
}) {
  return (
    <Panel id="clarification" title="Which did you mean?">
      <p className="mb-3 text-[13px] text-ink-2">
        This question has {result.interpretations.length} defensible readings. QueryGuard asks instead of guessing.
        Pick one and it runs through the same checks.
      </p>
      <div className="grid gap-2.5 md:grid-cols-2" data-testid="interpretations">
        {result.interpretations.map((reading) => (
          <article key={reading.label} className="flex flex-col rounded-md border border-line bg-surface-2/60 p-3">
            <h3 className="font-mono text-[12px] font-semibold text-accent-strong">{reading.label}</h3>
            <p className="mt-1 flex-1 text-[12.5px] text-ink-2">{reading.explanation}</p>
            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-surface px-2 py-1.5 font-mono text-[11px] leading-4 text-ink-2">{reading.sql}</pre>
            <Button className="mt-2 self-start" onClick={() => { onPick(reading.sql); }} disabled={running}
              aria-label={`Run the ${reading.label} reading`}>
              <Play size={12} /> Run this reading
            </Button>
          </article>
        ))}
      </div>
    </Panel>
  );
}
