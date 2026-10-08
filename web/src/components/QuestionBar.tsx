import { EXAMPLES, type Example } from "../lib/examples";
import { Play } from "./icons";
import { Button } from "./ui";

export function QuestionBar({ value, onChange, onAsk, onExample, running }: {
  value: string; onChange: (v: string) => void; onAsk: (q: string) => void; onExample: (e: Example) => void; running: boolean;
}) {
  return (
    <div>
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (value.trim()) onAsk(value.trim()); }}>
        <label htmlFor="question" className="sr-only">Ask a question about the store's data</label>
        <input id="question" value={value} onChange={(e) => { onChange(e.target.value); }} maxLength={500} autoComplete="off"
          placeholder="Ask about orders, customers, products or refunds…"
          className="min-w-0 flex-1 rounded-md border border-line-strong bg-surface px-3 py-2 text-[14px] text-ink placeholder:text-ink-3" />
        <Button type="submit" variant="primary" disabled={!value.trim()} className="w-[84px]"
          title={running ? "Stops the current run and asks this instead" : undefined}>
          <Play size={13} /> Ask
        </Button>
      </form>
      <div className="mt-2 flex items-center gap-1.5">
        <span className="shrink-0 text-[12px] text-ink-2">Try:</span>
        <div data-testid="examples" className="-mr-4 flex min-w-0 flex-1 items-center gap-1.5 overflow-x-auto pr-4 pb-1 sm:mr-0 sm:flex-wrap sm:overflow-visible sm:pr-0 sm:pb-0">
        {EXAMPLES.map((example) => (
          <button key={example.label} type="button" onClick={() => { onExample(example); }}
            className="shrink-0 whitespace-nowrap rounded-full border border-line bg-surface px-2.5 py-0.5 text-[12px] text-ink-2 hover:border-line-strong hover:text-ink">
            {example.label}
          </button>
        ))}
        </div>
      </div>
    </div>
  );
}
