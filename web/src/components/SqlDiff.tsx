import { diffLines } from "diff";

/** Line diff, old → new. Additions and removals carry a +/− marker, not colour alone. */
export function SqlDiff({ before, after, beforeLabel, afterLabel }: {
  before: string; after: string; beforeLabel: string; afterLabel: string;
}) {
  const parts = diffLines(before.trimEnd() + "\n", after.trimEnd() + "\n");
  return (
    <div className="overflow-hidden rounded-md border border-line">
      <div className="flex gap-4 border-b border-line bg-surface-2 px-3 py-1 text-[11px] text-ink-2">
        <span><span className="font-mono text-fail">−</span> {beforeLabel}</span>
        <span><span className="font-mono text-ok">+</span> {afterLabel}</span>
      </div>
      <pre className="whitespace-pre-wrap break-words py-1 font-mono text-[12px] leading-5">
        {parts.flatMap((part, i) =>
          part.value.replace(/\n$/, "").split("\n").map((line, j) => (
            <div key={`${i}-${j}`}
              className={part.added ? "bg-diff-add" : part.removed ? "bg-diff-del" : ""}>
              <span className="inline-block w-6 select-none text-center text-ink-3" aria-hidden="true">
                {part.added ? "+" : part.removed ? "−" : " "}
              </span>
              <span className="sr-only">{part.added ? "added: " : part.removed ? "removed: " : ""}</span>
              {line || " "}
            </div>
          )),
        )}
      </pre>
    </div>
  );
}
