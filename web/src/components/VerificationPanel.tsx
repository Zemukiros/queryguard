import type { QueryResult } from "../api/client";
import { sentence } from "../lib/format";
import { SqlDiff } from "./SqlDiff";
import { Chip, Panel, type Tone } from "./ui";

const AGREEMENT: Record<"agree" | "disagree" | "incomparable", { tone: Tone; text: string }> = {
  agree: { tone: "ok", text: "Agreed" },
  disagree: { tone: "fail", text: "Disagreed" },
  incomparable: { tone: "warn", text: "Incomparable" },
};
const SEVERITY: Record<"fail" | "warn" | "info", Tone> = { fail: "fail", warn: "warn", info: "info" };

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="border-t border-line pt-3 first:border-t-0 first:pt-0">
      <h3 className="mb-1.5 text-[12px] font-semibold text-ink">{title}</h3>
      {children}
    </div>
  );
}

export function VerificationPanel({ result }: { result: QueryResult }) {
  if (result.executed_sql === null) return null;
  // Amber, not red: a low alignment lowers trust, it does not block the answer (the timeline agrees).
  const alignmentTone: Tone = result.alignment === null ? "neutral" : result.alignment >= 0.7 ? "ok" : "warn";
  return (
    <Panel id="verification" title="Verification">
      <div className="space-y-3" data-testid="verification">
        <Section title="Blind back-translation">
          <dl className="grid grid-cols-[88px_minmax(0,1fr)] gap-x-2 gap-y-1 text-[12.5px]">
            <dt className="text-ink-2">Asked</dt><dd className="text-ink">{result.question}</dd>
            <dt className="text-ink-2">SQL reads as</dt>
            <dd className="text-ink">{result.back_translation ?? <span className="text-ink-3">not available</span>}</dd>
          </dl>
          <div className="mt-1.5 flex items-center gap-2">
            <Chip tone={alignmentTone}>alignment {result.alignment === null ? "n/a" : result.alignment.toFixed(2)}</Chip>
            <span className="text-[11px] text-ink-2">The translator never saw the question.</span>
          </div>
          {result.discrepancies.length > 0 && (
            <ul className="mt-1.5 list-disc space-y-0.5 pl-4 text-[12.5px] text-ink-2">
              {result.discrepancies.map((d) => <li key={d}>{sentence(d)}</li>)}
            </ul>
          )}
        </Section>

        <Section title="Independent second query">
          {result.agreement === null ? (
            <p className="text-[12.5px] text-ink-3">Not run: a query with no join, aggregate, CTE, subquery or top-N has nothing to disagree on.</p>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <Chip tone={AGREEMENT[result.agreement].tone}>{AGREEMENT[result.agreement].text}</Chip>
                {result.agreement_explanation && <span className="text-[12.5px] text-ink-2">{sentence(result.agreement_explanation)}</span>}
              </div>
              {result.second_sql && result.sql && (
                <details className="mt-2">
                  <summary className="cursor-pointer text-[12px] text-accent-strong">Compare the two queries</summary>
                  <div className="mt-1.5">
                    <SqlDiff before={result.sql} after={result.second_sql} beforeLabel="first query" afterLabel="second query" />
                  </div>
                </details>
              )}
            </>
          )}
        </Section>

        <Section title="Result sanity checks">
          {result.sanity.length === 0 ? (
            <p className="text-[12.5px] text-ink-3">No suspicious shapes in the result.</p>
          ) : (
            <ul className="space-y-1.5">
              {result.sanity.map((f) => (
                <li key={`${f.check}-${f.column ?? ""}`} className="flex gap-2 text-[12.5px]">
                  <Chip tone={SEVERITY[f.severity]} className="shrink-0 uppercase">{f.severity}</Chip>
                  <span className="text-ink-2"><span className="font-mono text-[11.5px] text-ink">{f.check}</span>{f.column ? ` (${f.column})` : ""}: {sentence(f.explanation)}</span>
                </li>
              ))}
            </ul>
          )}
        </Section>

        {result.validation_errors.length > 0 && (
          <Section title="Checks that could not run">
            <ul className="space-y-1 text-[12px] text-warn">{result.validation_errors.map((e) => <li key={e}>{e}</li>)}</ul>
          </Section>
        )}
      </div>
    </Panel>
  );
}
