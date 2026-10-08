import type { QueryResult, Schemas } from "../api/client";
import { BAND } from "../lib/bands";
import { formatSigned } from "../lib/format";
import { Chip, Panel } from "./ui";

type Contribution = Schemas["Contribution"];


/**
 * A diverging bar per signal: blue raises the score, orange lowers it, scaled
 * to the largest term. The baseline (the intercept) is a starting point, not a
 * signal, so it is stated above the bars instead of drawn as one.
 */
export function ContributionBars({ terms }: { terms: Contribution[] }) {
  const baseline = terms.find((t) => t.feature === "bias");
  const signals = terms.filter((t) => t.feature !== "bias");
  const active = signals.filter((t) => Math.abs(t.contribution) > 1e-9).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const idle = signals.filter((t) => Math.abs(t.contribution) <= 1e-9);
  const max = Math.max(...active.map((t) => Math.abs(t.contribution)), 1e-9);
  return (
    <div>
      {baseline && (
        <p className="mb-1.5 text-[12px] text-ink-2">Starting point <span className="font-mono text-ink">{formatSigned(baseline.contribution)}</span>, then:</p>
      )}
      <ul className="space-y-1" data-testid="contributions">
        {active.map((t) => {
          const width = `${(Math.abs(t.contribution) / max) * 50}%`;
          const raises = t.contribution > 0;
          return (
            <li key={t.feature} className="grid grid-cols-[minmax(0,1.5fr)_minmax(64px,1fr)_44px] items-center gap-2 text-[12px]"
              title={`${t.label}: weight ${t.weight.toFixed(3)} × value ${t.value.toFixed(3)}`}>
              <span className="leading-tight text-ink-2">{t.label}</span>
              <span className="relative h-2.5 rounded-sm bg-surface-2" aria-hidden="true">
                <span className={`absolute inset-y-0 rounded-sm ${raises ? "left-1/2 bg-raise" : "right-1/2 bg-lower"}`} style={{ width }} />
                <span className="absolute -inset-y-0.5 left-1/2 w-px -translate-x-px bg-ink-3" />
              </span>
              <span className="text-right font-mono tabular-nums text-ink">{formatSigned(t.contribution)}</span>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-2">
        <span className="inline-flex items-center gap-1"><span className="inline-block size-2 rounded-sm bg-raise" /> raises trust</span>
        <span className="inline-flex items-center gap-1"><span className="inline-block size-2 rounded-sm bg-lower" /> lowers trust</span>
        {idle.length > 0 && (
          <span title={idle.map((t) => t.label).join(", ")}>{idle.length} other signal{idle.length === 1 ? "" : "s"} had no effect</span>
        )}
      </p>
    </div>
  );
}

export function ConfidenceCard({ result }: { result: QueryResult }) {
  if (result.confidence === null) return null;
  if (result.confidence_logit === null) {
    return (
      <Panel id="confidence" title="Confidence">
        <p className="font-mono text-[28px] font-semibold leading-none text-ink-3" data-testid="confidence-score">—</p>
        <p className="mt-2 text-[12.5px] text-ink-2">No answer to score: nothing executed.</p>
      </Panel>
    );
  }
  const band = result.confidence_band ? BAND[result.confidence_band] : null;
  return (
    <Panel id="confidence" title="Confidence" aside={band && <Chip tone={band.tone}>{band.text}</Chip>}>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[28px] font-semibold tabular-nums leading-none text-ink" data-testid="confidence-score">
          {result.confidence.toFixed(2)}
        </span>
        <span className="text-[12px] text-ink-2">probability the answer is right</span>
      </div>
      <div className="mt-3">
        <p className="mb-1.5 text-[12px] font-medium text-ink">Why: each signal's push on the score (log-odds)</p>
        <ContributionBars terms={result.contributions} />
      </div>
      <p className="mt-3 border-t border-line pt-2 text-[11px] text-ink-2">
        Logistic model fitted on the live eval run · <span className="font-mono">{result.scorer_version}</span>
      </p>
    </Panel>
  );
}
