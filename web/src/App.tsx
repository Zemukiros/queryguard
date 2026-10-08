import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";

import type { PayloadOf } from "./api/client";
import { ClarificationView } from "./components/ClarificationView";
import { ConfidenceCard } from "./components/ConfidenceCard";
import { FeedbackControls } from "./components/FeedbackControls";
import { Header } from "./components/Header";
import { HistorySidebar } from "./components/HistorySidebar";
import { PipelineTimeline } from "./components/PipelineTimeline";
import { QuestionBar } from "./components/QuestionBar";
import { ResultsTable } from "./components/ResultsTable";
import { SchemaDrawer } from "./components/SchemaDrawer";
import { SqlPanel } from "./components/SqlPanel";
import { StatusBanner } from "./components/StatusBanner";
import { Chip, Panel } from "./components/ui";
import { VerificationPanel } from "./components/VerificationPanel";
import { useQueryRun } from "./hooks/useQueryRun";
import { useTheme } from "./hooks/useTheme";
import { BAND } from "./lib/bands";
import type { Example } from "./lib/examples";
import { formatCell, formatMs, formatUsd } from "./lib/format";
import { modeCopy } from "./lib/mode";
import { timelineRows } from "./lib/stages";

const DEFAULT_RUN_QUESTION = "Run my SQL";

export default function App() {
  const queryClient = useQueryClient();
  const refreshHistory = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ["history"] });
    void queryClient.invalidateQueries({ queryKey: ["health"] });
  }, [queryClient]);
  const { state, ask, run, open } = useQueryRun(refreshHistory);
  const { theme, toggle } = useTheme();
  const [question, setQuestion] = useState("");
  const [schemaOpen, setSchemaOpen] = useState(false);
  const [edit, setEdit] = useState<{ runId: number; sql: string } | null>(null);

  const { result } = state;
  const running = state.status === "running";
  const generating = state.seen.generating?.payload as PayloadOf<"generating"> | undefined;
  const modelSql = result?.sql ?? generating?.sql ?? state.submittedSql ?? "";
  const draft = edit && edit.runId === state.runId ? edit.sql : modelSql;

  const rows = timelineRows({
    seen: state.seen, mode: state.mode, running,
    failedStage: state.failure?.kind === "stage" ? state.failure.stage : null,
    result: state.replayed ? result : null,
    replayed: state.replayed !== null,
  });

  const runSql = (sql: string, forQuestion?: string, source: "user" | "reading" = "user") => {
    const q = (forQuestion ?? question).trim() || state.question || DEFAULT_RUN_QUESTION;
    void run(q, sql.trim(), source);
  };
  const onExample = (example: Example) => {
    setQuestion(example.question);
    if (example.sql) runSql(example.sql, example.question);
    else void ask(example.question);
  };

  const showResults = result && result.outcome !== "clarification" && result.outcome !== "cannot_answer";
  const scalar = result?.outcome === "answered" && result.row_count === 1 && result.columns.length === 1 ? result.rows[0]?.[0] : undefined;
  const band = result?.confidence_band && result.confidence_logit !== null ? BAND[result.confidence_band] : null;

  return (
    <div className="min-h-dvh">
      <Header theme={theme} onToggleTheme={toggle} onOpenSchema={() => { setSchemaOpen(true); }} />
      <div className="mx-auto grid max-w-[1600px] gap-4 px-4 py-4 lg:grid-cols-[230px_minmax(0,1fr)] xl:grid-cols-[230px_minmax(0,1fr)_380px]">
        <aside className="sticky top-[68px] hidden max-h-[calc(100dvh-84px)] min-h-0 flex-col lg:flex">
          <HistorySidebar activeId={result?.query_id ?? null} onOpen={(item) => { void open(item.query_id); }} />
        </aside>

        <main className="min-w-0 space-y-4">
          <QuestionBar value={question} onChange={setQuestion} running={running}
            onAsk={(q) => { void ask(q); }} onExample={onExample} />

          {state.failure && <StatusBanner key={state.runId} failure={state.failure} />}

          {result && (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-line bg-surface px-3.5 py-2.5" data-testid="result-header">
              <div className="min-w-0 basis-full sm:flex-1 sm:basis-auto">
                <p className="text-[14px] font-medium text-ink">{result.question}</p>
                {scalar !== undefined && (
                  <p className="mt-0.5 text-[13px] text-ink-2" data-testid="answer">
                    <span className="font-mono text-[18px] font-semibold tabular-nums text-ink">{formatCell(scalar)}</span>
                    <span className="ml-1.5">{result.columns[0]}</span>
                  </p>
                )}
              </div>
              {band && result.confidence !== null && <Chip tone={band.tone}>confidence {result.confidence.toFixed(2)} · {band.text.toLowerCase()}</Chip>}
              <span className="font-mono text-[11.5px] text-ink-2" data-testid="cost-latency">
                {formatUsd(result.cost_usd)} · {result.n_calls} call{result.n_calls === 1 ? "" : "s"} · {formatMs(result.elapsed_ms)}
              </span>
              {result.cached && <Chip tone="accent">cached</Chip>}
              {result.mode === "demo" && result.mode_reason && (
                <span data-testid="simulated" title={modeCopy(result.mode_reason).detail}><Chip tone="info">simulated model</Chip></span>
              )}
              <div className="basis-full"><FeedbackControls key={result.query_id} queryId={result.query_id} /></div>
            </div>
          )}

          <PipelineTimeline rows={rows} replayed={state.replayed} running={running} startedAt={state.startedAt} result={result}
            awaitingFirstEvent={running && state.order.length === 0} />

          {result?.outcome === "clarification" && (
            <ClarificationView result={result} running={running} onPick={(sql) => { runSql(sql, result.question, "reading"); }} />
          )}
          {result?.outcome === "cannot_answer" && (
            <Panel id="cannot" title="Can't be answered from this schema">
              <p className="text-[13px] text-ink-2">{result.cannot_answer_reason}</p>
              <p className="mt-1.5 text-[12px] text-ink-3">QueryGuard declines rather than inventing a table or column.</p>
            </Panel>
          )}

          {showResults && result.executed_sql !== null && result.outcome === "answered" && <ResultsTable result={result} />}
          {showResults && (result.outcome === "failed" || result.outcome === "refused") && (
            <Panel id="exec-error" title="Execution">
              <p className="text-[13px] text-fail" role="alert">{result.outcome === "refused" ? "Refused before running" : "The database rejected the query"}: {result.execution_error}</p>
            </Panel>
          )}

          {result?.outcome !== "clarification" && result?.outcome !== "cannot_answer" && (
            <SqlPanel result={result} guardrailEvent={state.seen.guardrails} draft={draft} running={running}
              onDraftChange={(sql) => { setEdit({ runId: state.runId, sql }); }}
              onRun={(sql) => { runSql(sql); }} />
          )}

          {state.status === "idle" && (
            <Panel id="intro" title="How it works">
              <ol className="list-decimal space-y-1 pl-5 text-[13px] text-ink-2">
                <li>A model writes SQL for your question, or asks which reading you meant.</li>
                <li>A static guardrail admits one read-only SELECT and nothing else.</li>
                <li>It runs as a database role that cannot write, inside a read-only transaction.</li>
                <li>Blind back-translation, an independent second query and sanity checks look for plausible but wrong answers.</li>
                <li>A calibrated score says how far to trust it, and shows why.</li>
              </ol>
            </Panel>
          )}
        </main>

        <aside className="min-w-0 space-y-4 lg:col-start-2 xl:col-start-auto">
          {running && !result && (
            <>
              <Panel id="confidence-pending" title="Confidence"><p className="text-[12.5px] text-ink-2">Scored once the checks finish.</p></Panel>
              <Panel id="verification-pending" title="Verification"><p className="text-[12.5px] text-ink-2">Back-translation, a second query and sanity checks run after the query executes.</p></Panel>
            </>
          )}
          {result && <ConfidenceCard result={result} />}
          {result && <VerificationPanel result={result} />}
        </aside>

        <section className="lg:hidden" aria-label="History">
          <HistorySidebar activeId={result?.query_id ?? null} onOpen={(item) => { void open(item.query_id); }} />
        </section>
      </div>
      <SchemaDrawer open={schemaOpen} onClose={() => { setSchemaOpen(false); }} />
    </div>
  );
}
