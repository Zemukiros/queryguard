import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { Stage, StageEvent } from "../api/client";
import { timelineRows } from "../lib/stages";
import { blockedResult, clarificationResult, event, guardrails, normalRun, resetClock, result } from "../test/fixtures";
import { ClarificationView } from "./ClarificationView";
import { ConfidenceCard } from "./ConfidenceCard";
import { PipelineTimeline } from "./PipelineTimeline";
import { GuardrailRejection, SqlPanel } from "./SqlPanel";

function timelineAfter(events: StageEvent[], running: boolean) {
  const seen = Object.fromEntries(events.map((e) => [e.stage, e])) as Partial<Record<Stage, StageEvent>>;
  const rows = timelineRows({ seen, mode: "query", running, failedStage: null, result: null });
  return render(<PipelineTimeline rows={rows} replayed={null} running={running} startedAt={running ? Date.now() : null} result={null} />);
}

const rowStates = () =>
  within(screen.getByTestId("timeline")).getAllByRole("listitem").map((li) => [li.dataset.stage, li.dataset.state]);

describe("PipelineTimeline", () => {
  it("renders every stage in pipeline order and lights them as events arrive", () => {
    const events = normalRun();
    const { rerender } = timelineAfter(events.slice(0, 3), true);
    expect(rowStates()).toEqual([
      ["generating", "done"], ["clarification", "skipped"], ["guardrails", "done"], ["executing", "done"],
      ["sanity", "active"], ["backtranslate", "pending"], ["agreement", "pending"], ["confidence", "pending"], ["done", "pending"],
    ]);
    expect(screen.getByText(/Running/)).toBeInTheDocument();

    const seen = Object.fromEntries(events.map((e) => [e.stage, e])) as Partial<Record<Stage, StageEvent>>;
    const final = timelineRows({ seen, mode: "query", running: false, failedStage: null, result: null });
    rerender(<PipelineTimeline rows={final} replayed={null} running={false} startedAt={null} result={result()} />);
    expect(rowStates().filter(([, state]) => state === "done").map(([stage]) => stage)).toEqual([
      "generating", "guardrails", "executing", "sanity", "backtranslate", "agreement", "confidence", "done",
    ]);
    expect(screen.getByText(/alignment 1\.00/)).toBeInTheDocument();
    expect(screen.getByText("Answered · all checks passed")).toBeInTheDocument();
    expect(screen.getByText("80 ms")).toBeInTheDocument();
  });

  it("shows a guardrail rejection as a blocked row, with the rule", () => {
    resetClock();
    timelineAfter([event(guardrails({ allowed: false, rule: "statement_type", reason: "got DROP", rewritten_sql: null, sql_to_execute: null }))], true);
    const row = within(screen.getByTestId("timeline")).getAllByRole("listitem").find((li) => li.dataset.stage === "guardrails");
    expect(row?.dataset.state).toBe("blocked");
    expect(row).toHaveTextContent("rejected · statement_type");
    expect(row).toHaveTextContent("stopped the answer");
  });

  it("flags a weak signal as a warning, not a pass", () => {
    resetClock();
    timelineAfter([event({ stage: "backtranslate", back_translation: "Something else?", alignment: 0.4, discrepancies: ["different filter"], error: null })], true);
    const row = within(screen.getByTestId("timeline")).getAllByRole("listitem").find((li) => li.dataset.stage === "backtranslate");
    expect(row?.dataset.state).toBe("warn");
  });
});

describe("guardrail rejection", () => {
  it("names the rule and the reason, and says nothing ran", () => {
    render(<GuardrailRejection rule="statement_type" reason="only SELECT and WITH ... SELECT may be executed, got DROP" />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Blocked by the guardrail");
    expect(alert).toHaveTextContent("Rule: statement_type");
    expect(alert).toHaveTextContent("Only SELECT and WITH ... SELECT may be executed, got DROP.");
    expect(alert).toHaveTextContent("statement_type");
    expect(alert).toHaveTextContent("got DROP");
    expect(alert).toHaveTextContent("Nothing was executed");
  });

  it("appears in the SQL panel for a blocked result, and Run my SQL sends the draft", async () => {
    const onRun = vi.fn();
    render(<SqlPanel result={blockedResult()} guardrailEvent={undefined} draft="DROP TABLE orders;" onDraftChange={() => undefined} onRun={onRun} running={false} />);
    expect(screen.getByTestId("guardrail-rejection")).toHaveTextContent("statement_type");
    await userEvent.click(screen.getByTestId("run-sql"));
    expect(onRun).toHaveBeenCalledWith("DROP TABLE orders;");
  });

  it("summarises an appended LIMIT in one line, with the diff on request", () => {
    render(<SqlPanel result={result()} guardrailEvent={undefined} draft="" onDraftChange={() => undefined} onRun={() => undefined} running={false} />);
    expect(screen.getByText(/The guardrail appended/)).toHaveTextContent("LIMIT 1001");
    expect(screen.getByText("written by the model")).toBeInTheDocument();
    expect(screen.queryByTestId("guardrail-rejection")).not.toBeInTheDocument();
  });
});

describe("ClarificationView", () => {
  it("lists every reading, and clicking one runs its SQL", async () => {
    const onPick = vi.fn();
    const clarification = clarificationResult();
    render(<ClarificationView result={clarification} onPick={onPick} running={false} />);
    expect(screen.getAllByRole("article")).toHaveLength(2);
    await userEvent.click(screen.getByRole("button", { name: "Run the by_order_count reading" }));
    expect(onPick).toHaveBeenCalledWith(clarification.interpretations[1]?.sql);
  });

  it("disables the readings while a run is in flight", () => {
    render(<ClarificationView result={clarificationResult()} onPick={() => undefined} running />);
    for (const button of screen.getAllByRole("button")) expect(button).toBeDisabled();
  });
});

describe("ConfidenceCard", () => {
  it("shows the score, its band and the non-zero contributions, largest first", () => {
    render(<ConfidenceCard result={result()} />);
    expect(screen.getByTestId("confidence-score")).toHaveTextContent("0.95");
    expect(screen.getByText("High")).toBeInTheDocument();
    const labels = within(screen.getByTestId("contributions")).getAllByRole("listitem").map((li) => li.textContent);
    expect(labels).toEqual(["Second query agreed+2.07", "Back-translation alignment+1.01"]);
    expect(screen.getByText(/Starting point/)).toHaveTextContent("−0.40");
    expect(screen.getByText("1 other signal had no effect")).toBeInTheDocument();
  });

  it("shows no number at all when nothing executed", () => {
    render(<ConfidenceCard result={blockedResult()} />);
    expect(screen.getByTestId("confidence-score")).toHaveTextContent("—");
    expect(screen.getByText(/nothing executed/)).toBeInTheDocument();
  });
});
