/**
 * Words for why a question runs on the simulated model instead of the real one.
 * The API decides the mode (settings.py, "Modes"); this only explains it.
 */
import type { Health } from "../api/client";

export type ModeReason = NonNullable<Health["mode_reason"]>;

/** "6 h", "45 min": coarse on purpose, the reset is at 00:00 UTC. */
export function formatResetIn(seconds: number | null | undefined): string | null {
  if (seconds === null || seconds === undefined) return null;
  if (seconds >= 3600) return `${String(Math.round(seconds / 3600))} h`;
  return `${String(Math.max(1, Math.round(seconds / 60)))} min`;
}

/** Short label for the header badge, and the sentence for its tooltip. */
export function modeCopy(reason: ModeReason, resetsInS?: number | null): { label: string; detail: string } {
  const reset = formatResetIn(resetsInS);
  const resets = reset ? `, resets in ${reset}` : "";
  const sim = "A simulated model answers the example questions; the guardrail, database and checks are real. Nothing is spent.";
  switch (reason) {
    case "demo_deployment":
      return { label: "Demo mode · simulated model · $0", detail: `Demo mode: ${sim}` };
    case "switched_off":
      return { label: "Demo mode · live answers are off", detail: `Live answers are switched off. ${sim}` };
    case "budget":
      return { label: `Demo mode · today's live budget is used up${resets}`, detail: `Today's spending limit for the real model is reached. ${sim}` };
    case "call_cap":
      return { label: `Demo mode · today's live limit is reached${resets}`, detail: `Today's limit on model calls is reached. ${sim}` };
  }
}
