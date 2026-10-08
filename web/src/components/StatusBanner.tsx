import { useEffect, useState } from "react";

import type { Failure } from "../hooks/useQueryRun";
import { STAGE_LABEL } from "../lib/stages";
import { Alert } from "./icons";

/** Seconds left of `seconds`, counted from first render. Remount (key) to restart it. */
function useCountdown(seconds: number | null): number | null {
  const [deadline] = useState(() => (seconds === null ? null : Date.now() + seconds * 1000));
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (deadline === null) return;
    const timer = setInterval(() => { setNow(Date.now()); }, 1000);
    return () => { clearInterval(timer); };
  }, [deadline]);
  return deadline === null ? null : Math.max(0, Math.ceil((deadline - now) / 1000));
}

function duration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  return `${Math.floor(seconds / 3600)} h ${Math.ceil((seconds % 3600) / 60)} min`;
}

export function StatusBanner({ failure }: { failure: Failure }) {
  const left = useCountdown(failure.kind === "http" ? failure.retryAfter : null);
  let title: string;
  let body: string;
  if (failure.kind === "http" && failure.status === 429) {
    title = "Slow down a little";
    body = left !== null && left > 0 ? `The demo allows a few questions a minute per visitor. Try again in ${duration(left)}.` : "You can ask again now.";
  } else if (failure.kind === "http" && failure.status === 503) {
    title = "Today's demo budget is used up";
    body = `${failure.message}${left !== null && left > 0 ? ` (resets in ${duration(left)})` : ""}`;
  } else if (failure.kind === "http") {
    title = `The API refused the request (${failure.status})`;
    body = failure.message;
  } else if (failure.kind === "network") {
    title = "Can't reach the API";
    body = "Is it running? Start it with `make api` (or `make dev FAKE=1` for the $0 fake-LLM mode).";
  } else {
    title = `Stopped at: ${STAGE_LABEL[failure.stage]}`;
    body = `${failure.errorType}: ${failure.message}`;
  }
  const tone = failure.kind === "http" && failure.status === 429 ? "bg-warn-soft text-warn border-warn/40" : "bg-fail-soft text-fail border-fail/40";
  return (
    <div role="alert" data-testid="status-banner" className={`flex gap-2.5 rounded-lg border px-3.5 py-2.5 ${tone}`}>
      <Alert size={16} className="mt-0.5 shrink-0" />
      <div>
        <p className="text-[13px] font-semibold">{title}</p>
        <p className="text-[12.5px] text-ink">{body}</p>
      </div>
    </div>
  );
}
