import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../api/client";
import { ThumbDown, ThumbUp } from "./icons";
import { Button } from "./ui";

/** Was this right? A "no" can become a new golden eval case (export-feedback). */
export function FeedbackControls({ queryId }: { queryId: string }) {
  const client = useQueryClient();
  const [verdict, setVerdict] = useState<boolean | null>(null);
  const [note, setNote] = useState("");
  const send = useMutation({
    mutationFn: (correct: boolean) => api.feedback({ query_id: queryId, correct, note: note.trim() || null }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["history"] }),
  });

  if (send.isSuccess) return <p className="text-[12px] text-ink-2" role="status">Thanks, recorded.{send.data.correct ? "" : " Wrong answers become candidate eval cases."}</p>;

  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="feedback">
      <span className="text-[12px] text-ink-2">Was this right?</span>
      <Button variant={verdict === true ? "primary" : "secondary"} aria-pressed={verdict === true} aria-label="Correct"
        onClick={() => { setVerdict(true); send.mutate(true); }}><ThumbUp size={13} /></Button>
      <Button variant={verdict === false ? "primary" : "secondary"} aria-pressed={verdict === false} aria-label="Wrong"
        onClick={() => { setVerdict(false); }}><ThumbDown size={13} /></Button>
      {verdict === false && (
        <form className="flex min-w-[220px] flex-1 gap-2" onSubmit={(e) => { e.preventDefault(); send.mutate(false); }}>
          <label className="sr-only" htmlFor={`note-${queryId}`}>What was wrong?</label>
          <input id={`note-${queryId}`} value={note} onChange={(e) => { setNote(e.target.value); }} maxLength={1000}
            placeholder="What was wrong? (optional)"
            className="min-w-0 flex-1 rounded-md border border-line-strong bg-surface px-2 py-1 text-[12.5px] text-ink placeholder:text-ink-3" />
          <Button type="submit" variant="primary" disabled={send.isPending}>Send</Button>
        </form>
      )}
      {send.isError && <span className="text-[12px] text-fail">Could not send feedback.</span>}
    </div>
  );
}
