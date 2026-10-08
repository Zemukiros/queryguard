import type { QueryResult } from "../api/client";
import type { Tone } from "../components/ui";

export const BAND: Record<NonNullable<QueryResult["confidence_band"]>, { tone: Tone; text: string }> = {
  high: { tone: "ok", text: "High" },
  medium: { tone: "warn", text: "Medium" },
  low: { tone: "fail", text: "Low" },
};
