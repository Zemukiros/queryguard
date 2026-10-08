import { formatResetIn, modeCopy } from "./mode";

describe("mode copy", () => {
  it("says why demo mode is on, with the reset when it matters", () => {
    expect(modeCopy("budget", 6 * 3600).label).toBe("Demo mode · today's live budget is used up, resets in 6 h");
    expect(modeCopy("call_cap", 1800).label).toBe("Demo mode · today's live limit is reached, resets in 30 min");
    expect(modeCopy("switched_off").label).toBe("Demo mode · live answers are off");
    expect(modeCopy("demo_deployment").label).toBe("Demo mode · simulated model · $0");
    for (const reason of ["budget", "call_cap", "switched_off", "demo_deployment"] as const) {
      expect(modeCopy(reason).detail).toContain("Nothing is spent");
    }
  });

  it("rounds the reset coarsely and never says zero", () => {
    expect(formatResetIn(null)).toBeNull();
    expect(formatResetIn(20)).toBe("1 min");
    expect(formatResetIn(5400)).toBe("2 h");
  });
});
