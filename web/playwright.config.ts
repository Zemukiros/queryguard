import os from "node:os";
import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

/**
 * e2e against the real API in fake-LLM mode ($0, no key needed) and the
 * production build. Postgres must be up (`docker compose up -d db`). Ports differ
 * from `make dev` (8000 / 5173) so both can run at once.
 */
const API_PORT = 8010;
const WEB_PORT = 4174;
/** A second API, live mode with a $0 ceiling: every question falls back to demo mode (e2e/fallback.spec.ts). */
export const FALLBACK_API_PORT = 8011;
const tmp = (name: string) => path.join(os.tmpdir(), `queryguard-e2e-${String(process.pid)}-${name}`);
const appDb = tmp("app.db");

export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  fullyParallel: false,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${String(WEB_PORT)}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: `cd .. && uv run python -m queryguard.api --port ${String(API_PORT)}`,
      env: {
        QUERYGUARD_FAKE_LLM: "1",
        QUERYGUARD_FAKE_LLM_DELAY_MS: "50",
        QUERYGUARD_APP_DB: appDb,
        QUERYGUARD_RATE_PER_MINUTE: "1000",
        QUERYGUARD_RATE_PER_DAY: "100000",
      },
      url: `http://127.0.0.1:${String(API_PORT)}/healthz`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      // Live mode, so the budget is checked, but a $0 ceiling: no question is
      // ever admitted to the real model, so none can spend. The key is a dud,
      // and the live client is never even built.
      command: `cd .. && uv run python -m queryguard.api --port ${String(FALLBACK_API_PORT)}`,
      env: {
        QUERYGUARD_FAKE_LLM: "0",
        QUERYGUARD_LIVE: "1",
        QUERYGUARD_DAILY_SPEND_USD: "0",
        ANTHROPIC_API_KEY: "e2e-never-used",
        QUERYGUARD_FAKE_LLM_DELAY_MS: "50",
        QUERYGUARD_APP_DB: tmp("fallback.db"),
        QUERYGUARD_LLM_LOG: tmp("fallback-llm.jsonl"),
        QUERYGUARD_FAKE_LLM_LOG: tmp("fallback-fake.jsonl"),
        QUERYGUARD_RATE_PER_MINUTE: "1000",
        QUERYGUARD_RATE_PER_DAY: "100000",
      },
      url: `http://127.0.0.1:${String(FALLBACK_API_PORT)}/healthz`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: `npm run build && npx vite preview --host 127.0.0.1 --port ${String(WEB_PORT)} --strictPort`,
      env: { QUERYGUARD_API: `http://127.0.0.1:${String(API_PORT)}` },
      url: `http://127.0.0.1:${String(WEB_PORT)}`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
