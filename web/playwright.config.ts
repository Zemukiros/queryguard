import os from "node:os";
import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

/**
 * e2e against the real API in fake-LLM mode ($0, no key needed) and the
 * production build. Postgres must be up (`docker compose up -d`). Ports differ
 * from `make dev` (8000 / 5173) so both can run at once.
 */
const API_PORT = 8010;
const WEB_PORT = 4174;
const appDb = path.join(os.tmpdir(), `queryguard-e2e-${String(process.pid)}.db`);

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
      command: `npm run build && npx vite preview --host 127.0.0.1 --port ${String(WEB_PORT)} --strictPort`,
      env: { QUERYGUARD_API: `http://127.0.0.1:${String(API_PORT)}` },
      url: `http://127.0.0.1:${String(WEB_PORT)}`,
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
