// Captures docs/img/pipeline.png from a running demo-mode app ($0, no API key):
//   docker compose up -d && node web/scripts/capture-hero.mjs [http://localhost:8080]
import { chromium } from "@playwright/test";

const base = process.argv[2] ?? "http://localhost:8080";
const out = new URL("../../docs/img/pipeline.png", import.meta.url).pathname;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });
await page.goto(base);
await page.getByTestId("fake-badge").waitFor();
await page.locator("#question").fill("What was gross revenue from orders placed in 2025, before refunds?");
await page.getByRole("button", { name: "Ask", exact: true }).click();
await page.locator("[data-testid=timeline] [data-stage=done][data-state=done]").waitFor({ timeout: 30_000 });
await page.getByTestId("confidence-score").waitFor();
await page.waitForTimeout(500);
await page.screenshot({ path: out });
await browser.close();
console.log(`wrote ${out}`);
