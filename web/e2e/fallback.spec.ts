import { expect, test } from "@playwright/test";

import { FALLBACK_API_PORT } from "../playwright.config";

/**
 * The same production build, pointed at the API whose daily budget is spent.
 * Requests are rewritten at the network layer, so the page still sees its own
 * origin; the API decides the mode, and the UI has to say so.
 */
test.beforeEach(async ({ page }) => {
  await page.route(/\/(v1\/|healthz)/, (route) =>
    route.continue({ url: route.request().url().replace(/:\d+\//, `:${String(FALLBACK_API_PORT)}/`) }));
});

test("with today's live budget spent, questions run on the simulated model and say so", async ({ page }) => {
  await page.goto("/");
  const badge = page.getByTestId("mode-badge");
  await expect(badge).toHaveAttribute("data-reason", "budget");
  await expect(badge).toContainText("today's live budget is used up");
  await expect(page.getByTestId("live-budget")).toHaveCount(0);

  await page.getByRole("button", { name: "How many orders were cancelled?" }).click();
  await expect(page.locator("[data-testid=timeline] [data-stage=done]")).toHaveAttribute("data-state", "done");
  await expect(page.getByTestId("simulated")).toBeVisible();
  await expect(page.getByTestId("cost-latency")).toContainText("$0");
  await expect(page.getByTestId("results-table").locator("tbody tr")).toHaveCount(1);
});
