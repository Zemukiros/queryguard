import { expect, test } from "@playwright/test";

/** A cold serverless instance takes ~5 s to send its first byte; the timeline should explain the wait. */
test("a slow first byte shows 'Starting the server…' until the first event arrives", async ({ page }) => {
  await page.route("**/v1/query/stream", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2500)); // a cold start, simulated
    await route.continue();
  });
  await page.goto("/");
  await page.getByTestId("examples").getByRole("button", { name: "How many orders were cancelled?" }).click();

  const note = page.getByTestId("starting-server");
  await expect(note).toBeVisible({ timeout: 2400 });
  await expect(note).toContainText("Starting the server");
  await expect(page.locator("[data-testid=timeline] [data-stage=done]")).toHaveAttribute("data-state", "done");
  await expect(note).toHaveCount(0);
});
