import { expect, test, type Page } from "@playwright/test";

const timelineRow = (page: Page, stage: string) => page.locator(`[data-testid=timeline] [data-stage=${stage}]`);

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("mode-badge")).toHaveAttribute("data-reason", "demo_deployment");
});

test("a question streams through every stage and renders its result", async ({ page }) => {
  await page.getByRole("button", { name: "How many orders were cancelled?" }).click();

  await expect(timelineRow(page, "done")).toHaveAttribute("data-state", "done");
  for (const stage of ["generating", "guardrails", "executing", "sanity", "backtranslate", "agreement", "confidence"]) {
    await expect(timelineRow(page, stage)).toHaveAttribute("data-state", "done");
  }
  await expect(timelineRow(page, "clarification")).toHaveAttribute("data-state", "skipped");

  const table = page.getByTestId("results-table");
  await expect(table).toBeVisible();
  await expect(table.locator("tbody tr")).toHaveCount(1);
  await expect(page.getByTestId("row-count")).toContainText("1 row");
  await expect(page.getByTestId("confidence-score")).toHaveText(/^0\.\d\d$/);
  await expect(page.getByTestId("cost-latency")).toContainText("$0");
});

test("a pasted DROP TABLE is rejected by the guardrail, by name", async ({ page }) => {
  await page.locator("[data-testid=sql-editor]").click();
  await page.keyboard.insertText("DROP TABLE orders;");
  await page.getByTestId("run-sql").click();

  const rejection = page.getByTestId("guardrail-rejection");
  await expect(rejection).toBeVisible();
  await expect(rejection).toContainText("statement_type");
  await expect(rejection).toContainText("got DROP");
  await expect(timelineRow(page, "guardrails")).toHaveAttribute("data-state", "blocked");
  await expect(timelineRow(page, "executing")).toHaveAttribute("data-state", "skipped");
  await expect(page.getByTestId("results-table")).toHaveCount(0);
});

test("an ambiguous question asks which reading, and the chosen one runs", async ({ page }) => {
  await page.locator("#question").fill("Who are our top 10 customers?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();

  const readings = page.getByTestId("interpretations");
  await expect(readings.getByRole("article")).toHaveCount(3);
  await expect(timelineRow(page, "clarification")).toHaveAttribute("data-state", "done");

  await page.getByRole("button", { name: "Run the by_total_spend reading" }).click();
  await expect(timelineRow(page, "done")).toHaveAttribute("data-state", "done");
  await expect(page.getByTestId("results-table").locator("tbody tr")).toHaveCount(10);
  await expect(page.getByText("chosen reading", { exact: true }).first()).toBeVisible();
  await expect(page.locator("[data-testid=timeline] [data-stage=agreement]")).toHaveAttribute("data-state", "done");
});
