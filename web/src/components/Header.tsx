import { useQuery } from "@tanstack/react-query";

import { api } from "../api/client";
import { formatUsd } from "../lib/format";
import { Database, Moon, Shield, Sun } from "./icons";
import { Button, Chip } from "./ui";

export function Header({ theme, onToggleTheme, onOpenSchema }: {
  theme: "light" | "dark"; onToggleTheme: () => void; onOpenSchema: () => void;
}) {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1600px] items-center gap-3 px-4 py-2.5">
        <span className="grid size-7 place-items-center rounded-md bg-accent-fill text-accent-fill-ink"><Shield size={15} /></span>
        <div className="min-w-0 shrink-0">
          <h1 className="text-[15px] font-semibold leading-tight text-ink">QueryGuard</h1>
          <p className="hidden truncate text-[11.5px] text-ink-3 sm:block">Text-to-SQL behind a read-only database role, with every check shown</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {health.data?.fake_llm && (
            <span data-testid="fake-badge" title="Demo mode: a simulated model answers the eval set's questions. The guardrail, database and checks are real. Nothing is spent.">
              <Chip tone="info"><span className="sm:hidden">Demo · $0</span><span className="hidden sm:inline">Demo mode · simulated model · $0</span></Chip>
            </span>
          )}
          {health.data && !health.data.fake_llm && (
            <span className="hidden font-mono text-[11px] text-ink-3 md:inline">
              {formatUsd(health.data.budget.spent_today_usd)} of {formatUsd(health.data.budget.daily_ceiling_usd)} today
            </span>
          )}
          {health.isError && <Chip tone="fail">API offline</Chip>}
          <Button variant="secondary" onClick={onOpenSchema}><Database size={14} /> <span className="hidden sm:inline">Schema</span></Button>
          <Button variant="ghost" onClick={onToggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </Button>
        </div>
      </div>
    </header>
  );
}
