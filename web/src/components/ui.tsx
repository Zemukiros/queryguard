import type { ButtonHTMLAttributes, ReactNode } from "react";

import { cx } from "../lib/cx";

export function Panel({ title, aside, children, className, id }: {
  title?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string; id?: string;
}) {
  return (
    <section aria-labelledby={id ? `${id}-title` : undefined}
      className={cx("rounded-lg border border-line bg-surface", className)}>
      {title !== undefined && (
        <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 border-b border-line px-3.5 py-2">
          <h2 id={id ? `${id}-title` : undefined} className="qg-label text-ink-2">{title}</h2>
          {aside}
        </header>
      )}
      <div className="p-3.5">{children}</div>
    </section>
  );
}

export type Tone = "ok" | "warn" | "fail" | "info" | "neutral" | "accent";

const TONE: Record<Tone, string> = {
  ok: "bg-ok-soft text-ok",
  warn: "bg-warn-soft text-warn",
  fail: "bg-fail-soft text-fail",
  info: "bg-info-soft text-info",
  neutral: "bg-surface-2 text-ink-2",
  accent: "bg-accent-soft text-accent-strong",
};

export function Chip({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span className={cx("inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium leading-4", TONE[tone], className)}>
      {children}
    </span>
  );
}

export function Button({ variant = "secondary", className, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
}) {
  return (
    <button type="button" {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-accent-fill text-accent-fill-ink hover:brightness-110 disabled:bg-surface-3 disabled:text-ink-3 disabled:opacity-100",
        variant === "secondary" && "border border-line-strong bg-surface text-ink hover:bg-surface-2",
        variant === "ghost" && "text-ink-2 hover:bg-surface-2 hover:text-ink",
        className,
      )} />
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cx("font-mono text-[12px]", className)}>{children}</span>;
}
