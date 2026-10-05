import { WarningCircle } from "@phosphor-icons/react";
import type { ReactNode } from "react";

export function EmptyState({
  icon,
  title,
  body,
  action,
}: {
  icon: ReactNode;
  title: string;
  body?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-line px-6 py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-accent-soft text-accent">{icon}</div>
      <h2 className="text-lg font-semibold text-ink">{title}</h2>
      {body && <p className="max-w-[48ch] text-sm leading-relaxed text-ink-2">{body}</p>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div role="alert" className="flex flex-col items-center gap-3 rounded-2xl border border-line bg-surface px-6 py-12 text-center">
      <WarningCircle size={32} className="text-danger" />
      <p className="max-w-[48ch] text-sm text-ink-2">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="btn-secondary">
          Try again
        </button>
      )}
    </div>
  );
}

export function PageHeader({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2 pt-10 pb-8 md:pt-14">
      <h1 className="text-3xl font-semibold tracking-tight text-ink md:text-4xl">{title}</h1>
      {children && <div className="max-w-[65ch] text-base leading-relaxed text-ink-2">{children}</div>}
    </div>
  );
}
