import type { ReactNode } from "react";

/** Centered card used by the sign-in and sign-up pages. */
export function AuthLayout({ title, subtitle, children }: { title: string; subtitle: ReactNode; children: ReactNode }) {
  return (
    <div className="container-page flex justify-center py-12 md:py-20">
      <div className="flex w-full max-w-md flex-col gap-8">
        <div className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold tracking-tight text-ink">{title}</h1>
          <p className="text-sm text-ink-2">{subtitle}</p>
        </div>
        <div className="rounded-2xl border border-line bg-surface p-6 shadow-soft sm:p-8">{children}</div>
      </div>
    </div>
  );
}

export function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor={htmlFor} className="field-label">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-ink-3">{hint}</p>}
    </div>
  );
}
