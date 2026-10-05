import { CheckCircle, WarningCircle, X } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { createContext, useCallback, useContext, useRef, useState, type ReactNode } from "react";

type Tone = "success" | "error";
type Toast = { id: number; message: string; tone: Tone };

const ToastContext = createContext<((message: string, tone?: Tone) => void) | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const reduce = useReducedMotion();

  const dismiss = useCallback((id: number) => setToasts((t) => t.filter((x) => x.id !== id)), []);
  const show = useCallback(
    (message: string, tone: Tone = "success") => {
      const id = nextId.current++;
      setToasts((t) => [...t.slice(-2), { id, message, tone }]);
      setTimeout(() => dismiss(id), 4000);
    },
    [dismiss],
  );

  return (
    <ToastContext.Provider value={show}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed inset-x-4 bottom-4 z-50 flex flex-col items-center gap-2 sm:inset-x-auto sm:right-6 sm:bottom-6 sm:items-end"
      >
        <AnimatePresence initial={false}>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              layout
              initial={reduce ? false : { opacity: 0, y: 16, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={reduce ? { opacity: 0 } : { opacity: 0, y: 8, scale: 0.98 }}
              transition={{ type: "spring", stiffness: 380, damping: 30 }}
              className="pointer-events-auto flex w-full max-w-sm items-start gap-3 rounded-2xl border border-line bg-surface px-4 py-3 text-sm shadow-lift"
              role={t.tone === "error" ? "alert" : "status"}
            >
              {t.tone === "success" ? (
                <CheckCircle size={20} weight="fill" className="mt-px shrink-0 text-success" />
              ) : (
                <WarningCircle size={20} weight="fill" className="mt-px shrink-0 text-danger" />
              )}
              <p className="flex-1 text-ink">{t.message}</p>
              <button
                onClick={() => dismiss(t.id)}
                className="-mr-1 rounded-full p-1 text-ink-3 hover:bg-surface-2 hover:text-ink"
                aria-label="Dismiss"
              >
                <X size={14} />
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
