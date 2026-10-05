import { CaretLeft, CaretRight, CircleNotch, MagnifyingGlassMinus, MagnifyingGlassPlus, X } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import * as fmt from "@/lib/format";
import { queries } from "@/lib/queries";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

type Props = {
  bookId: number;
  title: string;
  pdfUrl: string;
  originalLanguage: string;
  onClose: () => void;
};

export default function PdfReader({ bookId, title, pdfUrl, originalLanguage, onClose }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const [doc, setDoc] = useState<pdfjs.PDFDocumentProxy | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [language, setLanguage] = useState("original");
  const languages = useQuery(queries.languages());

  const translating = language !== "original";
  const translation = useQuery({
    queryKey: ["translation", bookId, page, language],
    queryFn: () =>
      api.get<{ page: number; text: string }>(
        `/products/${bookId}/pages/${page}?language=${encodeURIComponent(language)}`,
      ),
    enabled: translating,
    staleTime: Infinity,
  });

  // Load the document once.
  useEffect(() => {
    const task = pdfjs.getDocument({ url: pdfUrl });
    task.promise.then(setDoc, (e: Error) => setLoadError(e.message));
    return () => {
      task.destroy();
    };
  }, [pdfUrl]);

  // Render the current page to the canvas.
  useEffect(() => {
    if (!doc || translating || !canvasRef.current) return;
    let cancelled = false;
    let renderTask: ReturnType<pdfjs.PDFPageProxy["render"]> | null = null;
    doc.getPage(page).then((p) => {
      if (cancelled || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const width = Math.min(window.innerWidth - 32, 860) * zoom;
      const base = p.getViewport({ scale: 1 });
      const ratio = window.devicePixelRatio || 1;
      const viewport = p.getViewport({ scale: (width / base.width) * ratio });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = `${viewport.width / ratio}px`;
      canvas.style.height = `${viewport.height / ratio}px`;
      renderTask = p.render({ canvas, viewport });
      renderTask.promise.catch(() => {
        // cancelled by a newer render
      });
    });
    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
  }, [doc, page, zoom, translating]);

  // Keyboard: Esc closes, arrows turn pages. Lock body scroll while open.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") setPage((p) => (doc ? Math.min(p + 1, doc.numPages) : p));
      if (e.key === "ArrowLeft") setPage((p) => Math.max(p - 1, 1));
    };
    document.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialogRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [doc, onClose]);

  const total = doc?.numPages ?? 0;
  const iconBtn =
    "flex h-9 w-9 items-center justify-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink disabled:opacity-40";

  return (
    <div
      ref={dialogRef}
      tabIndex={-1}
      role="dialog"
      aria-modal="true"
      aria-label={`Reading ${title}`}
      className="fixed inset-0 z-50 flex flex-col bg-bg focus:outline-none"
    >
      <div className="flex flex-wrap items-center gap-2 border-b border-line bg-surface px-3 py-2 sm:px-5">
        <p className="mr-auto min-w-0 flex-1 truncate text-sm font-medium text-ink">{title}</p>

        <button className={iconBtn} onClick={() => setPage((p) => Math.max(p - 1, 1))} disabled={page <= 1} aria-label="Previous page">
          <CaretLeft size={18} />
        </button>
        <span className="min-w-20 text-center text-sm text-ink-2 tabular-nums">
          {total ? `${page} / ${total}` : "..."}
        </span>
        <button
          className={iconBtn}
          onClick={() => setPage((p) => Math.min(p + 1, total))}
          disabled={!total || page >= total}
          aria-label="Next page"
        >
          <CaretRight size={18} />
        </button>

        {!translating && (
          <>
            <button className={iconBtn} onClick={() => setZoom((z) => Math.max(z - 0.2, 0.6))} aria-label="Zoom out">
              <MagnifyingGlassMinus size={18} />
            </button>
            <button className={iconBtn} onClick={() => setZoom((z) => Math.min(z + 0.2, 2))} aria-label="Zoom in">
              <MagnifyingGlassPlus size={18} />
            </button>
          </>
        )}

        <label className="flex items-center gap-2 text-sm text-ink-2">
          <span className="hidden sm:inline">Language</span>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="rounded-full border border-line bg-surface px-3 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
          >
            <option value="original">Original ({fmt.titleCase(originalLanguage)})</option>
            {languages.data
              ?.filter((l) => l !== originalLanguage.toLowerCase())
              .map((l) => (
                <option key={l} value={l}>
                  {fmt.titleCase(l)}
                </option>
              ))}
          </select>
        </label>

        <button className={iconBtn} onClick={onClose} aria-label="Close reader">
          <X size={18} />
        </button>
      </div>

      <div className="flex flex-1 justify-center overflow-auto p-4 sm:p-8">
        {loadError ? (
          <p className="self-center text-sm text-danger">Couldn't open this PDF: {loadError}</p>
        ) : translating ? (
          <article className="w-full max-w-[70ch] rounded-2xl bg-surface p-6 shadow-soft sm:p-10">
            {translation.isPending ? (
              <div className="flex items-center gap-2 text-sm text-ink-3">
                <CircleNotch size={16} className="animate-spin" /> Translating page {page}...
              </div>
            ) : translation.isError ? (
              <p className="text-sm text-danger">{(translation.error as Error).message}</p>
            ) : translation.data?.text.trim() ? (
              <p className="text-base leading-relaxed whitespace-pre-line text-ink">{translation.data.text}</p>
            ) : (
              <p className="text-sm text-ink-3">
                This page has no selectable text (it may be a scanned image), so it can't be translated.
              </p>
            )}
          </article>
        ) : (
          <>
            {!doc && (
              <div className="flex items-center gap-2 self-center text-sm text-ink-3">
                <CircleNotch size={16} className="animate-spin" /> Opening book...
              </div>
            )}
            <canvas ref={canvasRef} className={`h-fit rounded-lg bg-white shadow-lift ${doc ? "" : "hidden"}`} />
          </>
        )}
      </div>
    </div>
  );
}
