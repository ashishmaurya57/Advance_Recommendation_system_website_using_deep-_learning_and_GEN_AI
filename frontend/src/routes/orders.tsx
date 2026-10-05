import { Package } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";

import { BookCover } from "@/components/BookCard";
import { EmptyState, ErrorState, PageHeader } from "@/components/States";
import { requireUser } from "@/lib/auth";
import * as fmt from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { queries, useCancelOrder } from "@/lib/queries";
import { useToast } from "@/lib/toast";

export const Route = createFileRoute("/orders")({
  beforeLoad: ({ context, location }) => requireUser(context.queryClient, location.href),
  component: OrdersPage,
});

const STATUS: Record<string, { label: string; className: string }> = {
  paid: { label: "Paid", className: "bg-success/12 text-success" },
  pending: { label: "Pay on delivery", className: "bg-accent-soft text-accent" },
};

function OrdersPage() {
  const { t } = useI18n();
  const orders = useQuery(queries.orders());
  const cancel = useCancelOrder();
  const toast = useToast();
  const [confirming, setConfirming] = useState<number | null>(null);

  return (
    <div className="container-page">
      <PageHeader title={t("orders.title")} />
      {orders.isPending ? (
        <div className="flex animate-pulse flex-col gap-4">
          {[0, 1, 2].map((i) => (
            <div key={i} className="h-28 rounded-2xl bg-surface-2" />
          ))}
        </div>
      ) : orders.isError ? (
        <ErrorState error={orders.error} onRetry={() => orders.refetch()} />
      ) : orders.data.length === 0 ? (
        <EmptyState
          icon={<Package size={26} />}
          title={t("orders.empty")}
          body="When you order a book it will show up here with its status."
          action={
            <Link to="/books" className="btn-primary">
              {t("home.browse")}
            </Link>
          }
        />
      ) : (
        <ul className="flex flex-col gap-3">
          {orders.data.map(({ id, status, date, product: book }) => {
            const s = STATUS[status] ?? { label: fmt.titleCase(status), className: "bg-surface-2 text-ink-2" };
            return (
              <li key={id} className="flex items-center gap-4 rounded-2xl border border-line bg-surface p-4 sm:gap-6">
                <Link to="/books/$bookId" params={{ bookId: String(book.id) }} className="group w-14 shrink-0 sm:w-16">
                  <BookCover book={book} className="aspect-[3/4] rounded-xl" />
                </Link>
                <div className="flex min-w-0 flex-1 flex-col gap-1">
                  <Link
                    to="/books/$bookId"
                    params={{ bookId: String(book.id) }}
                    className="truncate font-medium text-ink hover:text-accent"
                  >
                    {book.name}
                  </Link>
                  <p className="text-sm text-ink-3">
                    Order #{id}, placed {fmt.date(date)}
                  </p>
                  <span className={`mt-1 w-fit rounded-full px-2.5 py-0.5 text-xs font-medium ${s.className}`}>{s.label}</span>
                </div>
                <div className="flex shrink-0 flex-col items-end gap-2">
                  <span className="font-semibold text-ink tabular-nums">{fmt.price(book.price)}</span>
                  {confirming === id ? (
                    <div className="flex items-center gap-1">
                      <button
                        className="rounded-full px-3 py-1 text-xs font-medium text-danger hover:bg-danger/10"
                        disabled={cancel.isPending}
                        onClick={() =>
                          cancel.mutate(id, {
                            onSuccess: (r) => {
                              toast(r.message);
                              setConfirming(null);
                            },
                            onError: (e) => toast(e.message, "error"),
                          })
                        }
                      >
                        Yes, cancel
                      </button>
                      <button
                        className="rounded-full px-3 py-1 text-xs text-ink-2 hover:bg-surface-2"
                        onClick={() => setConfirming(null)}
                      >
                        Keep
                      </button>
                    </div>
                  ) : (
                    <button
                      className="rounded-full px-3 py-1 text-xs text-ink-3 hover:bg-surface-2 hover:text-ink"
                      onClick={() => setConfirming(id)}
                    >
                      Cancel order
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
