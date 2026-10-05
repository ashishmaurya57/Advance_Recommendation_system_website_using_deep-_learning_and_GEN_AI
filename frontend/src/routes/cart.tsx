import { CreditCard, Package, ShoppingBagOpen, Trash } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";

import { BookCover } from "@/components/BookCard";
import { EmptyState, ErrorState, PageHeader } from "@/components/States";
import { requireUser } from "@/lib/auth";
import * as fmt from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { queries, usePlaceOrder, useRemoveFromCart } from "@/lib/queries";
import { useToast } from "@/lib/toast";
import { usePayOnline } from "@/lib/usePayOnline";

export const Route = createFileRoute("/cart")({
  beforeLoad: ({ context, location }) => requireUser(context.queryClient, location.href),
  component: CartPage,
});

function CartPage() {
  const { t } = useI18n();
  const cart = useQuery(queries.cart());
  const remove = useRemoveFromCart();
  const order = usePlaceOrder();
  const { pay, payingId } = usePayOnline();
  const toast = useToast();
  const navigate = useNavigate();

  if (cart.isPending) {
    return (
      <div className="container-page">
        <PageHeader title={t("cart.title")} />
        <div className="flex animate-pulse flex-col gap-4">
          {[0, 1].map((i) => (
            <div key={i} className="h-36 rounded-2xl bg-surface-2" />
          ))}
        </div>
      </div>
    );
  }
  if (cart.isError) {
    return (
      <div className="container-page py-12">
        <ErrorState error={cart.error} onRetry={() => cart.refetch()} />
      </div>
    );
  }

  const { items, total } = cart.data;
  return (
    <div className="container-page">
      <PageHeader title={t("cart.title")} />
      {items.length === 0 ? (
        <EmptyState
          icon={<ShoppingBagOpen size={26} />}
          title={t("cart.empty")}
          body="Books you add will wait here until you're ready to order."
          action={
            <Link to="/books" className="btn-primary">
              {t("home.browse")}
            </Link>
          }
        />
      ) : (
        <div className="grid gap-10 lg:grid-cols-[1fr_320px]">
          <ul className="flex flex-col divide-y divide-line">
            {items.map(({ id, product: book }) => (
              <li key={id} className="flex gap-4 py-5 first:pt-0 sm:gap-6">
                <Link to="/books/$bookId" params={{ bookId: String(book.id) }} className="group w-20 shrink-0 sm:w-24">
                  <BookCover book={book} className="aspect-[3/4]" />
                </Link>
                <div className="flex min-w-0 flex-1 flex-col gap-3">
                  <div className="flex flex-col gap-0.5">
                    <Link
                      to="/books/$bookId"
                      params={{ bookId: String(book.id) }}
                      className="line-clamp-2 font-medium text-ink hover:text-accent"
                    >
                      {book.name}
                    </Link>
                    <p className="text-sm text-ink-3">{book.category.name}</p>
                    <p className="mt-1 text-sm font-semibold text-ink tabular-nums">{fmt.price(book.price)}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      className="btn-primary px-4 py-2"
                      disabled={payingId === book.id}
                      onClick={() => pay(book.id)}
                    >
                      <CreditCard size={16} /> {t("book.payOnline")}
                    </button>
                    <button
                      className="btn-secondary px-4 py-2"
                      disabled={order.isPending}
                      onClick={() =>
                        order.mutate(
                          { product_id: book.id, from_cart: true },
                          {
                            onSuccess: (r) => {
                              toast(`${r.message} Pay when it's delivered.`);
                              navigate({ to: "/orders" });
                            },
                            onError: (e) => toast(e.message, "error"),
                          },
                        )
                      }
                    >
                      <Package size={16} /> {t("book.buyNow")}
                    </button>
                    <button
                      className="btn-ghost px-3 py-2 text-ink-3 hover:text-danger"
                      disabled={remove.isPending}
                      onClick={() =>
                        remove.mutate(id, {
                          onSuccess: (r) => toast(r.message),
                          onError: (e) => toast(e.message, "error"),
                        })
                      }
                      aria-label={`Remove ${book.name} from cart`}
                    >
                      <Trash size={16} /> Remove
                    </button>
                  </div>
                </div>
              </li>
            ))}
          </ul>

          <aside className="h-fit rounded-2xl border border-line bg-surface p-6 lg:sticky lg:top-24">
            <h2 className="text-base font-semibold text-ink">Summary</h2>
            <dl className="mt-4 flex flex-col gap-2 text-sm">
              <div className="flex justify-between text-ink-2">
                <dt>
                  {items.length} {items.length === 1 ? "book" : "books"}
                </dt>
                <dd className="tabular-nums">{fmt.price(items.reduce((s, i) => s + i.product.mrp, 0))}</dd>
              </div>
              <div className="flex justify-between text-success">
                <dt>Discount</dt>
                <dd className="tabular-nums">
                  -{fmt.price(items.reduce((s, i) => s + i.product.mrp, 0) - total)}
                </dd>
              </div>
              <div className="mt-2 flex justify-between border-t border-line pt-3 text-base font-semibold text-ink">
                <dt>Total</dt>
                <dd className="tabular-nums">{fmt.price(total)}</dd>
              </div>
            </dl>
            <p className="mt-4 text-xs leading-relaxed text-ink-3">
              Each book is ordered on its own. Pay online now, or order and pay on delivery.
            </p>
          </aside>
        </div>
      )}
    </div>
  );
}
