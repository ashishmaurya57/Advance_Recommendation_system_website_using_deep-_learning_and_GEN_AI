import {
  BookOpen,
  CreditCard,
  Package,
  ShoppingBagOpen,
  ThumbsDown,
  ThumbsUp,
} from "@phosphor-icons/react";
import { useQuery, useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link, notFound, useNavigate } from "@tanstack/react-router";
import { lazy, Suspense, useState, type FormEvent } from "react";

import { BookCover } from "@/components/BookCard";
import { StarInput, Stars } from "@/components/Stars";
import { ApiError } from "@/lib/api";
import * as fmt from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { queries, useAddReview, useAddToCart, usePlaceOrder, useReact } from "@/lib/queries";
import { useToast } from "@/lib/toast";
import type { BookDetail, Reaction } from "@/lib/types";
import { usePayOnline } from "@/lib/usePayOnline";

const PdfReader = lazy(() => import("@/components/PdfReader"));

export const Route = createFileRoute("/books/$bookId")({
  loader: async ({ context, params }) => {
    const id = Number(params.bookId);
    if (!Number.isInteger(id)) throw notFound();
    try {
      await context.queryClient.ensureQueryData(queries.book(id));
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) throw notFound();
      throw e;
    }
  },
  pendingComponent: DetailSkeleton,
  component: BookPage,
});

function DetailSkeleton() {
  return (
    <div className="container-page grid animate-pulse gap-10 py-12 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="aspect-[3/4] rounded-2xl bg-surface-2" />
      <div className="flex flex-col gap-4">
        <div className="h-4 w-24 rounded-full bg-surface-2" />
        <div className="h-9 w-3/4 rounded-full bg-surface-2" />
        <div className="h-6 w-32 rounded-full bg-surface-2" />
        <div className="mt-6 h-24 rounded-2xl bg-surface-2" />
      </div>
    </div>
  );
}

function BookPage() {
  const bookId = Number(Route.useParams().bookId);
  const { data: book } = useSuspenseQuery(queries.book(bookId));
  const [reading, setReading] = useState(false);

  return (
    <div className="container-page">
      <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 pt-8 text-sm text-ink-3">
        <Link to="/books" className="hover:text-ink">
          Books
        </Link>
        <span>/</span>
        <Link to="/books" search={{ category: book.category.id }} className="hover:text-ink">
          {book.category.name}
        </Link>
      </nav>

      <div className="grid gap-10 pt-6 pb-12 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] lg:gap-16">
        <div className="md:sticky md:top-24 md:self-start">
          <BookCover book={book} className="mx-auto aspect-[3/4] max-w-sm shadow-lift md:max-w-none" />
        </div>
        <div className="flex flex-col gap-8">
          <Summary book={book} onRead={() => setReading(true)} />
          <Details book={book} />
          <Reviews book={book} />
        </div>
      </div>

      {reading && book.pdf_url && (
        <Suspense fallback={null}>
          <PdfReader
            bookId={book.id}
            title={book.name}
            pdfUrl={book.pdf_url}
            originalLanguage={book.language}
            onClose={() => setReading(false)}
          />
        </Suspense>
      )}
    </div>
  );
}

function Summary({ book, onRead }: { book: BookDetail; onRead: () => void }) {
  const { t } = useI18n();
  const { data: user } = useQuery(queries.me());
  const navigate = useNavigate();
  const toast = useToast();
  const addToCart = useAddToCart();
  const order = usePlaceOrder();
  const { pay, payingId } = usePayOnline();
  const off = fmt.discount(book.price, book.mrp);

  const needLogin = () => navigate({ to: "/signin", search: { redirect: `/books/${book.id}` } });
  const onError = (e: Error) => toast(e.message, "error");

  return (
    <section className="flex flex-col gap-5">
      <div className="flex flex-col gap-2">
        <p className="text-sm text-ink-3">{book.category.name}</p>
        <h1 className="text-3xl leading-tight font-semibold tracking-tight text-ink md:text-4xl">{book.name}</h1>
        <p className="text-sm text-ink-2">{book.publisher}</p>
        {book.average_rating !== null && (
          <a href="#reviews" className="flex w-fit items-center gap-2 text-sm text-ink-2 hover:text-ink">
            <Stars value={book.average_rating} />
            {book.average_rating.toFixed(1)} ({book.reviews.length})
          </a>
        )}
      </div>

      <div className="flex items-baseline gap-3">
        <span className="text-3xl font-semibold text-ink tabular-nums">{fmt.price(book.price)}</span>
        {off > 0 && (
          <>
            <span className="text-base text-ink-3 line-through tabular-nums">{fmt.price(book.mrp)}</span>
            <span className="rounded-full bg-accent-soft px-2.5 py-0.5 text-sm font-medium text-accent">{off}% off</span>
          </>
        )}
      </div>

      <div className="flex flex-wrap gap-3">
        <button
          className="btn-primary"
          disabled={addToCart.isPending}
          onClick={() =>
            user
              ? addToCart.mutate(book.id, { onSuccess: (r) => toast(r.message), onError })
              : needLogin()
          }
        >
          <ShoppingBagOpen size={18} /> {t("book.addToCart")}
        </button>
        <button className="btn-secondary" disabled={payingId === book.id} onClick={() => pay(book.id)}>
          <CreditCard size={18} /> {t("book.payOnline")}
        </button>
        <button
          className="btn-secondary"
          disabled={order.isPending}
          onClick={() =>
            user
              ? order.mutate(
                  { product_id: book.id },
                  {
                    onSuccess: (r) => {
                      toast(`${r.message} Pay when it's delivered.`);
                      navigate({ to: "/orders" });
                    },
                    onError,
                  },
                )
              : needLogin()
          }
        >
          <Package size={18} /> {t("book.buyNow")}
        </button>
        {book.pdf_url && (
          <button className="btn-ghost" onClick={onRead}>
            <BookOpen size={18} /> {t("book.read")}
          </button>
        )}
      </div>

      <ReactionButtons book={book} onNeedLogin={needLogin} />
    </section>
  );
}

function ReactionButtons({ book, onNeedLogin }: { book: BookDetail; onNeedLogin: () => void }) {
  const { data: user } = useQuery(queries.me());
  const react = useReact(book.id);
  const toast = useToast();
  const click = (action: Reaction) =>
    user ? react.mutate(action, { onError: (e) => toast(e.message, "error") }) : onNeedLogin();

  const base =
    "inline-flex items-center gap-1.5 rounded-full border px-3.5 py-1.5 text-sm tabular-nums transition-colors active:scale-[0.97]";
  const on = "border-accent bg-accent-soft text-accent";
  const off = "border-line text-ink-2 hover:border-ink-3 hover:text-ink";
  return (
    <div className="flex items-center gap-2">
      <button
        className={`${base} ${book.my_reaction === "like" ? on : off}`}
        aria-pressed={book.my_reaction === "like"}
        disabled={react.isPending}
        onClick={() => click("like")}
      >
        <ThumbsUp size={16} weight={book.my_reaction === "like" ? "fill" : "regular"} /> {book.likes}
        <span className="sr-only">likes</span>
      </button>
      <button
        className={`${base} ${book.my_reaction === "dislike" ? on : off}`}
        aria-pressed={book.my_reaction === "dislike"}
        disabled={react.isPending}
        onClick={() => click("dislike")}
      >
        <ThumbsDown size={16} weight={book.my_reaction === "dislike" ? "fill" : "regular"} /> {book.dislikes}
        <span className="sr-only">dislikes</span>
      </button>
    </div>
  );
}

function Details({ book }: { book: BookDetail }) {
  const facts = [
    ["Language", fmt.titleCase(book.language)],
    ["Format", book.hardcover],
    ["Publisher", book.publisher],
    ["Published", fmt.date(book.published)],
  ];
  return (
    <section className="flex flex-col gap-6 border-t border-line pt-8">
      <p className="max-w-[68ch] text-base leading-relaxed whitespace-pre-line text-ink-2">{book.description}</p>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-4 rounded-2xl bg-surface-2 p-5 sm:grid-cols-4">
        {facts.map(([label, value]) => (
          <div key={label} className="flex flex-col gap-0.5">
            <dt className="text-xs text-ink-3">{label}</dt>
            <dd className="text-sm font-medium text-ink">{value || "Not listed"}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Reviews({ book }: { book: BookDetail }) {
  const { t } = useI18n();
  const { data: user } = useQuery(queries.me());
  return (
    <section id="reviews" className="flex scroll-mt-24 flex-col gap-6 border-t border-line pt-8">
      <h2 className="text-xl font-semibold tracking-tight text-ink">
        {t("book.reviews")} <span className="text-ink-3">({book.reviews.length})</span>
      </h2>

      {user ? (
        <ReviewForm bookId={book.id} />
      ) : (
        <p className="text-sm text-ink-2">
          <Link to="/signin" search={{ redirect: `/books/${book.id}` }} className="font-medium text-accent hover:text-accent-hover">
            {t("nav.signin")}
          </Link>{" "}
          to write a review.
        </p>
      )}

      {book.reviews.length === 0 ? (
        <p className="text-sm text-ink-3">{t("book.noReviews")}</p>
      ) : (
        <ul className="flex flex-col gap-6">
          {book.reviews.map((r) => (
            <li key={r.id} className="flex gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-surface-2 text-sm font-semibold text-ink-2">
                {r.user_name.charAt(0).toUpperCase()}
              </div>
              <div className="flex min-w-0 flex-col gap-1">
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="text-sm font-medium text-ink">{r.user_name}</span>
                  <Stars value={r.rating} size={12} />
                  <span className="text-xs text-ink-3">{fmt.date(r.created_at)}</span>
                </div>
                {r.comment && <p className="max-w-[65ch] text-sm leading-relaxed text-ink-2">{r.comment}</p>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function ReviewForm({ bookId }: { bookId: number }) {
  const { t } = useI18n();
  const add = useAddReview(bookId);
  const toast = useToast();
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!rating) return setError("Choose a star rating.");
    setError(null);
    add.mutate(
      { rating, comment },
      {
        onSuccess: () => {
          toast("Thanks, your review is posted.");
          setRating(0);
          setComment("");
        },
        onError: (e) => setError(e.message),
      },
    );
  };

  return (
    <form onSubmit={submit} className="flex flex-col gap-4 rounded-2xl border border-line bg-surface p-5">
      <p className="text-sm font-medium text-ink">{t("book.writeReview")}</p>
      <StarInput value={rating} onChange={setRating} />
      <div className="flex flex-col gap-2">
        <label htmlFor="review-comment" className="field-label">
          Your thoughts (optional)
        </label>
        <textarea
          id="review-comment"
          rows={3}
          maxLength={1000}
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          className="field-input resize-y"
        />
      </div>
      {error && <p className="text-sm text-danger">{error}</p>}
      <button type="submit" className="btn-primary self-start" disabled={add.isPending}>
        {add.isPending ? "Posting..." : t("book.submit")}
      </button>
    </form>
  );
}
