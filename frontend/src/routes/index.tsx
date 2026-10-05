import { ArrowRight, Sparkle } from "@phosphor-icons/react";
import { useQuery, useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { motion, useReducedMotion } from "motion/react";

import { BookCard, BookCover, BookGrid, BookGridSkeleton } from "@/components/BookCard";
import { useI18n } from "@/lib/i18n";
import { queries } from "@/lib/queries";
import type { BookCard as Book, Category } from "@/lib/types";

export const Route = createFileRoute("/")({
  loader: ({ context }) => context.queryClient.ensureQueryData(queries.home()),
  pendingComponent: () => (
    <div className="container-page py-16">
      <BookGridSkeleton />
    </div>
  ),
  component: HomePage,
});

function HomePage() {
  const { data } = useSuspenseQuery(queries.home());
  const { data: user } = useQuery(queries.me());
  return (
    <>
      <Hero covers={data.latest.filter((b) => b.image).slice(0, 3)} />
      {user && <ForYou hasInterests={user.interests.length > 0} />}
      <Genres categories={data.categories} />
      <Latest books={data.latest} />
    </>
  );
}

const ease = [0.16, 1, 0.3, 1] as const;

function Hero({ covers }: { covers: Book[] }) {
  const { t } = useI18n();
  const reduce = useReducedMotion();
  const enter = (delay: number) =>
    reduce ? {} : { initial: { opacity: 0, y: 16 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.7, delay, ease } };

  return (
    <section className="container-page grid items-center gap-12 pt-12 pb-16 md:grid-cols-[1.1fr_1fr] md:pt-20 md:pb-24">
      <div className="flex flex-col items-start gap-6">
        <motion.h1
          {...enter(0)}
          className="max-w-[14ch] text-4xl leading-[1.05] font-semibold tracking-tighter text-ink md:text-5xl lg:text-6xl"
        >
          {t("home.heroTitle")}
        </motion.h1>
        <motion.p {...enter(0.08)} className="max-w-[42ch] text-lg leading-relaxed text-ink-2">
          {t("home.heroSub")}
        </motion.p>
        <motion.div {...enter(0.16)} className="flex flex-wrap gap-3">
          <Link to="/books" className="btn-primary px-6 py-3 text-base">
            {t("home.browse")} <ArrowRight size={18} />
          </Link>
        </motion.div>
      </div>

      {covers.length > 0 && (
        <div className="relative mx-auto grid h-[340px] w-full max-w-md grid-cols-3 items-end sm:h-[420px]">
          {covers.map((book, i) => (
            <motion.div
              key={book.id}
              initial={reduce ? false : { opacity: 0, y: 40, rotate: 0 }}
              animate={{ opacity: 1, y: 0, rotate: [-6, 0, 6][i] ?? 0 }}
              transition={{ type: "spring", stiffness: 90, damping: 18, delay: 0.15 + i * 0.1 }}
              className={`${i === 1 ? "z-10 -mx-4 mb-10" : "z-0"}`}
            >
              <Link to="/books/$bookId" params={{ bookId: String(book.id) }} className="group block">
                <BookCover
                  book={book}
                  className={`aspect-[3/4] shadow-lift ring-1 ring-line ${i === 1 ? "scale-110" : ""}`}
                />
              </Link>
            </motion.div>
          ))}
        </div>
      )}
    </section>
  );
}

function ForYou({ hasInterests }: { hasInterests: boolean }) {
  const { t } = useI18n();
  const { data, isPending } = useQuery(queries.recommendations());
  const books = data?.products ?? [];
  const computing = isPending || (data?.computing && books.length === 0);

  return (
    <section className="border-y border-line bg-surface py-14">
      <div className="container-page flex flex-col gap-8">
        <div className="flex flex-col gap-1.5">
          <h2 className="flex items-center gap-2 text-2xl font-semibold tracking-tight text-ink">
            <Sparkle size={22} weight="fill" className="text-accent" />
            {t("home.forYou")}
          </h2>
          <p className="text-sm text-ink-2">{t("home.forYouSub")}</p>
        </div>

        {computing ? (
          <div className="flex flex-col gap-4">
            <BookGridSkeleton count={6} dense />
            <p className="text-sm text-ink-3">Working out what you might like. This can take up to a minute the first time.</p>
          </div>
        ) : books.length > 0 ? (
          <div className="-mx-4 flex snap-x snap-mandatory gap-5 overflow-x-auto px-4 pb-2 sm:-mx-6 sm:px-6 lg:-mx-8 lg:px-8">
            {books.map((b) => (
              <div key={b.id} className="w-40 shrink-0 snap-start sm:w-44">
                <BookCard book={b} />
              </div>
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-start gap-3 rounded-2xl bg-surface-2 p-6 sm:flex-row sm:items-center sm:justify-between">
            <p className="max-w-[60ch] text-sm leading-relaxed text-ink-2">
              {hasInterests
                ? "No strong matches yet. Open a few books, add some to your cart or leave a review and your picks will appear here."
                : "Tell us a few genres you enjoy and we'll start picking books for you."}
            </p>
            {!hasInterests && (
              <Link to="/profile" className="btn-secondary shrink-0">
                Add interests
              </Link>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

function Genres({ categories }: { categories: Category[] }) {
  const { t } = useI18n();
  if (categories.length === 0) return null;
  return (
    <section className="container-page flex flex-col gap-8 pt-16">
      <h2 className="text-2xl font-semibold tracking-tight text-ink">{t("home.categories")}</h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {categories.map((c, i) => (
          <Link
            key={c.id}
            to="/books"
            search={{ category: c.id }}
            className={`group relative isolate flex min-h-32 items-end overflow-hidden rounded-2xl bg-surface-2 p-4 ${
              i === 0 ? "col-span-2 row-span-2 min-h-[17rem] lg:col-span-2" : ""
            }`}
          >
            {c.image && (
              <img
                src={c.image}
                alt=""
                loading="lazy"
                className="absolute inset-0 -z-10 h-full w-full object-cover transition-transform duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] group-hover:scale-105"
              />
            )}
            <div className="absolute inset-0 -z-10 bg-gradient-to-t from-black/70 via-black/20 to-transparent" />
            <span className={`font-semibold text-white ${i === 0 ? "text-2xl" : "text-base"}`}>{c.name}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

function Latest({ books }: { books: Book[] }) {
  const { t } = useI18n();
  return (
    <section className="container-page flex flex-col gap-8 pt-20">
      <div className="flex items-end justify-between gap-4">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">{t("home.latest")}</h2>
        <Link to="/books" className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:text-accent-hover">
          {t("common.viewAll")} <ArrowRight size={14} />
        </Link>
      </div>
      <BookGrid books={books.slice(0, 10)} />
    </section>
  );
}
