import { BookOpenText } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";

import { BookGrid, BookGridSkeleton } from "@/components/BookCard";
import { EmptyState, ErrorState, PageHeader } from "@/components/States";
import { queries } from "@/lib/queries";

type Sort = "newest" | "price-asc" | "price-desc";

export const Route = createFileRoute("/books/")({
  validateSearch: (s: Record<string, unknown>): { category?: number } => {
    const n = Number(s.category);
    return Number.isInteger(n) && n > 0 ? { category: n } : {};
  },
  component: BooksPage,
});

function BooksPage() {
  const { category } = Route.useSearch();
  const categories = useQuery(queries.categories());
  const books = useQuery(queries.books(category));
  const [sort, setSort] = useState<Sort>("newest");

  const sorted = useMemo(() => {
    const list = [...(books.data?.products ?? [])];
    if (sort === "price-asc") list.sort((a, b) => a.price - b.price);
    if (sort === "price-desc") list.sort((a, b) => b.price - a.price);
    return list;
  }, [books.data, sort]);

  const pill =
    "shrink-0 snap-start rounded-full border px-4 py-1.5 text-sm transition-colors whitespace-nowrap";
  const title = books.data?.category?.name ?? "All books";

  return (
    <div className="container-page">
      <PageHeader title={title}>
        {books.data && (
          <p>
            {sorted.length} {sorted.length === 1 ? "book" : "books"}
          </p>
        )}
      </PageHeader>

      <div className="flex flex-col gap-4 pb-8 md:flex-row md:items-center md:justify-between">
        <div className="-mx-4 flex snap-x gap-2 overflow-x-auto px-4 pb-1 md:mx-0 md:flex-wrap md:px-0">
          <Link
            to="/books"
            search={{}}
            className={`${pill} ${!category ? "border-ink bg-ink text-bg" : "border-line text-ink-2 hover:border-ink-3 hover:text-ink"}`}
          >
            All
          </Link>
          {categories.data?.map((c) => (
            <Link
              key={c.id}
              to="/books"
              search={{ category: c.id }}
              className={`${pill} ${
                category === c.id ? "border-ink bg-ink text-bg" : "border-line text-ink-2 hover:border-ink-3 hover:text-ink"
              }`}
            >
              {c.name}
            </Link>
          ))}
        </div>
        <label className="flex shrink-0 items-center gap-2 text-sm text-ink-2">
          Sort by
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as Sort)}
            className="rounded-full border border-line bg-surface px-3 py-1.5 text-sm text-ink focus:border-accent focus:outline-none"
          >
            <option value="newest">Newest</option>
            <option value="price-asc">Price: low to high</option>
            <option value="price-desc">Price: high to low</option>
          </select>
        </label>
      </div>

      {books.isPending ? (
        <BookGridSkeleton />
      ) : books.isError ? (
        <ErrorState error={books.error} onRetry={() => books.refetch()} />
      ) : sorted.length === 0 ? (
        <EmptyState
          icon={<BookOpenText size={26} />}
          title="No books in this genre yet"
          body="Try another genre, or browse everything we have."
          action={
            <Link to="/books" search={{}} className="btn-secondary">
              See all books
            </Link>
          }
        />
      ) : (
        <BookGrid books={sorted} />
      )}
    </div>
  );
}
