import { MagnifyingGlass } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link } from "@tanstack/react-router";

import { BookGrid, BookGridSkeleton } from "@/components/BookCard";
import { EmptyState, ErrorState, PageHeader } from "@/components/States";
import { requireUser } from "@/lib/auth";
import { queries } from "@/lib/queries";

export const Route = createFileRoute("/search")({
  validateSearch: (s: Record<string, unknown>): { q: string } => ({ q: typeof s.q === "string" ? s.q : "" }),
  beforeLoad: ({ context, location }) => requireUser(context.queryClient, location.href),
  component: SearchPage,
});

function SearchPage() {
  const { q } = Route.useSearch();
  const query = q.trim();
  const search = useQuery({ ...queries.search(query), enabled: query.length > 0 });

  return (
    <div className="container-page">
      <PageHeader title={query ? <>Results for “{query}”</> : "Search"}>
        {search.data && search.data.products.length > 0 && (
          <p>
            {search.data.products.length} {search.data.products.length === 1 ? "book" : "books"}
          </p>
        )}
      </PageHeader>

      {!query ? (
        <EmptyState
          icon={<MagnifyingGlass size={26} />}
          title="What would you like to read?"
          body="Search by title, genre or publisher, or describe what you're in the mood for."
        />
      ) : search.isPending ? (
        <div className="flex flex-col gap-4">
          <p className="text-sm text-ink-3">Searching. Descriptive searches use the AI model and can take a few seconds.</p>
          <BookGridSkeleton count={5} />
        </div>
      ) : search.isError ? (
        <ErrorState error={search.error} onRetry={() => search.refetch()} />
      ) : search.data.products.length === 0 ? (
        <EmptyState
          icon={<MagnifyingGlass size={26} />}
          title={search.data.message ?? "No books found"}
          body="Try a genre like “horror” or “computer science”, a publisher, or describe a book you'd enjoy, for example “an exciting space adventure”."
          action={
            <Link to="/books" className="btn-secondary">
              Browse all books
            </Link>
          }
        />
      ) : (
        <BookGrid books={search.data.products} />
      )}
    </div>
  );
}
