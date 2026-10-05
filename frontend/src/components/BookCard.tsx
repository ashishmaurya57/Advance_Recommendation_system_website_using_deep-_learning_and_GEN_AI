import { BookOpenText } from "@phosphor-icons/react";
import { Link } from "@tanstack/react-router";

import * as fmt from "@/lib/format";
import type { BookCard as Book } from "@/lib/types";

export function BookCover({ book, className = "" }: { book: Pick<Book, "image" | "name">; className?: string }) {
  return (
    <div className={`relative overflow-hidden rounded-2xl bg-surface-2 ${className}`}>
      {book.image ? (
        <img
          src={book.image}
          alt={`Cover of ${book.name}`}
          loading="lazy"
          className="h-full w-full object-cover transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] group-hover:scale-[1.03]"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center text-ink-3">
          <BookOpenText size={40} weight="light" />
        </div>
      )}
    </div>
  );
}

export function BookCard({ book }: { book: Book }) {
  const off = fmt.discount(book.price, book.mrp);
  return (
    <Link
      to="/books/$bookId"
      params={{ bookId: String(book.id) }}
      className="group flex flex-col gap-3 rounded-2xl outline-offset-4"
    >
      <BookCover book={book} className="aspect-[3/4] shadow-soft transition-shadow duration-300 group-hover:shadow-lift" />
      <div className="flex flex-col gap-1 px-0.5">
        <p className="text-xs text-ink-3">{book.category.name}</p>
        <h3 className="line-clamp-2 text-sm leading-snug font-medium text-ink group-hover:text-accent">{book.name}</h3>
        <p className="mt-0.5 flex items-baseline gap-2 text-sm">
          <span className="font-semibold text-ink">{fmt.price(book.price)}</span>
          {off > 0 && (
            <>
              <span className="text-xs text-ink-3 line-through">{fmt.price(book.mrp)}</span>
              <span className="text-xs font-medium text-success">{off}% off</span>
            </>
          )}
        </p>
      </div>
    </Link>
  );
}

export function BookGrid({ books, dense = false }: { books: Book[]; dense?: boolean }) {
  return (
    <div
      className={`grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 sm:gap-x-6 ${
        dense ? "md:grid-cols-4 lg:grid-cols-6" : "md:grid-cols-4 lg:grid-cols-5"
      }`}
    >
      {books.map((b) => (
        <BookCard key={b.id} book={b} />
      ))}
    </div>
  );
}

export function BookGridSkeleton({ count = 10, dense = false }: { count?: number; dense?: boolean }) {
  return (
    <div
      aria-hidden
      className={`grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 sm:gap-x-6 ${
        dense ? "md:grid-cols-4 lg:grid-cols-6" : "md:grid-cols-4 lg:grid-cols-5"
      }`}
    >
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="flex animate-pulse flex-col gap-3">
          <div className="aspect-[3/4] rounded-2xl bg-surface-2" />
          <div className="h-3 w-1/3 rounded-full bg-surface-2" />
          <div className="h-3.5 w-4/5 rounded-full bg-surface-2" />
          <div className="h-3.5 w-1/2 rounded-full bg-surface-2" />
        </div>
      ))}
    </div>
  );
}
