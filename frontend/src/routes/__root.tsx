import { BookOpenText } from "@phosphor-icons/react";
import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Link, Outlet } from "@tanstack/react-router";

import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";
import { EmptyState, ErrorState } from "@/components/States";

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: RootLayout,
  notFoundComponent: () => (
    <div className="container-page py-16">
      <EmptyState
        icon={<BookOpenText size={26} />}
        title="This page isn't on our shelves"
        body="The link may be old, or the book may have been removed."
        action={
          <Link to="/" className="btn-primary">
            Back to home
          </Link>
        }
      />
    </div>
  ),
  errorComponent: ({ error, reset }) => (
    <div className="container-page py-16">
      <ErrorState error={error} onRetry={reset} />
    </div>
  ),
});

function RootLayout() {
  return (
    <div className="flex min-h-[100dvh] flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:rounded-full focus:bg-surface focus:px-4 focus:py-2 focus:shadow-lift"
      >
        Skip to content
      </a>
      <Header />
      <main id="main" className="flex-1">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
