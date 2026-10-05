import "./styles.css";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRouter, RouterProvider } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { ApiError } from "./lib/api";
import { I18nProvider } from "./lib/i18n";
import { ToastProvider } from "./lib/toast";
import { routeTree } from "./routeTree.gen";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      // Don't retry auth / not-found errors; retry flaky network once.
      retry: (count, error) => !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 1,
    },
  },
});

const router = createRouter({
  routeTree,
  context: { queryClient },
  defaultPreload: "intent",
  defaultPreloadStaleTime: 0,
  scrollRestoration: true,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <ToastProvider>
          <RouterProvider router={router} />
        </ToastProvider>
      </I18nProvider>
    </QueryClientProvider>
  </StrictMode>,
);
