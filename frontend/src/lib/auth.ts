import type { QueryClient } from "@tanstack/react-query";
import { redirect } from "@tanstack/react-router";

import { queries } from "./queries";

/** Route guard: sends signed-out visitors to /signin and brings them back afterwards. */
export async function requireUser(queryClient: QueryClient, returnTo: string) {
  const user = await queryClient.ensureQueryData(queries.me());
  if (!user) {
    throw redirect({ to: "/signin", search: { redirect: returnTo } });
  }
  return user;
}

/** Only allow same-site relative paths as post-login destinations. */
export function safeRedirect(path: string | undefined) {
  return path && path.startsWith("/") && !path.startsWith("//") ? path : "/";
}
