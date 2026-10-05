import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";

import { AuthLayout, Field } from "@/components/AuthLayout";
import { safeRedirect } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { queries, useSignIn } from "@/lib/queries";
import { useToast } from "@/lib/toast";

export const Route = createFileRoute("/signin")({
  validateSearch: (s: Record<string, unknown>): { redirect?: string } =>
    typeof s.redirect === "string" ? { redirect: s.redirect } : {},
  beforeLoad: async ({ context, search }) => {
    if (await context.queryClient.ensureQueryData(queries.me())) {
      throw redirect({ to: safeRedirect(search.redirect) });
    }
  },
  component: SignInPage,
});

function SignInPage() {
  const { t } = useI18n();
  const { redirect: returnTo } = Route.useSearch();
  const signIn = useSignIn();
  const navigate = useNavigate();
  const toast = useToast();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    signIn.mutate(
      { email: email.trim(), password },
      {
        onSuccess: (user) => {
          toast(`Welcome back, ${user.name.split(" ")[0]}.`);
          navigate({ to: safeRedirect(returnTo) });
        },
      },
    );
  };

  return (
    <AuthLayout
      title={t("nav.signin")}
      subtitle={
        <>
          New to BookTown?{" "}
          <Link to="/signup" search={returnTo ? { redirect: returnTo } : {}} className="font-medium text-accent hover:text-accent-hover">
            {t("nav.signup")}
          </Link>
        </>
      }
    >
      <form onSubmit={submit} className="flex flex-col gap-5">
        <Field label="Email" htmlFor="email">
          <input
            id="email"
            type="email"
            autoComplete="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="field-input"
          />
        </Field>
        <Field label="Password" htmlFor="password">
          <input
            id="password"
            type="password"
            autoComplete="current-password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="field-input"
          />
        </Field>
        {signIn.isError && (
          <p role="alert" className="text-sm text-danger">
            {signIn.error.message}
          </p>
        )}
        <button type="submit" className="btn-primary w-full py-3" disabled={signIn.isPending}>
          {signIn.isPending ? "Signing in..." : t("nav.signin")}
        </button>
      </form>
    </AuthLayout>
  );
}
