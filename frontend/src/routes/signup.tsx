import { createFileRoute, Link, redirect, useNavigate } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";

import { AuthLayout, Field } from "@/components/AuthLayout";
import { InterestPicker } from "@/components/InterestPicker";
import { safeRedirect } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { queries, useSignUp } from "@/lib/queries";
import { useToast } from "@/lib/toast";

export const Route = createFileRoute("/signup")({
  validateSearch: (s: Record<string, unknown>): { redirect?: string } =>
    typeof s.redirect === "string" ? { redirect: s.redirect } : {},
  beforeLoad: async ({ context }) => {
    if (await context.queryClient.ensureQueryData(queries.me())) throw redirect({ to: "/" });
  },
  component: SignUpPage,
});

function SignUpPage() {
  const { t } = useI18n();
  const { redirect: returnTo } = Route.useSearch();
  const signUp = useSignUp();
  const navigate = useNavigate();
  const toast = useToast();
  const [interests, setInterests] = useState<string[]>([]);

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    form.set("interests", interests.join(","));
    if (!(form.get("dob") as string)) form.delete("dob");
    const avatar = form.get("avatar");
    if (avatar instanceof File && !avatar.name) form.delete("avatar");
    signUp.mutate(form, {
      onSuccess: (user) => {
        toast(`Welcome to BookTown, ${user.name.split(" ")[0]}.`);
        navigate({ to: safeRedirect(returnTo) });
      },
    });
  };

  return (
    <AuthLayout
      title={t("nav.signup")}
      subtitle={
        <>
          Already have an account?{" "}
          <Link to="/signin" search={returnTo ? { redirect: returnTo } : {}} className="font-medium text-accent hover:text-accent-hover">
            {t("nav.signin")}
          </Link>
        </>
      }
    >
      <form onSubmit={submit} className="flex flex-col gap-5">
        <Field label="Full name" htmlFor="name">
          <input id="name" name="name" autoComplete="name" required className="field-input" />
        </Field>
        <Field label="Email" htmlFor="email">
          <input id="email" name="email" type="email" autoComplete="email" required className="field-input" />
        </Field>
        <Field label="Password" htmlFor="password" hint="At least 6 characters.">
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="new-password"
            minLength={6}
            required
            className="field-input"
          />
        </Field>
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Mobile" htmlFor="mobile">
            <input id="mobile" name="mobile" type="tel" autoComplete="tel" className="field-input" />
          </Field>
          <Field label="Date of birth" htmlFor="dob">
            <input id="dob" name="dob" type="date" className="field-input" />
          </Field>
        </div>
        <Field label="Address" htmlFor="address">
          <textarea id="address" name="address" rows={2} autoComplete="street-address" className="field-input resize-y" />
        </Field>
        <Field label="What do you like to read?" htmlFor="interests">
          <InterestPicker value={interests} onChange={setInterests} />
        </Field>
        <Field label="Profile photo (optional)" htmlFor="avatar">
          <input
            id="avatar"
            name="avatar"
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            className="text-sm text-ink-2 file:mr-3 file:rounded-full file:border-0 file:bg-surface-2 file:px-4 file:py-2 file:text-sm file:text-ink hover:file:bg-line"
          />
        </Field>
        {signUp.isError && (
          <p role="alert" className="text-sm text-danger">
            {signUp.error.message}
          </p>
        )}
        <button type="submit" className="btn-primary w-full py-3" disabled={signUp.isPending}>
          {signUp.isPending ? "Creating your account..." : t("nav.signup")}
        </button>
      </form>
    </AuthLayout>
  );
}
