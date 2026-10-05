import { Camera } from "@phosphor-icons/react";
import { useSuspenseQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState, type FormEvent } from "react";

import { Field } from "@/components/AuthLayout";
import { InterestPicker } from "@/components/InterestPicker";
import { PageHeader } from "@/components/States";
import { requireUser } from "@/lib/auth";
import { queries, useUpdateProfile } from "@/lib/queries";
import { useToast } from "@/lib/toast";
import type { User } from "@/lib/types";

export const Route = createFileRoute("/profile")({
  beforeLoad: ({ context, location }) => requireUser(context.queryClient, location.href),
  component: ProfilePage,
});

function ProfilePage() {
  const { data: user } = useSuspenseQuery(queries.me());
  // requireUser guarantees a user; guard for the moment after signing out.
  if (!user) return null;
  return (
    <div className="container-page">
      <PageHeader title="Your profile">
        <p>Your interests shape the books we pick for you on the home page.</p>
      </PageHeader>
      <ProfileForm user={user} />
    </div>
  );
}

function ProfileForm({ user }: { user: User }) {
  const update = useUpdateProfile();
  const toast = useToast();
  const [interests, setInterests] = useState(user.interests);
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    form.set("interests", interests.join(","));
    if (!(form.get("dob") as string)) form.delete("dob");
    const avatar = form.get("avatar");
    if (avatar instanceof File && !avatar.name) form.delete("avatar");
    update.mutate(form, {
      onSuccess: () => {
        toast("Your profile is updated. Your picks will refresh shortly.");
        (e.target as HTMLFormElement).querySelector<HTMLInputElement>("#new_password")!.value = "";
      },
    });
  };

  const avatar = preview ?? user.avatar;
  return (
    <form onSubmit={submit} className="grid gap-10 lg:grid-cols-[240px_1fr]">
      <div className="flex flex-col items-center gap-3 lg:items-start">
        <label className="group relative h-32 w-32 cursor-pointer overflow-hidden rounded-full bg-surface-2 ring-1 ring-line">
          {avatar ? (
            <img src={avatar} alt="Your profile photo" className="h-full w-full object-cover" />
          ) : (
            <span className="flex h-full w-full items-center justify-center text-4xl font-semibold text-ink-3">
              {user.name.charAt(0).toUpperCase()}
            </span>
          )}
          <span className="absolute inset-0 flex items-center justify-center bg-black/45 text-white opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
            <Camera size={24} />
          </span>
          <input
            name="avatar"
            type="file"
            accept="image/jpeg,image/png,image/webp,image/gif"
            className="sr-only"
            aria-label="Change profile photo"
            onChange={(e) => {
              const f = e.target.files?.[0];
              setPreview(f ? URL.createObjectURL(f) : null);
            }}
          />
        </label>
        <p className="text-sm text-ink-2">{user.email}</p>
      </div>

      <div className="flex max-w-2xl flex-col gap-6">
        <div className="grid gap-5 sm:grid-cols-2">
          <Field label="Full name" htmlFor="name">
            <input id="name" name="name" required defaultValue={user.name} className="field-input" />
          </Field>
          <Field label="Mobile" htmlFor="mobile">
            <input id="mobile" name="mobile" type="tel" defaultValue={user.mobile} className="field-input" />
          </Field>
          <Field label="Date of birth" htmlFor="dob">
            <input id="dob" name="dob" type="date" defaultValue={user.dob ?? ""} className="field-input" />
          </Field>
          <Field label="New password" htmlFor="new_password" hint="Leave blank to keep your current password.">
            <input
              id="new_password"
              name="new_password"
              type="password"
              autoComplete="new-password"
              minLength={6}
              className="field-input"
            />
          </Field>
        </div>
        <Field label="Address" htmlFor="address">
          <textarea id="address" name="address" rows={3} defaultValue={user.address} className="field-input resize-y" />
        </Field>
        <Field label="Interests" htmlFor="interests">
          <InterestPicker value={interests} onChange={setInterests} />
        </Field>
        {update.isError && (
          <p role="alert" className="text-sm text-danger">
            {update.error.message}
          </p>
        )}
        <button type="submit" className="btn-primary self-start px-6" disabled={update.isPending}>
          {update.isPending ? "Saving..." : "Save changes"}
        </button>
      </div>
    </form>
  );
}
