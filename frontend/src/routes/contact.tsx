import { CheckCircle } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";

import { Field } from "@/components/AuthLayout";
import { PageHeader } from "@/components/States";
import { queries, useContact } from "@/lib/queries";

export const Route = createFileRoute("/contact")({ component: ContactPage });

function ContactPage() {
  const { data: user } = useQuery(queries.me());
  const contact = useContact();
  const [sent, setSent] = useState<string | null>(null);

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    contact.mutate(
      {
        name: String(f.get("name")),
        email: String(f.get("email")),
        mobile: String(f.get("mobile") ?? ""),
        message: String(f.get("message")),
      },
      { onSuccess: (r) => setSent(r.message) },
    );
  };

  return (
    <div className="container-page">
      <PageHeader title="Contact us">
        <p>Questions about an order, a book request, or feedback on your recommendations. We read every message.</p>
      </PageHeader>

      <div className="max-w-2xl">
        {sent ? (
          <div role="status" className="flex items-start gap-3 rounded-2xl border border-line bg-surface p-6">
            <CheckCircle size={24} weight="fill" className="shrink-0 text-success" />
            <div className="flex flex-col gap-3">
              <p className="text-ink">{sent}</p>
              <button className="btn-secondary self-start" onClick={() => setSent(null)}>
                Send another message
              </button>
            </div>
          </div>
        ) : (
          <form onSubmit={submit} className="flex flex-col gap-5 rounded-2xl border border-line bg-surface p-6 sm:p-8">
            <div className="grid gap-5 sm:grid-cols-2">
              <Field label="Name" htmlFor="name">
                <input id="name" name="name" required defaultValue={user?.name} autoComplete="name" className="field-input" />
              </Field>
              <Field label="Email" htmlFor="email">
                <input
                  id="email"
                  name="email"
                  type="email"
                  required
                  defaultValue={user?.email}
                  autoComplete="email"
                  className="field-input"
                />
              </Field>
            </div>
            <Field label="Mobile (optional)" htmlFor="mobile">
              <input id="mobile" name="mobile" type="tel" defaultValue={user?.mobile} autoComplete="tel" className="field-input" />
            </Field>
            <Field label="Message" htmlFor="message">
              <textarea id="message" name="message" rows={5} maxLength={600} required className="field-input resize-y" />
            </Field>
            {contact.isError && (
              <p role="alert" className="text-sm text-danger">
                {contact.error.message}
              </p>
            )}
            <button type="submit" className="btn-primary self-start px-6" disabled={contact.isPending}>
              {contact.isPending ? "Sending..." : "Send message"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
