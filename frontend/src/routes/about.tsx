import { Brain, ChatCircleText, Cursor, MagnifyingGlass } from "@phosphor-icons/react";
import { createFileRoute, Link } from "@tanstack/react-router";
import type { ReactNode } from "react";

export const Route = createFileRoute("/about")({ component: AboutPage });

const signals: { icon: ReactNode; title: string; body: string }[] = [
  {
    icon: <Brain size={22} />,
    title: "Your interests",
    body: "A language model scores how well each book fits the genres and topics you've told us you like.",
  },
  {
    icon: <Cursor size={22} />,
    title: "What you do",
    body: "Opening a book, adding it to your cart and rating it all count. Books similar to the ones you engage with move up.",
  },
  {
    icon: <MagnifyingGlass size={22} />,
    title: "What you search",
    body: "Genres you search for, and descriptive searches like “a cosy mystery”, are added to your interests.",
  },
  {
    icon: <ChatCircleText size={22} />,
    title: "How books read",
    body: "Sentence embeddings and sentiment analysis compare descriptions, so matches are about meaning, not just keywords.",
  },
];

function AboutPage() {
  return (
    <div className="container-page">
      <section className="grid gap-10 pt-12 pb-16 md:grid-cols-[1.2fr_1fr] md:pt-20">
        <div className="flex flex-col gap-5">
          <h1 className="max-w-[18ch] text-4xl leading-[1.05] font-semibold tracking-tighter text-ink md:text-5xl">
            A page for every book, and a shelf for every reader
          </h1>
          <p className="max-w-[55ch] text-lg leading-relaxed text-ink-2">
            BookTown is an open library catalogue working towards a page for every book ever published. It's a
            lofty goal, but an achievable one.
          </p>
        </div>
        <div className="flex flex-col justify-end gap-4 text-base leading-relaxed text-ink-2">
          <p>
            BookTown is an open project and we welcome contributions. It is non-profit and built to help people find
            books worth their time.
          </p>
          <p>Thank you to the team who designed and built it.</p>
        </div>
      </section>

      <section className="rounded-3xl bg-surface p-6 ring-1 ring-line sm:p-10">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">How your picks are chosen</h2>
        <div className="mt-8 grid gap-x-10 gap-y-8 sm:grid-cols-2">
          {signals.map((s) => (
            <div key={s.title} className="flex gap-4">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-accent-soft text-accent">
                {s.icon}
              </div>
              <div className="flex flex-col gap-1">
                <h3 className="font-semibold text-ink">{s.title}</h3>
                <p className="max-w-[48ch] text-sm leading-relaxed text-ink-2">{s.body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="flex flex-col items-start gap-4 pt-16">
        <h2 className="text-2xl font-semibold tracking-tight text-ink">Questions or ideas?</h2>
        <Link to="/contact" className="btn-primary">
          Contact us
        </Link>
      </section>
    </div>
  );
}
