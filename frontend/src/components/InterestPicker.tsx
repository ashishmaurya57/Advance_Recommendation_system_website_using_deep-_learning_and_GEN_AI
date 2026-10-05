import { Plus, X } from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { useState, type KeyboardEvent } from "react";

import { queries } from "@/lib/queries";

export const MAX_INTERESTS = 5;

/** Choose up to five interests: genre suggestions plus free-text tags. */
export function InterestPicker({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  const { data: categories } = useQuery(queries.categories());
  const [draft, setDraft] = useState("");
  const full = value.length >= MAX_INTERESTS;
  const has = (name: string) => value.some((v) => v.toLowerCase() === name.toLowerCase());

  const add = (name: string) => {
    const tag = name.trim();
    if (!tag || has(tag) || full) return;
    onChange([...value, tag]);
    setDraft("");
  };
  const remove = (name: string) => onChange(value.filter((v) => v !== name));

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      add(draft);
    } else if (e.key === "Backspace" && !draft && value.length) {
      remove(value[value.length - 1]);
    }
  };

  const suggestions = (categories ?? []).map((c) => c.name).filter((n) => !has(n));

  return (
    <div className="flex flex-col gap-3">
      <div className="field-input flex min-h-11 flex-wrap items-center gap-1.5 py-1.5 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/25">
        {value.map((tag) => (
          <span key={tag} className="inline-flex items-center gap-1 rounded-full bg-accent-soft py-1 pr-1.5 pl-3 text-sm text-accent">
            {tag}
            <button
              type="button"
              onClick={() => remove(tag)}
              aria-label={`Remove ${tag}`}
              className="rounded-full p-0.5 hover:bg-accent/15"
            >
              <X size={12} />
            </button>
          </span>
        ))}
        <input
          id="interests"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKey}
          onBlur={() => add(draft)}
          disabled={full}
          placeholder={full ? "" : value.length ? "Add another" : "Type an interest and press Enter"}
          className="min-w-32 flex-1 bg-transparent py-1 text-sm text-ink placeholder:text-ink-3 focus:outline-none disabled:cursor-not-allowed"
        />
      </div>
      {suggestions.length > 0 && !full && (
        <div className="flex flex-wrap gap-1.5">
          {suggestions.map((name) => (
            <button
              key={name}
              type="button"
              onClick={() => add(name)}
              className="inline-flex items-center gap-1 rounded-full border border-line px-3 py-1 text-xs text-ink-2 transition-colors hover:border-accent hover:text-accent"
            >
              <Plus size={10} /> {name}
            </button>
          ))}
        </div>
      )}
      <p className="text-xs text-ink-3">
        {value.length} of {MAX_INTERESTS}. Your searches add to these over time.
      </p>
    </div>
  );
}
