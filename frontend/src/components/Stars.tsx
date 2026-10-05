import { Star } from "@phosphor-icons/react";
import { useState } from "react";

export function Stars({ value, size = 14 }: { value: number; size?: number }) {
  return (
    <span className="inline-flex items-center gap-0.5 text-accent" aria-label={`${value} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <Star key={i} size={size} weight={i <= Math.round(value) ? "fill" : "regular"} />
      ))}
    </span>
  );
}

export function StarInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const [hover, setHover] = useState(0);
  const shown = hover || value;
  return (
    <div role="radiogroup" aria-label="Rating" className="flex items-center gap-1" onMouseLeave={() => setHover(0)}>
      {[1, 2, 3, 4, 5].map((i) => (
        <button
          key={i}
          type="button"
          role="radio"
          aria-checked={value === i}
          aria-label={`${i} star${i > 1 ? "s" : ""}`}
          onMouseEnter={() => setHover(i)}
          onClick={() => onChange(i)}
          className="rounded-full p-1 text-accent transition-transform hover:scale-110 active:scale-95"
        >
          <Star size={24} weight={i <= shown ? "fill" : "regular"} />
        </button>
      ))}
    </div>
  );
}
