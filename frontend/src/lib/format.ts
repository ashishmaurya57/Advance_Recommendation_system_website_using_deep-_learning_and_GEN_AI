const inr = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });

export const price = (n: number) => inr.format(n);

export const discount = (price: number, mrp: number) =>
  mrp > price ? Math.round(((mrp - price) / mrp) * 100) : 0;

export const date = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

export const titleCase = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());
