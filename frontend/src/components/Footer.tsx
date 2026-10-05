import { Link } from "@tanstack/react-router";

import { useI18n } from "@/lib/i18n";

export function Footer() {
  const { t } = useI18n();
  return (
    <footer className="mt-24 border-t border-line">
      <div className="container-page flex flex-col gap-8 py-10 md:flex-row md:items-start md:justify-between">
        <div className="flex max-w-xs flex-col gap-2">
          <p className="text-base font-semibold text-ink">BookTown</p>
          <p className="text-sm leading-relaxed text-ink-2">
            A book store with a recommendation engine that learns from what you read and search.
          </p>
        </div>
        <nav className="grid grid-cols-2 gap-x-12 gap-y-2 text-sm" aria-label="Footer">
          <Link to="/books" className="text-ink-2 hover:text-ink">
            {t("nav.books")}
          </Link>
          <Link to="/about" className="text-ink-2 hover:text-ink">
            {t("nav.about")}
          </Link>
          <Link to="/orders" className="text-ink-2 hover:text-ink">
            {t("nav.orders")}
          </Link>
          <Link to="/contact" className="text-ink-2 hover:text-ink">
            {t("nav.contact")}
          </Link>
        </nav>
      </div>
      <div className="container-page border-t border-line py-5 text-xs text-ink-3">
        © {new Date().getFullYear()} BookTown
      </div>
    </footer>
  );
}
