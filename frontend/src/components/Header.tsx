import {
  List,
  MagnifyingGlass,
  Moon,
  Package,
  ShoppingBagOpen,
  SignOut,
  Sun,
  Translate,
  UserCircle,
  X,
} from "@phosphor-icons/react";
import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState, type FormEvent } from "react";

import { LANGUAGES, useI18n, type Lang } from "@/lib/i18n";
import { queries, useSignOut } from "@/lib/queries";
import { useToast } from "@/lib/toast";

function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("theme", next ? "dark" : "light");
    } catch {
      // storage blocked
    }
    setDark(next);
  };
  return { dark, toggle };
}

function SearchBox({ onDone, autoFocus = false }: { onDone?: () => void; autoFocus?: boolean }) {
  const { t } = useI18n();
  const navigate = useNavigate();
  const current = useRouterState({ select: (s) => (s.location.search as { q?: string }).q ?? "" });
  const [q, setQ] = useState(current);

  useEffect(() => setQ(current), [current]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const query = q.trim();
    if (!query) return;
    navigate({ to: "/search", search: { q: query } });
    onDone?.();
  };

  return (
    <form onSubmit={submit} role="search" className="relative w-full">
      <MagnifyingGlass size={16} className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-ink-3" />
      <input
        type="search"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={t("search.placeholder")}
        aria-label={t("search.placeholder")}
        autoFocus={autoFocus}
        className="w-full rounded-full border border-line bg-surface-2 py-2 pr-4 pl-9 text-sm text-ink placeholder:text-ink-3 transition-colors focus:border-accent focus:bg-surface focus:outline-none"
      />
    </form>
  );
}

function AccountMenu() {
  const { t } = useI18n();
  const { data: user } = useQuery(queries.me());
  const signOut = useSignOut();
  const navigate = useNavigate();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  if (!user) {
    return (
      <Link to="/signin" className="btn-primary px-4 py-2">
        {t("nav.signin")}
      </Link>
    );
  }

  const item = "flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-sm text-ink-2 hover:bg-surface-2 hover:text-ink";
  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="menu"
        aria-label="Account menu"
        className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-full border border-line bg-surface-2 transition-colors hover:border-ink-3"
      >
        {user.avatar ? (
          <img src={user.avatar} alt="" className="h-full w-full object-cover" />
        ) : (
          <span className="text-sm font-semibold text-ink-2">{user.name.charAt(0).toUpperCase()}</span>
        )}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            role="menu"
            initial={reduce ? false : { opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, y: -6, scale: 0.98 }}
            transition={{ duration: 0.16, ease: [0.16, 1, 0.3, 1] }}
            className="absolute right-0 z-40 mt-2 w-60 origin-top-right rounded-2xl border border-line bg-surface p-1.5 shadow-lift"
          >
            <div className="px-3 pt-2 pb-2.5">
              <p className="truncate text-sm font-medium text-ink">{user.name}</p>
              <p className="truncate text-xs text-ink-3">{user.email}</p>
            </div>
            <Link to="/profile" role="menuitem" className={item} onClick={() => setOpen(false)}>
              <UserCircle size={18} /> {t("nav.profile")}
            </Link>
            <Link to="/orders" role="menuitem" className={item} onClick={() => setOpen(false)}>
              <Package size={18} /> {t("nav.orders")}
            </Link>
            <button
              role="menuitem"
              className={item}
              disabled={signOut.isPending}
              onClick={() =>
                signOut.mutate(undefined, {
                  onSuccess: () => {
                    setOpen(false);
                    toast("You've been signed out.");
                    navigate({ to: "/" });
                  },
                })
              }
            >
              <SignOut size={18} /> {t("nav.signout")}
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function CartButton() {
  const { t } = useI18n();
  const { data: user } = useQuery(queries.me());
  const { data: cart } = useQuery({ ...queries.cart(), enabled: !!user });
  const count = cart?.items.length ?? 0;
  if (!user) return null;
  return (
    <Link
      to="/cart"
      aria-label={`${t("nav.cart")} (${count})`}
      className="relative flex h-9 w-9 items-center justify-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
    >
      <ShoppingBagOpen size={20} />
      {count > 0 && (
        <span className="absolute -top-0.5 -right-0.5 flex h-4.5 min-w-4.5 items-center justify-center rounded-full bg-accent px-1 text-[10px] font-semibold text-on-accent">
          {count}
        </span>
      )}
    </Link>
  );
}

function Preferences() {
  const { lang, setLang } = useI18n();
  const { dark, toggle } = useTheme();
  return (
    <div className="flex items-center gap-1">
      <label className="relative flex h-9 items-center rounded-full text-ink-2 hover:bg-surface-2 hover:text-ink">
        <Translate size={18} className="pointer-events-none absolute left-2.5" />
        <span className="sr-only">Language</span>
        <select
          value={lang}
          onChange={(e) => setLang(e.target.value as Lang)}
          className="h-9 cursor-pointer appearance-none rounded-full bg-transparent pr-3 pl-8 text-sm focus:outline-none"
        >
          {Object.entries(LANGUAGES).map(([code, name]) => (
            <option key={code} value={code}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <button
        onClick={toggle}
        aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
        className="flex h-9 w-9 items-center justify-center rounded-full text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink"
      >
        {dark ? <Sun size={18} /> : <Moon size={18} />}
      </button>
    </div>
  );
}

export function Header() {
  const { t } = useI18n();
  const [mobileOpen, setMobileOpen] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const reduce = useReducedMotion();

  useEffect(() => setMobileOpen(false), [pathname]);

  const links = [
    { to: "/", label: t("nav.home") },
    { to: "/books", label: t("nav.books") },
    { to: "/about", label: t("nav.about") },
    { to: "/contact", label: t("nav.contact") },
  ] as const;

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-bg/85 backdrop-blur-md">
      <div className="container-page flex h-16 items-center gap-4">
        <Link to="/" className="flex shrink-0 items-center gap-2 text-lg font-semibold tracking-tight text-ink">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm font-bold text-on-accent">
            B
          </span>
          BookTown
        </Link>

        <nav className="ml-4 hidden items-center gap-1 lg:flex" aria-label="Main">
          {links.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              activeOptions={{ exact: l.to === "/" }}
              className="rounded-full px-3 py-1.5 text-sm text-ink-2 transition-colors hover:text-ink data-[status=active]:bg-surface-2 data-[status=active]:text-ink"
            >
              {l.label}
            </Link>
          ))}
        </nav>

        <div className="ml-auto hidden max-w-sm flex-1 md:block">
          <SearchBox />
        </div>

        <div className="ml-auto flex items-center gap-1 md:ml-2">
          <div className="hidden sm:block">
            <Preferences />
          </div>
          <CartButton />
          <div className="hidden sm:block">
            <AccountMenu />
          </div>
          <button
            onClick={() => setMobileOpen((o) => !o)}
            aria-expanded={mobileOpen}
            aria-label="Menu"
            className="flex h-9 w-9 items-center justify-center rounded-full text-ink-2 hover:bg-surface-2 lg:hidden"
          >
            {mobileOpen ? <X size={20} /> : <List size={20} />}
          </button>
        </div>
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={reduce ? false : { opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={reduce ? { opacity: 0 } : { opacity: 0, height: 0 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden border-t border-line lg:hidden"
          >
            <div className="container-page flex flex-col gap-4 py-4">
              <div className="md:hidden">
                <SearchBox onDone={() => setMobileOpen(false)} />
              </div>
              <nav className="flex flex-col" aria-label="Mobile">
                {links.map((l) => (
                  <Link
                    key={l.to}
                    to={l.to}
                    activeOptions={{ exact: l.to === "/" }}
                    className="rounded-xl px-3 py-2.5 text-ink-2 data-[status=active]:bg-surface-2 data-[status=active]:text-ink"
                  >
                    {l.label}
                  </Link>
                ))}
              </nav>
              <div className="flex items-center justify-between gap-3 sm:hidden">
                <Preferences />
                <AccountMenu />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
