import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export const LANGUAGES = {
  en: "English",
  hi: "हिन्दी",
  fr: "Français",
  es: "Español",
  de: "Deutsch",
} as const;

export type Lang = keyof typeof LANGUAGES;

const en = {
  "nav.home": "Home",
  "nav.books": "Books",
  "nav.about": "About",
  "nav.contact": "Contact",
  "nav.cart": "Cart",
  "nav.orders": "My orders",
  "nav.profile": "Profile",
  "nav.signin": "Sign in",
  "nav.signup": "Create account",
  "nav.signout": "Sign out",
  "search.placeholder": "Search titles, genres, publishers",
  "home.heroTitle": "Find your next favourite book",
  "home.heroSub": "Browse the shelves, or let BookTown learn what you love and pick for you.",
  "home.browse": "Browse books",
  "home.categories": "Shop by genre",
  "home.forYou": "Picked for you",
  "home.forYouSub": "Based on your interests, searches and the books you open.",
  "home.latest": "New on the shelves",
  "common.viewAll": "View all",
  "book.addToCart": "Add to cart",
  "book.buyNow": "Order now",
  "book.payOnline": "Pay online",
  "book.read": "Read a sample",
  "book.reviews": "Reviews",
  "book.writeReview": "Write a review",
  "book.noReviews": "No reviews yet. Be the first to share what you thought.",
  "book.submit": "Post review",
  "cart.title": "Your cart",
  "cart.empty": "Your cart is empty.",
  "orders.title": "Your orders",
  "orders.empty": "You haven't ordered anything yet.",
} as const;

export type MessageKey = keyof typeof en;

const dictionaries: Record<Lang, Record<MessageKey, string>> = {
  en,
  hi: {
    "nav.home": "होम",
    "nav.books": "किताबें",
    "nav.about": "हमारे बारे में",
    "nav.contact": "संपर्क",
    "nav.cart": "कार्ट",
    "nav.orders": "मेरे ऑर्डर",
    "nav.profile": "प्रोफ़ाइल",
    "nav.signin": "साइन इन",
    "nav.signup": "खाता बनाएँ",
    "nav.signout": "साइन आउट",
    "search.placeholder": "शीर्षक, शैली या प्रकाशक खोजें",
    "home.heroTitle": "अपनी अगली पसंदीदा किताब खोजें",
    "home.heroSub": "किताबें देखें, या BookTown को आपकी पसंद सीखकर आपके लिए चुनने दें।",
    "home.browse": "किताबें देखें",
    "home.categories": "शैली के अनुसार",
    "home.forYou": "आपके लिए चुनी गईं",
    "home.forYouSub": "आपकी रुचियों, खोजों और खोली गई किताबों के आधार पर।",
    "home.latest": "नई किताबें",
    "common.viewAll": "सभी देखें",
    "book.addToCart": "कार्ट में डालें",
    "book.buyNow": "अभी ऑर्डर करें",
    "book.payOnline": "ऑनलाइन भुगतान",
    "book.read": "नमूना पढ़ें",
    "book.reviews": "समीक्षाएँ",
    "book.writeReview": "समीक्षा लिखें",
    "book.noReviews": "अभी कोई समीक्षा नहीं है। पहली समीक्षा आप लिखें।",
    "book.submit": "समीक्षा भेजें",
    "cart.title": "आपका कार्ट",
    "cart.empty": "आपका कार्ट खाली है।",
    "orders.title": "आपके ऑर्डर",
    "orders.empty": "आपने अभी तक कुछ ऑर्डर नहीं किया है।",
  },
  fr: {
    "nav.home": "Accueil",
    "nav.books": "Livres",
    "nav.about": "À propos",
    "nav.contact": "Contact",
    "nav.cart": "Panier",
    "nav.orders": "Mes commandes",
    "nav.profile": "Profil",
    "nav.signin": "Connexion",
    "nav.signup": "Créer un compte",
    "nav.signout": "Déconnexion",
    "search.placeholder": "Titres, genres, éditeurs",
    "home.heroTitle": "Trouvez votre prochain livre préféré",
    "home.heroSub": "Parcourez les rayons, ou laissez BookTown apprendre vos goûts et choisir pour vous.",
    "home.browse": "Voir les livres",
    "home.categories": "Par genre",
    "home.forYou": "Choisis pour vous",
    "home.forYouSub": "Selon vos centres d'intérêt, vos recherches et les livres consultés.",
    "home.latest": "Nouveautés",
    "common.viewAll": "Tout voir",
    "book.addToCart": "Ajouter au panier",
    "book.buyNow": "Commander",
    "book.payOnline": "Payer en ligne",
    "book.read": "Lire un extrait",
    "book.reviews": "Avis",
    "book.writeReview": "Donner votre avis",
    "book.noReviews": "Aucun avis pour l'instant. Soyez le premier à partager le vôtre.",
    "book.submit": "Publier",
    "cart.title": "Votre panier",
    "cart.empty": "Votre panier est vide.",
    "orders.title": "Vos commandes",
    "orders.empty": "Vous n'avez encore rien commandé.",
  },
  es: {
    "nav.home": "Inicio",
    "nav.books": "Libros",
    "nav.about": "Nosotros",
    "nav.contact": "Contacto",
    "nav.cart": "Carrito",
    "nav.orders": "Mis pedidos",
    "nav.profile": "Perfil",
    "nav.signin": "Iniciar sesión",
    "nav.signup": "Crear cuenta",
    "nav.signout": "Cerrar sesión",
    "search.placeholder": "Títulos, géneros, editoriales",
    "home.heroTitle": "Encuentra tu próximo libro favorito",
    "home.heroSub": "Recorre las estanterías o deja que BookTown aprenda lo que te gusta y elija por ti.",
    "home.browse": "Ver libros",
    "home.categories": "Por género",
    "home.forYou": "Elegidos para ti",
    "home.forYouSub": "Según tus intereses, búsquedas y los libros que abres.",
    "home.latest": "Novedades",
    "common.viewAll": "Ver todo",
    "book.addToCart": "Añadir al carrito",
    "book.buyNow": "Pedir ahora",
    "book.payOnline": "Pagar en línea",
    "book.read": "Leer un fragmento",
    "book.reviews": "Reseñas",
    "book.writeReview": "Escribir una reseña",
    "book.noReviews": "Aún no hay reseñas. Sé el primero en opinar.",
    "book.submit": "Publicar",
    "cart.title": "Tu carrito",
    "cart.empty": "Tu carrito está vacío.",
    "orders.title": "Tus pedidos",
    "orders.empty": "Todavía no has pedido nada.",
  },
  de: {
    "nav.home": "Start",
    "nav.books": "Bücher",
    "nav.about": "Über uns",
    "nav.contact": "Kontakt",
    "nav.cart": "Warenkorb",
    "nav.orders": "Bestellungen",
    "nav.profile": "Profil",
    "nav.signin": "Anmelden",
    "nav.signup": "Konto erstellen",
    "nav.signout": "Abmelden",
    "search.placeholder": "Titel, Genres, Verlage",
    "home.heroTitle": "Finde dein nächstes Lieblingsbuch",
    "home.heroSub": "Stöbere in den Regalen oder lass BookTown lernen, was dir gefällt.",
    "home.browse": "Bücher ansehen",
    "home.categories": "Nach Genre",
    "home.forYou": "Für dich ausgewählt",
    "home.forYouSub": "Basierend auf deinen Interessen, Suchen und geöffneten Büchern.",
    "home.latest": "Neu im Regal",
    "common.viewAll": "Alle ansehen",
    "book.addToCart": "In den Warenkorb",
    "book.buyNow": "Jetzt bestellen",
    "book.payOnline": "Online bezahlen",
    "book.read": "Leseprobe",
    "book.reviews": "Bewertungen",
    "book.writeReview": "Bewertung schreiben",
    "book.noReviews": "Noch keine Bewertungen. Schreib die erste.",
    "book.submit": "Veröffentlichen",
    "cart.title": "Dein Warenkorb",
    "cart.empty": "Dein Warenkorb ist leer.",
    "orders.title": "Deine Bestellungen",
    "orders.empty": "Du hast noch nichts bestellt.",
  },
};

type I18n = { lang: Lang; setLang: (l: Lang) => void; t: (key: MessageKey) => string };

const I18nContext = createContext<I18n | null>(null);

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem("lang");
    if (saved && saved in LANGUAGES) return saved as Lang;
  } catch {
    // storage blocked
  }
  const browser = navigator.language.slice(0, 2);
  return browser in LANGUAGES ? (browser as Lang) : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = (l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem("lang", l);
    } catch {
      // storage blocked
    }
  };

  const t = (key: MessageKey) => dictionaries[lang][key] ?? en[key];
  return <I18nContext.Provider value={{ lang, setLang, t }}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used inside I18nProvider");
  return ctx;
}
