"""Smart book search: fuzzy title/publisher match, category match, then semantic search.

Searches also teach the recommender: matched categories and strongly positive free-text
queries are added to the user's interest tags.
"""

from dataclasses import dataclass
from functools import lru_cache

from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models import Category, Product, Profile
from app.services import ml
from app.services.interactions import add_interest
from app.services.recommender import refresh_async


@dataclass
class SearchResult:
    query: str
    products: list[Product]
    message: str | None = None


@lru_cache(maxsize=1)
def _stop_words() -> frozenset[str]:
    import nltk
    from nltk.corpus import stopwords

    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")
    return frozenset(words)


def preprocess(text: str) -> str:
    stop = _stop_words()
    return " ".join(w for w in text.lower().split() if w not in stop)


def search(db: Session, user: Profile, raw_query: str) -> SearchResult:
    query_lower = raw_query.strip().lower()
    # "book" is noise in a book store query ("horror book" -> "horror").
    query = " ".join(w for w in query_lower.split() if w != "book")

    products = db.scalars(select(Product).options(selectinload(Product.category))).all()
    if not products:
        return SearchResult(query, [], "No books in the store yet.")
    if not query:
        return SearchResult(query, [], "Type something to search for.")

    # 1. Near-exact title match
    for p in products:
        if fuzz.ratio(p.name.lower(), query_lower) > 80:
            if add_interest(db, user, p.category.cname):
                refresh_async(user.email)
            return SearchResult(query, [p])

    # 2. Publisher match
    for p in products:
        if p.publisher and fuzz.partial_ratio(p.publisher.lower(), query_lower) > 85:
            publisher = p.publisher.lower()
            return SearchResult(query, [x for x in products if publisher in x.publisher.lower()])

    # 3. Category named in the query
    categories = db.scalars(select(Category.cname)).all()
    matched = next((c for c in categories if c.lower() in query_lower), None)
    if matched:
        if add_interest(db, user, matched):
            refresh_async(user.email)
        return SearchResult(
            query, [p for p in products if p.category.cname.lower() == matched.lower()]
        )

    # 4. Semantic search. Skip clearly negative queries ("books I hate"); neutral and
    # positive ones go through. If the sentiment LLM is unavailable, don't block.
    sentiment = ml.sentiment_score(query)
    if sentiment is not None and sentiment < 0.4:
        return SearchResult(query, [], "No books found.")

    cleaned = preprocess(query)
    ml.embed_many([cleaned] + [p.pdes.lower() for p in products] + [p.category.cname.lower() for p in products])
    query_words = set(cleaned.split())
    scored: list[tuple[Product, float]] = []
    for p in products:
        desc, cat = p.pdes.lower(), p.category.cname.lower()
        sim_desc = ml.semantic_similarity(cleaned, desc)
        sim_cat = ml.semantic_similarity(cat, cleaned)
        desc_words = set(preprocess(f"{p.name} {desc} {cat}").split())
        if sim_desc > 0.52 or sim_cat > 0.45 or query_words & desc_words:
            scored.append((p, sim_desc))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = [p for p, _ in scored]
    if not results:
        return SearchResult(query, [], "No relevant results found.")

    # Remember the query as a new interest unless it's close to one the user already has.
    already_similar = any(ml.semantic_similarity(cleaned, t.name) > 0.7 for t in user.interests)
    if not already_similar and sentiment is not None and sentiment > 0.8 and cleaned:
        if add_interest(db, user, cleaned):
            refresh_async(user.email)

    return SearchResult(query, results)
