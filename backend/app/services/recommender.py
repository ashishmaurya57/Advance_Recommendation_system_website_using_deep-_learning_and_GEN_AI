"""Hybrid recommender: LLM relevance scoring + interaction history + semantic similarity.

Ported from the original Django view. Results are cached per user for 30 minutes and
recomputed in a background thread whenever the user does something that matters
(views a book, adds to cart, reviews, changes interests, searches).
"""

import logging
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.models import Product, Profile, UserInteraction
from app.db.session import SessionLocal
from app.services import ml

log = logging.getLogger(__name__)

CACHE_TTL = 30 * 60
# Groq's free tier allows ~8k tokens/minute, so keep LLM calls few at a time.
LLM_WORKERS = 3
LLM_THRESHOLD = 0.8


class _TTLCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self._data: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            item = self._data.get(key)
            if item is None or item[0] < time.monotonic():
                return None
            return item[1]

    def set(self, key: str, value) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)


_recommendations = _TTLCache(CACHE_TTL)  # email -> [product ids]
_llm_scores = _TTLCache(CACHE_TTL)  # email -> {product id: score}

_running: set[str] = set()
_dirty: set[str] = set()
_state_lock = threading.Lock()


def interaction_score(interactions: list[UserInteraction]) -> float:
    by_type = defaultdict(list)
    for i in interactions:
        by_type[i.interaction_type].append(i)
    ratings = [float(i.rating) for i in by_type["rating"] if i.rating is not None]
    rating_score = sum(ratings) / len(ratings) if ratings else 0
    score = (
        0.2 * min(len(by_type["view"]), 10)
        + 0.1 * min(len(by_type["click"]), 10)
        + 2.0 * min(len(by_type["add_to_cart"]), 2)
        + 0.4 * rating_score
    )
    return round(score, 2)


def _llm_score(product: Product, user_tags: list[str]) -> float | None:
    try:
        result = ml.llm_score_chain().invoke(
            {
                "user_interests": user_tags,
                "product_description": product.pdes.lower(),
                "category_name": product.category.cname.lower(),
            }
        )
    except Exception as exc:  # network / rate limit / bad key
        log.warning("LLM scoring failed for product %s: %s", product.id, exc)
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(result))
    return float(match.group()) if match else None


def compute_recommendations(db: Session, user: Profile) -> list[int]:
    user_tags = [t.name.lower() for t in user.interests]
    products = db.scalars(select(Product).options(selectinload(Product.category))).all()

    # 1. LLM relevance for products that are semantically close to the user's interests.
    llm_cache: dict[int, float] = dict(_llm_scores.get(user.email) or {})
    llm_scores: dict[int, float] = {}
    if user_tags:
        def max_sim(text: str) -> float:
            return max(ml.semantic_similarity(tag, text) for tag in user_tags)

        candidates = [
            p
            for p in products
            if max_sim(p.pdes.lower()) > 0.4 or max_sim(p.category.cname.lower()) > 0.5
        ]
        to_score = [p for p in candidates if p.id not in llm_cache]
        if to_score and settings.groq_api_key:
            with ThreadPoolExecutor(max_workers=LLM_WORKERS) as pool:
                futures = {pool.submit(_llm_score, p, user_tags): p for p in to_score}
                for fut in as_completed(futures):
                    score = fut.result()
                    if score is not None:
                        llm_cache[futures[fut].id] = score
        llm_scores = {
            p.id: llm_cache[p.id]
            for p in candidates
            if llm_cache.get(p.id, 0) >= LLM_THRESHOLD
        }
        _llm_scores.set(user.email, llm_cache)

    # 2. Interaction-based: books similar to ones the user engaged with heavily.
    interactions = db.scalars(select(UserInteraction).filter_by(user_id=user.email)).all()
    per_product = defaultdict(list)
    for i in interactions:
        per_product[i.product_id].append(i)
    scores = {p.id: interaction_score(per_product[p.id]) for p in products}

    interaction_recs: dict[int, float] = {}
    for p in products:
        score = scores[p.id]
        if score <= 1.6:
            continue
        desc, cat = p.pdes.lower(), p.category.cname.lower()
        for other in products:
            if other.id == p.id or other.category.cname.lower() != cat:
                continue
            other_desc = other.pdes.lower()
            sim = ml.semantic_similarity(desc, other_desc)
            if sim >= 0.7:
                combined = 0.4 * score + 0.4 * sim + 0.2 * ml.sentiment_score(other_desc)
                interaction_recs[other.id] = max(interaction_recs.get(other.id, 0), combined)
    for p in products:
        if scores[p.id] > 1.5 and p.id not in interaction_recs:
            interaction_recs[p.id] = scores[p.id]

    # 3. Combine: both signals first, then LLM-only, then interaction-only.
    both = set(llm_scores) & set(interaction_recs)
    llm_only = set(llm_scores) - both
    interaction_only = set(interaction_recs) - both
    return (
        sorted(both, key=lambda pid: -llm_scores[pid])
        + sorted(llm_only, key=lambda pid: -llm_scores[pid])
        + sorted(interaction_only, key=lambda pid: -interaction_recs[pid])
    )


def _refresh(email: str) -> None:
    while True:
        try:
            with SessionLocal() as db:
                user = db.get(Profile, email)
                if user is not None:
                    _recommendations.set(email, compute_recommendations(db, user))
        except Exception:
            log.exception("Recommendation refresh failed for %s", email)
        with _state_lock:
            # Something changed while we were computing; run once more.
            if email in _dirty:
                _dirty.discard(email)
                continue
            _running.discard(email)
            return


def refresh_async(email: str) -> None:
    """Recompute a user's recommendations in the background (coalesces repeated calls)."""
    with _state_lock:
        if email in _running:
            _dirty.add(email)
            return
        _running.add(email)
    threading.Thread(target=_refresh, args=(email,), daemon=True).start()


def get_recommendations(email: str) -> tuple[list[int], bool]:
    """Return (product ids, still computing). Starts a computation if nothing is cached."""
    cached = _recommendations.get(email)
    if cached is None:
        refresh_async(email)
        return [], True
    with _state_lock:
        computing = email in _running
    return cached, computing
