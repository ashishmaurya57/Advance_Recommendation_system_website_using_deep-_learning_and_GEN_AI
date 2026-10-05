"""ML helpers without torch, so the API fits in a small (512 MB) server.

- Sentence embeddings: fastembed runs all-MiniLM-L6-v2 (the same model as before) with
  ONNX Runtime, so similarity scores and thresholds are unchanged.
- Sentiment and recommendation scoring: Groq-hosted LLMs (free tier).
"""

import logging
import re
import threading
from collections.abc import Iterable
from functools import lru_cache

import numpy as np

from app.core.config import settings

log = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_lock = threading.RLock()
_embedder = None
_embeddings: dict[str, np.ndarray] = {}
_MAX_CACHED_EMBEDDINGS = 20_000


def _model():
    global _embedder
    with _lock:
        if _embedder is None:
            from fastembed import TextEmbedding

            _embedder = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(settings.model_cache_dir))
        return _embedder


def warm_up() -> None:
    """Load the embedding model ahead of the first request (run in a background thread)."""
    try:
        _model()
    except Exception:
        log.exception("Couldn't load the embedding model")


def embed_many(texts: Iterable[str]) -> None:
    """Embed any texts not cached yet, in one batch (much faster than one at a time)."""
    missing = list(dict.fromkeys(t for t in texts if t not in _embeddings))
    if not missing:
        return
    vectors = list(_model().embed(missing, batch_size=64))
    with _lock:
        if len(_embeddings) + len(missing) > _MAX_CACHED_EMBEDDINGS:
            _embeddings.clear()
        for text, vec in zip(missing, vectors, strict=True):
            _embeddings[text] = vec / (np.linalg.norm(vec) or 1.0)


def _embed(text: str) -> np.ndarray:
    vec = _embeddings.get(text)
    if vec is None:
        embed_many([text])
        vec = _embeddings[text]
    return vec


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of two texts' sentence embeddings (-1..1)."""
    return float(np.dot(_embed(a), _embed(b)))


# ---- Groq LLMs --------------------------------------------------------------------------


def _groq(model: str):
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.groq_api_key,
        model=model,
        temperature=0,
        reasoning_effort="low",  # it only has to output a number; keeps token use down
        max_retries=6,  # the client backs off and retries on 429 rate limits
    )


@lru_cache(maxsize=1)
def llm_score_chain():
    """Prompt -> Groq -> text chain that scores a product against a user's interests."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["user_interests", "product_description", "category_name"],
        template="""
You are a recommendation engine.

Input:
- User Interests: {user_interests}
- Product Description: {product_description}
- Category: {category_name}

Instructions:
1. Treat 'user_interests' as a list of keyword phrases.
2. Process each interest **one at a time**, in order:
   a. Check if the interest **is a substring** of the category (e.g., "action" matches "action & adventure"). If it is, count it as a strong match.
   b. Also compare the interest with product_description for context/sentiment alignment.
3. If **category_name contains** the interest word as a substring, add **0.8** to relevance.
4. If **product_description** also strongly aligns with the interest, boost relevance up to **1.0**.
5. If any interest yields a score **greater than or equal to 0.8**, immediately return that score and stop further checks.
6. If none reach that threshold, return the **highest score** found.
7. **Important: Return only the score as a float. No explanation, no extra text. Example: 0.75**
""",
    )
    return prompt | _groq(settings.groq_model) | StrOutputParser()


@lru_cache(maxsize=1)
def _sentiment_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate

    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Rate the sentiment of the text below as the probability that it is positive, "
            "from 0.0 (clearly negative) to 1.0 (clearly positive). A neutral text is about 0.5. "
            "Reply with only the number.\n\nText:\n{text}"
        ),
    )
    return prompt | _groq(settings.groq_fast_model) | StrOutputParser()


def parse_score(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(text))
    if not match:
        return None
    return min(max(float(match.group()), 0.0), 1.0)


_sentiments: dict[str, float] = {}


def sentiment_score(text: str) -> float | None:
    """0..1, where 1 is very positive. None if the LLM is unavailable (not cached)."""
    if text in _sentiments:
        return _sentiments[text]
    if not settings.groq_api_key:
        return None
    try:
        score = parse_score(_sentiment_chain().invoke({"text": text[:2000]}))
    except Exception as exc:
        log.warning("Sentiment scoring failed: %s", exc)
        return None
    if score is not None:
        if len(_sentiments) > 4096:
            _sentiments.clear()
        _sentiments[text] = score
    return score
