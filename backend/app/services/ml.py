"""Lazily loaded ML models (sentiment, sentence embeddings, Groq LLM).

Models are loaded on first use, not at import time, so the API starts quickly.
"""

import threading
from functools import lru_cache

from app.core.config import settings

# Importing transformers from two threads at once can fail with a half-initialised
# module, so imports and model loading are serialised.
_lock = threading.RLock()
_models: dict[str, object] = {}


def _sentiment_pipeline():
    with _lock:
        if "sentiment" not in _models:
            from transformers import pipeline

            _models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                revision="714eb0f",
            )
        return _models["sentiment"]


def _semantic_model():
    with _lock:
        if "semantic" not in _models:
            from sentence_transformers import SentenceTransformer

            _models["semantic"] = SentenceTransformer("all-MiniLM-L6-v2")
        return _models["semantic"]


def warm_up() -> None:
    """Load both models ahead of the first request (run in a background thread)."""
    _semantic_model()
    _sentiment_pipeline()


@lru_cache(maxsize=1)
def llm_score_chain():
    """Prompt -> Groq -> text chain that scores a product against a user's interests."""
    with _lock:
        return _build_llm_chain()


def _build_llm_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_groq import ChatGroq

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
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.5,
        reasoning_effort="low",  # it only has to output a number; keeps token use down
        max_retries=6,  # the client backs off and retries on 429 rate limits
    )
    return prompt | llm | StrOutputParser()


@lru_cache(maxsize=4096)
def sentiment_score(text: str) -> float:
    """0..1, where 1 is very positive and 0 is very negative."""
    result = _sentiment_pipeline()(text[:512])[0]
    return result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]


@lru_cache(maxsize=4096)
def _embed(text: str):
    return _semantic_model().encode(text, convert_to_tensor=True)


def semantic_similarity(a: str, b: str) -> float:
    """Cosine similarity of two texts' sentence embeddings."""
    with _lock:
        from sentence_transformers import util
    return util.pytorch_cos_sim(_embed(a), _embed(b)).item()
