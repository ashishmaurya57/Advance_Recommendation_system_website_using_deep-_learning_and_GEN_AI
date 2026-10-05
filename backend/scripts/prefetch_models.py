"""Download the embedding model and NLTK data at build time, so the server starts fast.

    uv run python -m scripts.prefetch_models

Runs without any settings/secrets (it doesn't import the app config).
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BACKEND_DIR / ".cache" / "fastembed"  # same default as Settings.model_cache_dir
NLTK_DIR = Path(os.environ.get("NLTK_DATA", BACKEND_DIR / ".cache" / "nltk_data"))


def main() -> None:
    import nltk
    from fastembed import TextEmbedding

    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2", cache_dir=str(MODEL_DIR))
    list(model.embed(["warm up"]))
    print(f"embedding model ready in {MODEL_DIR}")

    nltk.download("stopwords", download_dir=str(NLTK_DIR), quiet=True)
    print(f"NLTK stopwords ready in {NLTK_DIR}")


if __name__ == "__main__":
    main()
