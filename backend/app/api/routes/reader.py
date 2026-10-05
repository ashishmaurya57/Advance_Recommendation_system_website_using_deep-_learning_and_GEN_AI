"""Page-by-page translation of a book's PDF."""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from app.core.deps import DB
from app.db.models import Product
from app.schemas.shop import TranslatedPage
from app.services.pdf import local_copy, page_text, translate

router = APIRouter(tags=["reader"])

LANGUAGES = {
    "english": "en",
    "hindi": "hi",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "chinese": "zh-CN",
    "japanese": "ja",
    "russian": "ru",
    "arabic": "ar",
    "portuguese": "pt",
}


@router.get("/reader/languages", response_model=list[str])
def languages():
    return list(LANGUAGES)


@router.get("/products/{product_id}/pages/{page}", response_model=TranslatedPage)
async def translated_page(product_id: int, page: int, db: DB, language: str = Query("english")):
    code = LANGUAGES.get(language.lower())
    if code is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Unsupported language.")
    product = db.get(Product, product_id)
    if product is None or not product.pdf:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This book has no PDF.")
    path = await run_in_threadpool(local_copy, product.pdf)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "PDF file is missing on the server.")
    text = await run_in_threadpool(page_text, path, page)
    if text is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Page not found.")
    if not text.strip():
        return TranslatedPage(page=page, text="")
    return TranslatedPage(page=page, text=await run_in_threadpool(translate, text, code))
