"""Search and personalised recommendations."""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.deps import DB, CurrentUser
from app.db.models import Product
from app.schemas.catalog import ProductCard
from app.schemas.shop import SearchOut
from app.services import recommender
from app.services.search import search as run_search

router = APIRouter(tags=["discover"])


class RecommendationsOut(BaseModel):
    products: list[ProductCard]
    computing: bool


@router.get("/search", response_model=SearchOut)
async def search(q: str, db: DB, user: CurrentUser):
    # The ML models are CPU-bound; keep them off the event loop.
    result = await run_in_threadpool(run_search, db, user, q)
    return SearchOut(
        query=result.query,
        products=[ProductCard.of(p) for p in result.products],
        message=result.message,
    )


@router.get("/recommendations", response_model=RecommendationsOut)
def recommendations(db: DB, user: CurrentUser):
    ids, computing = recommender.get_recommendations(user.email)
    by_id = {
        p.id: p
        for p in db.scalars(
            select(Product).options(selectinload(Product.category)).where(Product.id.in_(ids))
        )
    }
    return RecommendationsOut(
        products=[ProductCard.of(by_id[i]) for i in ids if i in by_id], computing=computing
    )
