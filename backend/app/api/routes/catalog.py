from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.deps import DB, CurrentUser, OptionalUser
from app.db.models import Category, Product, Review
from app.schemas.catalog import (
    CategoryOut,
    HomeOut,
    ProductCard,
    ProductDetail,
    ProductListOut,
    ReactionIn,
    ReactionOut,
    ReviewIn,
    ReviewOut,
    reaction_of,
)
from app.services.interactions import log_interaction, log_rating
from app.services.recommender import refresh_async

router = APIRouter(tags=["catalog"])


def _get_product(db: DB, product_id: int) -> Product:
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Book not found.")
    return product


@router.get("/home", response_model=HomeOut)
def home(db: DB):
    categories = db.scalars(select(Category).order_by(Category.id.desc())).all()
    latest = db.scalars(
        select(Product).options(selectinload(Product.category)).order_by(Product.id.desc()).limit(12)
    ).all()
    return HomeOut(
        categories=[CategoryOut.of(c) for c in categories],
        latest=[ProductCard.of(p) for p in latest],
    )


@router.get("/categories", response_model=list[CategoryOut])
def categories(db: DB):
    return [CategoryOut.of(c) for c in db.scalars(select(Category).order_by(Category.id.desc()))]


@router.get("/products", response_model=ProductListOut)
def products(db: DB, category: int | None = None):
    query = select(Product).options(selectinload(Product.category)).order_by(Product.id.desc())
    selected = None
    if category is not None:
        selected = db.get(Category, category)
        if selected is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Category not found.")
        query = query.where(Product.category_id == category)
    return ProductListOut(
        category=CategoryOut.of(selected) if selected else None,
        products=[ProductCard.of(p) for p in db.scalars(query)],
    )


@router.get("/products/{product_id}", response_model=ProductDetail)
def product_detail(product_id: int, db: DB, user: OptionalUser):
    product = _get_product(db, product_id)
    if user:
        log_interaction(db, user, product, "view")
        refresh_async(user.email)
    return ProductDetail.of(product, user)


@router.post(
    "/products/{product_id}/reviews", response_model=ReviewOut, status_code=status.HTTP_201_CREATED
)
def add_review(product_id: int, body: ReviewIn, db: DB, user: CurrentUser):
    if not 1 <= body.rating <= 5:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Rating must be between 1 and 5.")
    product = _get_product(db, product_id)
    review = Review(product=product, user=user, rating=body.rating, comment=body.comment.strip())
    db.add(review)
    db.commit()
    log_rating(db, user, product, body.rating)
    refresh_async(user.email)
    return ReviewOut.of(review)


@router.post("/products/{product_id}/reaction", response_model=ReactionOut)
def react(product_id: int, body: ReactionIn, db: DB, user: CurrentUser):
    """Like or dislike a book. Each reader counts once; reacting again toggles it off."""
    product = _get_product(db, product_id)
    if body.action not in ("like", "dislike"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Action must be 'like' or 'dislike'.")
    # Counters are adjusted rather than recomputed so counts from the old app are kept.
    like = body.action == "like"
    same, other = (
        (product.liked_users, product.disliked_users)
        if like
        else (product.disliked_users, product.liked_users)
    )
    delta_same, delta_other = 0, 0
    if user in same:
        same.remove(user)
        delta_same = -1
    else:
        same.append(user)
        delta_same = 1
        if user in other:
            other.remove(user)
            delta_other = -1
    if like:
        product.likes = max(0, product.likes + delta_same)
        product.dislikes = max(0, product.dislikes + delta_other)
    else:
        product.dislikes = max(0, product.dislikes + delta_same)
        product.likes = max(0, product.likes + delta_other)
    db.commit()
    return ReactionOut(
        likes=product.likes, dislikes=product.dislikes, my_reaction=reaction_of(product, user)
    )
