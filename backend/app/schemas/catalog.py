from datetime import date, datetime

from pydantic import BaseModel

from app.db.models import Category, Product, Profile, Review
from app.schemas.common import media_url


class CategoryOut(BaseModel):
    id: int
    name: str
    image: str | None

    @classmethod
    def of(cls, c: Category) -> "CategoryOut":
        return cls(id=c.id, name=c.cname, image=media_url(c.cpic))


class ProductCard(BaseModel):
    id: int
    name: str
    image: str | None
    language: str
    category: CategoryOut
    price: float
    mrp: float

    @classmethod
    def of(cls, p: Product) -> "ProductCard":
        return cls(
            id=p.id,
            name=p.name,
            image=media_url(p.ppic),
            language=p.language,
            category=CategoryOut.of(p.category),
            price=p.disprice,
            mrp=p.tprice,
        )


class ReviewOut(BaseModel):
    id: int
    user_name: str
    rating: int
    comment: str
    created_at: datetime

    @classmethod
    def of(cls, r: Review) -> "ReviewOut":
        return cls(
            id=r.id,
            user_name=r.user.name if r.user else "Former reader",
            rating=r.rating,
            comment=r.comment,
            created_at=r.created_at,
        )


class ProductDetail(ProductCard):
    description: str
    publisher: str
    hardcover: str
    published: date
    pdf_url: str | None
    likes: int
    dislikes: int
    average_rating: float | None
    reviews: list[ReviewOut]
    my_reaction: str | None = None

    @classmethod
    def of(cls, p: Product, viewer: Profile | None = None) -> "ProductDetail":
        card = ProductCard.of(p).model_dump()
        ratings = [r.rating for r in p.reviews]
        return cls(
            **card,
            description=p.pdes,
            publisher=p.publisher,
            hardcover=p.hardcover,
            published=p.pdate,
            pdf_url=media_url(p.pdf),
            likes=p.likes,
            dislikes=p.dislikes,
            average_rating=round(sum(ratings) / len(ratings), 1) if ratings else None,
            reviews=[ReviewOut.of(r) for r in p.reviews],
            my_reaction=reaction_of(p, viewer),
        )


def reaction_of(p: Product, viewer: Profile | None) -> str | None:
    if viewer is None:
        return None
    if viewer in p.liked_users:
        return "like"
    if viewer in p.disliked_users:
        return "dislike"
    return None


class HomeOut(BaseModel):
    categories: list[CategoryOut]
    latest: list[ProductCard]


class ProductListOut(BaseModel):
    category: CategoryOut | None
    products: list[ProductCard]


class ReviewIn(BaseModel):
    rating: int
    comment: str = ""


class ReactionIn(BaseModel):
    action: str  # "like" | "dislike"


class ReactionOut(BaseModel):
    likes: int
    dislikes: int
    my_reaction: str | None
