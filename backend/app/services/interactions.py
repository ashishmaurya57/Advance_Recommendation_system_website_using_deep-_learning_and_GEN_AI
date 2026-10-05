"""Tracking what users do (views, cart adds, ratings) and their interest tags."""

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import InterestTag, Product, Profile, UserInteraction, profile_interests, utcnow

MAX_INTERESTS = 5


def log_interaction(db: Session, user: Profile, product: Product, interaction_type: str) -> None:
    interaction = db.scalar(
        select(UserInteraction).filter_by(
            user_id=user.email, product_id=product.id, interaction_type=interaction_type
        )
    )
    if interaction is None:
        db.add(
            UserInteraction(
                user_id=user.email, product_id=product.id, interaction_type=interaction_type, count=1
            )
        )
    elif interaction_type in ("view", "click"):
        interaction.count += 1
        interaction.timestamp = utcnow()
    db.commit()


def log_rating(db: Session, user: Profile, product: Product, rating: int) -> None:
    interaction = db.scalar(
        select(UserInteraction).filter_by(
            user_id=user.email, product_id=product.id, interaction_type="rating"
        )
    )
    if interaction is None:
        db.add(
            UserInteraction(
                user_id=user.email, product_id=product.id, interaction_type="rating", rating=rating
            )
        )
    else:
        # Running average, as in the original app.
        interaction.rating = round(((interaction.rating or rating) + rating) / 2, 2)
        interaction.timestamp = utcnow()
    db.commit()


def remove_cart_interaction(db: Session, user: Profile, product_id: int) -> None:
    db.execute(
        delete(UserInteraction).filter_by(
            user_id=user.email, product_id=product_id, interaction_type="add_to_cart"
        )
    )
    db.commit()


def get_or_create_tag(db: Session, name: str) -> InterestTag:
    tag = db.scalar(select(InterestTag).filter_by(name=name))
    if tag is None:
        tag = InterestTag(name=name)
        db.add(tag)
        db.flush()
    return tag


def add_interest(db: Session, user: Profile, name: str) -> bool:
    """Add an interest tag, dropping the oldest one when the user already has the maximum.

    Returns True if the user's interests changed.
    """
    tag = get_or_create_tag(db, name)
    if tag in user.interests:
        db.commit()
        return False
    if len(user.interests) >= MAX_INTERESTS:
        oldest_id = db.scalar(
            select(profile_interests.c.id)
            .where(profile_interests.c.profile_id == user.email)
            .order_by(profile_interests.c.id)
            .limit(1)
        )
        db.execute(delete(profile_interests).where(profile_interests.c.id == oldest_id))
        db.expire(user, ["interests"])
    user.interests.append(tag)
    db.commit()
    return True


def set_interests(db: Session, user: Profile, names: list[str]) -> None:
    names = [n.strip() for n in names if n.strip()][-MAX_INTERESTS:]
    user.interests = [get_or_create_tag(db, n) for n in dict.fromkeys(names)]
    db.commit()
