"""ORM models (Supabase Postgres).

Table and column names match the original Django app so data copies across 1:1
(see scripts/migrate_from_sqlite.py). Schema changes go through Alembic.
Image/PDF columns hold public Supabase Storage URLs; every stored file is also
recorded in media_files.
"""

from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    # Django stored naive UTC datetimes; keep the same convention.
    return datetime.now(timezone.utc).replace(tzinfo=None)


profile_interests = Table(
    "user_profile_interests",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("profile_id", String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), nullable=False),
    Column("interesttag_id", Integer, ForeignKey("user_interesttag.id", ondelete="CASCADE"), nullable=False),
)

product_tags = Table(
    "user_product_tags",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", Integer, ForeignKey("user_product.id", ondelete="CASCADE"), nullable=False),
    Column("interesttag_id", Integer, ForeignKey("user_interesttag.id", ondelete="CASCADE"), nullable=False),
)

profile_purchased_products = Table(
    "user_profile_purchased_products",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("profile_id", String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), nullable=False),
    Column("product_id", Integer, ForeignKey("user_product.id", ondelete="CASCADE"), nullable=False),
)

product_liked_users = Table(
    "user_product_liked_users",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", Integer, ForeignKey("user_product.id", ondelete="CASCADE"), nullable=False),
    Column("profile_id", String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), nullable=False),
)

product_disliked_users = Table(
    "user_product_disliked_users",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("product_id", Integer, ForeignKey("user_product.id", ondelete="CASCADE"), nullable=False),
    Column("profile_id", String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), nullable=False),
)


class Category(Base):
    __tablename__ = "user_category"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cname: Mapped[str] = mapped_column(String(40))
    cpic: Mapped[str] = mapped_column(String(500), default="")
    cdate: Mapped[date] = mapped_column(Date, default=date.today)

    products: Mapped[list["Product"]] = relationship(
        back_populates="category", cascade="all, delete-orphan"
    )

    def __str__(self) -> str:
        return self.cname


class InterestTag(Base):
    __tablename__ = "user_interesttag"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)

    def __str__(self) -> str:
        return self.name


class Profile(Base):
    __tablename__ = "user_profile"

    email: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    mobile: Mapped[str] = mapped_column(String(20), default="")
    passwd: Mapped[str] = mapped_column(String(100))
    ppic: Mapped[str] = mapped_column(String(500), default="")
    address: Mapped[str] = mapped_column(Text, default="")

    interests: Mapped[list[InterestTag]] = relationship(
        secondary=profile_interests, order_by=profile_interests.c.id
    )
    purchased_products: Mapped[list["Product"]] = relationship(
        secondary=profile_purchased_products
    )
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    interactions: Mapped[list["UserInteraction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    search_logs: Mapped[list["SearchLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    def __str__(self) -> str:
        return f"{self.name} <{self.email}>"


class Product(Base):
    __tablename__ = "user_product"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(150))
    ppic: Mapped[str] = mapped_column(String(500), default="")
    language: Mapped[str] = mapped_column(String(40))
    hardcover: Mapped[str] = mapped_column(String(50))
    publisher: Mapped[str] = mapped_column(String(100))
    tprice: Mapped[float] = mapped_column(Float)
    disprice: Mapped[float] = mapped_column(Float)
    pdes: Mapped[str] = mapped_column(Text)
    pdate: Mapped[date] = mapped_column(Date, default=date.today)
    pdf: Mapped[str | None] = mapped_column(String(500), nullable=True)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    dislikes: Mapped[int] = mapped_column(Integer, default=0)
    category_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_category.id", ondelete="CASCADE"), index=True)

    category: Mapped[Category] = relationship(back_populates="products")
    tags: Mapped[list[InterestTag]] = relationship(secondary=product_tags)
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
        order_by="Review.created_at.desc()",
    )
    interactions: Mapped[list["UserInteraction"]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    liked_users: Mapped[list[Profile]] = relationship(secondary=product_liked_users)
    disliked_users: Mapped[list[Profile]] = relationship(secondary=product_disliked_users)

    def __str__(self) -> str:
        return self.name


class Order(Base):
    __tablename__ = "user_order"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pid: Mapped[int] = mapped_column(Integer)
    userid: Mapped[str] = mapped_column(String(100))
    remarks: Mapped[str] = mapped_column(String(40))
    status: Mapped[bool] = mapped_column(Boolean, default=True)
    odate: Mapped[date] = mapped_column(Date, default=date.today)


class CartItem(Base):
    __tablename__ = "user_addtocart"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    pid: Mapped[int] = mapped_column(Integer)
    userid: Mapped[str] = mapped_column(String(100))
    status: Mapped[bool] = mapped_column(Boolean, default=True)
    cdate: Mapped[date] = mapped_column(Date, default=date.today)


class Review(Base):
    __tablename__ = "user_review"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rating: Mapped[int] = mapped_column(Integer, default=1)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    product_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_product.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), index=True)

    product: Mapped[Product] = relationship(back_populates="reviews")
    user: Mapped[Profile] = relationship(back_populates="reviews")


class UserInteraction(Base):
    __tablename__ = "user_userinteraction"
    __table_args__ = (UniqueConstraint("user_id", "product_id", "interaction_type"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    interaction_type: Mapped[str] = mapped_column(String(50))
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    count: Mapped[int] = mapped_column(Integer, default=1)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    product_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_product.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), index=True)

    product: Mapped[Product] = relationship(back_populates="interactions")
    user: Mapped[Profile] = relationship(back_populates="interactions")


class SearchLog(Base):
    __tablename__ = "user_searchlog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    user_id: Mapped[str] = mapped_column(String(80), ForeignKey("user_profile.email", ondelete="CASCADE"), index=True)

    user: Mapped[Profile] = relationship(back_populates="search_logs")


class Contact(Base):
    __tablename__ = "user_contact"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(120))
    mobile: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(String(600))


class MediaFile(Base):
    """Every image or PDF stored in Supabase Storage, and what it belongs to."""

    __tablename__ = "media_files"
    __table_args__ = (UniqueConstraint("bucket", "path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)  # see MEDIA_KINDS
    bucket: Mapped[str] = mapped_column(String(63))
    path: Mapped[str] = mapped_column(String(400))
    public_url: Mapped[str] = mapped_column(String(500), unique=True)
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    original_name: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    product_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user_product.id", ondelete="SET NULL"), nullable=True, index=True
    )
    category_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user_category.id", ondelete="SET NULL"), nullable=True, index=True
    )
    profile_id: Mapped[str | None] = mapped_column(
        String(80), ForeignKey("user_profile.email", ondelete="SET NULL", onupdate="CASCADE"), nullable=True, index=True
    )

    def __str__(self) -> str:
        return f"{self.kind}: {self.path}"


MEDIA_KINDS = {
    "book_cover": "media",
    "book_pdf": "pdfs",
    "category_image": "media",
    "avatar": "media",
}


class AdminUser(Base):
    """Django's auth_user table; only superusers can sign in to /admin."""

    __tablename__ = "auth_user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(150), unique=True)
    password: Mapped[str] = mapped_column(String(128))
    email: Mapped[str] = mapped_column(String(254), default="")
    first_name: Mapped[str] = mapped_column(String(150), default="")
    last_name: Mapped[str] = mapped_column(String(150), default="")
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    date_joined: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
