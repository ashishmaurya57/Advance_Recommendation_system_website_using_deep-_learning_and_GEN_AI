"""Admin panel at /admin (replaces Django admin).

Sign in with a Django superuser from auth_user, the same account as before.
"""

from typing import Any

from markupsafe import Markup
from sqladmin import Admin, ModelView
from sqladmin.authentication import AuthenticationBackend
from sqlalchemy import select
from starlette.datastructures import UploadFile
from starlette.requests import Request
from wtforms import FileField, FloatField

from app.core.config import settings
from app.core.security import hash_password, is_hashed, verify_password
from app.db.models import (
    AdminUser,
    CartItem,
    Category,
    Contact,
    InterestTag,
    MediaFile,
    Order,
    Product,
    Profile,
    Review,
    SearchLog,
    UserInteraction,
    utcnow,
)
from app.db.session import SessionLocal, engine
from app.schemas.common import media_url
from app.services.files import store


class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        username, password = str(form.get("username", "")), str(form.get("password", ""))
        with SessionLocal() as db:
            admin = db.scalar(select(AdminUser).filter_by(username=username))
            if not (admin and admin.is_superuser and admin.is_active):
                return False
            if not verify_password(password, admin.password):
                return False
            admin.last_login = utcnow()
            db.commit()
        request.session["admin_user"] = username
        return True

    async def logout(self, request: Request) -> bool:
        request.session.pop("admin_user", None)
        return True

    async def authenticate(self, request: Request) -> bool:
        return bool(request.session.get("admin_user"))


def _image(path: str | None, size: int = 48) -> Markup:
    url = media_url(path)
    if not url:
        return Markup("")
    return Markup(
        f'<img src="{url}" style="height:{size}px;width:auto;border-radius:4px;object-fit:cover">'
    )


class UploadMixin:
    """Turns URL columns into file inputs: uploads go to Supabase Storage and are
    recorded in media_files, linked to the row they belong to."""

    upload_fields: dict[str, str] = {}  # column -> media kind
    owner_field: str = ""  # MediaFile column pointing at this model

    async def on_model_change(self, data: dict, model: Any, is_created: bool, request: Request) -> None:
        pending: list[MediaFile] = []
        for field, kind in self.upload_fields.items():
            upload = data.get(field)
            if isinstance(upload, UploadFile) and upload.filename:
                media = store(kind, upload.filename, await upload.read(), upload.content_type)
                data[field] = media.public_url
                pending.append(media)
            elif is_created:
                data[field] = None if field == "pdf" else ""
            else:
                data.pop(field, None)  # no new file chosen: keep the current one
        request.state.pending_media = pending

    async def after_model_change(self, data: dict, model: Any, is_created: bool, request: Request) -> None:
        pending: list[MediaFile] = getattr(request.state, "pending_media", [])
        if not pending:
            return
        owner = model.email if self.owner_field == "profile_id" else model.id
        with SessionLocal() as db:
            for media in pending:
                setattr(media, self.owner_field, owner)
                db.add(media)
            db.commit()


class CategoryAdmin(UploadMixin, ModelView, model=Category):
    name_plural = "Categories"
    icon = "fa-solid fa-layer-group"
    upload_fields = {"cpic": "category_image"}
    owner_field = "category_id"
    column_list = [Category.id, Category.cpic, Category.cname, Category.cdate]
    column_labels = {"cname": "Name", "cpic": "Image", "cdate": "Created"}
    column_formatters = {"cpic": lambda m, a: _image(m.cpic)}
    column_searchable_list = [Category.cname]
    form_columns = [Category.cname, Category.cpic, Category.cdate]
    form_overrides = {"cpic": FileField}


class ProductAdmin(UploadMixin, ModelView, model=Product):
    name = "Book"
    name_plural = "Books"
    icon = "fa-solid fa-book"
    upload_fields = {"ppic": "book_cover", "pdf": "book_pdf"}
    owner_field = "product_id"
    column_list = [
        Product.id, Product.ppic, Product.name, Product.category, Product.language,
        Product.publisher, Product.disprice, Product.tprice, Product.pdf, Product.pdate,
    ]
    column_labels = {
        "ppic": "Cover", "disprice": "Price", "tprice": "MRP", "pdes": "Description",
        "pdate": "Published", "pdf": "PDF",
    }
    column_formatters = {
        "ppic": lambda m, a: _image(m.ppic),
        "pdf": lambda m, a: Markup(f'<a href="{media_url(m.pdf)}" target="_blank">View PDF</a>') if m.pdf else "No PDF",
    }
    column_searchable_list = [Product.name, Product.publisher, Product.language]
    column_sortable_list = [Product.id, Product.name, Product.disprice, Product.pdate]
    form_columns = [
        Product.name, Product.category, Product.ppic, Product.pdf, Product.language,
        Product.hardcover, Product.publisher, Product.tprice, Product.disprice, Product.pdes,
        Product.pdate, Product.tags,
    ]
    # SQLAdmin has no default form field for Float columns.
    form_overrides = {"ppic": FileField, "pdf": FileField, "tprice": FloatField, "disprice": FloatField}
    page_size = 25


class ProfileAdmin(UploadMixin, ModelView, model=Profile):
    name = "Customer"
    name_plural = "Customers"
    icon = "fa-solid fa-user"
    upload_fields = {"ppic": "avatar"}
    owner_field = "profile_id"
    column_list = [Profile.ppic, Profile.name, Profile.email, Profile.mobile, Profile.dob, Profile.interests]
    column_labels = {"ppic": "Photo", "passwd": "Password (type a new one to change it)"}
    column_formatters = {"ppic": lambda m, a: _image(m.ppic, 36)}
    column_searchable_list = [Profile.name, Profile.email]
    column_details_exclude_list = [Profile.passwd]
    form_columns = [
        Profile.name, Profile.email, Profile.passwd, Profile.mobile, Profile.dob,
        Profile.address, Profile.ppic, Profile.interests,
    ]
    form_overrides = {"ppic": FileField}

    async def on_model_change(self, data, model, is_created, request):
        await super().on_model_change(data, model, is_created, request)
        password = data.get("passwd")
        if password and not is_hashed(password):
            data["passwd"] = hash_password(password)


class InterestTagAdmin(ModelView, model=InterestTag):
    name = "Interest tag"
    icon = "fa-solid fa-tags"
    column_list = [InterestTag.id, InterestTag.name]
    column_searchable_list = [InterestTag.name]
    form_columns = [InterestTag.name]


class OrderAdmin(ModelView, model=Order):
    icon = "fa-solid fa-receipt"
    column_list = [Order.id, Order.pid, Order.userid, Order.remarks, Order.status, Order.odate]
    column_labels = {"pid": "Book ID", "userid": "Customer", "remarks": "Status", "status": "Active", "odate": "Date"}
    column_searchable_list = [Order.userid, Order.remarks]
    column_sortable_list = [Order.id, Order.odate]


class CartItemAdmin(ModelView, model=CartItem):
    name = "Cart item"
    icon = "fa-solid fa-cart-shopping"
    column_list = [CartItem.id, CartItem.pid, CartItem.userid, CartItem.cdate]
    column_labels = {"pid": "Book ID", "userid": "Customer", "cdate": "Added"}


class ReviewAdmin(ModelView, model=Review):
    icon = "fa-solid fa-star"
    column_list = [Review.product, Review.user, Review.rating, Review.created_at]
    column_searchable_list = [Review.comment]
    form_columns = [Review.product, Review.user, Review.rating, Review.comment]


class UserInteractionAdmin(ModelView, model=UserInteraction):
    name = "Interaction"
    icon = "fa-solid fa-chart-line"
    column_list = [
        UserInteraction.user, UserInteraction.product, UserInteraction.interaction_type,
        UserInteraction.rating, UserInteraction.count, UserInteraction.timestamp,
    ]
    column_sortable_list = [UserInteraction.timestamp, UserInteraction.interaction_type]
    form_overrides = {"rating": FloatField}


class SearchLogAdmin(ModelView, model=SearchLog):
    name = "Search log"
    icon = "fa-solid fa-magnifying-glass"
    column_list = [SearchLog.user, SearchLog.query, SearchLog.timestamp]
    can_create = False


class MediaFileAdmin(ModelView, model=MediaFile):
    name = "Media file"
    icon = "fa-solid fa-photo-film"
    column_list = [
        MediaFile.id, MediaFile.public_url, MediaFile.kind, MediaFile.path,
        MediaFile.size_bytes, MediaFile.product_id, MediaFile.category_id, MediaFile.profile_id,
        MediaFile.created_at,
    ]
    column_labels = {"public_url": "Preview", "size_bytes": "Size"}
    column_formatters = {
        "public_url": lambda m, a: _image(m.public_url, 40)
        if m.content_type.startswith("image/")
        else Markup(f'<a href="{m.public_url}" target="_blank">Open PDF</a>'),
        "size_bytes": lambda m, a: f"{m.size_bytes / 1024:,.0f} KB",
    }
    column_searchable_list = [MediaFile.path, MediaFile.original_name]
    column_sortable_list = [MediaFile.id, MediaFile.size_bytes, MediaFile.created_at]
    can_create = False
    can_edit = False


class ContactAdmin(ModelView, model=Contact):
    name = "Contact message"
    icon = "fa-solid fa-envelope"
    column_list = [Contact.name, Contact.email, Contact.mobile, Contact.message]
    column_searchable_list = [Contact.name, Contact.email]
    can_create = False


def setup_admin(app) -> Admin:
    admin = Admin(
        app,
        engine,
        title="BookTown Admin",
        authentication_backend=AdminAuth(secret_key=settings.secret_key),
    )
    for view in (
        ProductAdmin, CategoryAdmin, ProfileAdmin, OrderAdmin, CartItemAdmin, ReviewAdmin,
        InterestTagAdmin, UserInteractionAdmin, SearchLogAdmin, MediaFileAdmin, ContactAdmin,
    ):
        admin.add_view(view)
    return admin
