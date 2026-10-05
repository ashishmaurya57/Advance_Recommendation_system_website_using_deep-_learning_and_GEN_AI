"""Tests run against a throwaway SQLite database built from the models, with
Supabase Storage replaced by an in-memory fake. Supabase itself is never touched."""

import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

_db = Path(tempfile.mkdtemp()) / "test.sqlite3"
os.environ["DATABASE_URL"] = f"sqlite:///{_db.as_posix()}"
os.environ["SUPABASE_URL"] = "https://test-project.supabase.co"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test-key"
os.environ["RAZORPAY_KEY_ID"] = ""
os.environ["RAZORPAY_KEY_SECRET"] = ""
os.environ["GROQ_API_KEY"] = ""
os.environ["MEDIA_BUCKET"] = "booktown"
os.environ["PDF_BUCKET"] = "booktown"

from app.core.security import hash_password  # noqa: E402
from app.db import models as m  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402

from tests.constants import ADMIN_PASSWORD, ADMIN_USER, LEGACY_EMAIL, LEGACY_PASSWORD, STORAGE  # noqa: E402


def _seed():
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        horror = m.Category(cname="Horror", cpic=f"{STORAGE}/booktown/category/h.jpg", cdate=date(2024, 1, 1))
        science = m.Category(cname="Computer Science", cpic="", cdate=date(2024, 1, 1))
        db.add_all([horror, science])
        db.flush()
        for i in range(3):
            db.add(
                m.Product(
                    name=f"Haunted House {i}", ppic=f"{STORAGE}/booktown/products/{i}.jpg", language="english",
                    hardcover="Paperback", publisher="Night Press", tprice=500, disprice=399,
                    pdes="A chilling ghost story.", pdate=date(2024, 2, 1), category_id=horror.id, likes=3,
                )
            )
        db.add(
            m.Product(
                name="Algorithms Illustrated", ppic="", language="english", hardcover="Hardcover",
                publisher="Byte Books", tprice=900, disprice=750, pdes="Sorting, graphs and more.",
                pdate=date(2024, 3, 1), category_id=science.id,
            )
        )
        # An account created by the old Django app: plain-text password.
        db.add(m.Profile(email=LEGACY_EMAIL, name="Old Reader", passwd=LEGACY_PASSWORD, mobile="", address=""))
        db.add(
            m.AdminUser(
                username=ADMIN_USER, password=hash_password(ADMIN_PASSWORD),
                is_superuser=True, is_staff=True, is_active=True,
            )
        )
        db.commit()


_seed()


@pytest.fixture(autouse=True)
def fake_storage(monkeypatch):
    """Record uploads instead of sending them to Supabase."""
    from app.services import storage

    uploaded: dict[str, bytes] = {}

    def upload(bucket, path, data, content_type):
        uploaded[f"{bucket}/{path}"] = data
        return storage.public_url(bucket, path)

    monkeypatch.setattr(storage, "upload", upload)
    return uploaded


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def db():
    with SessionLocal() as session:
        yield session
