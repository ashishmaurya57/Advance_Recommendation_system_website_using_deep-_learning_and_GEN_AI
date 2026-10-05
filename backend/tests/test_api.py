from sqlalchemy import select

from app.core.security import verify_password
from app.db import models as m
from tests.constants import ADMIN_PASSWORD, ADMIN_USER, LEGACY_EMAIL, LEGACY_PASSWORD, STORAGE

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


def test_home_and_catalog(client):
    home = client.get("/api/home").json()
    assert home["categories"] and home["latest"]
    assert any(b["image"] and b["image"].startswith(STORAGE) for b in home["latest"])

    cat = home["categories"][0]["id"]
    listing = client.get(f"/api/products?category={cat}").json()
    assert listing["category"]["id"] == cat
    assert all(p["category"]["id"] == cat for p in listing["products"])

    pid = home["latest"][0]["id"]
    detail = client.get(f"/api/products/{pid}").json()
    assert detail["id"] == pid and "reviews" in detail
    assert client.get("/api/products/999999").status_code == 404


def test_protected_routes_need_login(client):
    client.post("/api/auth/signout")
    for path in ("/api/cart", "/api/orders", "/api/recommendations", "/api/search?q=x"):
        assert client.get(path).status_code == 401
    assert client.get("/api/auth/me").json() is None


def test_legacy_login_upgrades_password(client, db):
    assert client.post("/api/auth/signin", json={"email": LEGACY_EMAIL, "password": "wrong"}).status_code == 401
    r = client.post("/api/auth/signin", json={"email": LEGACY_EMAIL, "password": LEGACY_PASSWORD})
    assert r.status_code == 200, r.text
    stored = db.get(m.Profile, LEGACY_EMAIL).passwd
    assert stored.startswith("pbkdf2_sha256$") and verify_password(LEGACY_PASSWORD, stored)
    assert client.get("/api/auth/me").json()["email"] == LEGACY_EMAIL
    client.post("/api/auth/signout")


def test_signup_with_avatar_records_media_file(client, db, fake_storage):
    r = client.post(
        "/api/auth/signup",
        data={"name": "Pic Reader", "email": "pic@example.com", "password": "secret123"},
        files={"avatar": ("me.png", PNG, "image/png")},
    )
    assert r.status_code == 201, r.text
    avatar = r.json()["avatar"]
    assert avatar.startswith(f"{STORAGE}/booktown/profile/me-")
    assert len(fake_storage) == 1
    media = db.scalar(select(m.MediaFile).filter_by(public_url=avatar))
    assert media.kind == "avatar" and media.profile_id == "pic@example.com" and media.size_bytes == len(PNG)

    bad = client.put("/api/profile", data={"name": "x"}, files={"avatar": ("a.txt", b"hello", "text/plain")})
    assert bad.status_code == 400
    client.post("/api/auth/signout")


def test_signup_cart_order_review_flow(client):
    r = client.post(
        "/api/auth/signup",
        data={"name": "Test Reader", "email": "Reader@Example.com", "password": "secret123", "interests": "Horror, Romance"},
    )
    assert r.status_code == 201, r.text
    assert r.json()["email"] == "reader@example.com"
    assert sorted(r.json()["interests"]) == ["Horror", "Romance"]
    dup = client.post("/api/auth/signup", data={"name": "x", "email": "reader@example.com", "password": "secret123"})
    assert dup.status_code == 409

    pid = client.get("/api/home").json()["latest"][0]["id"]
    assert client.post("/api/cart", json={"product_id": pid}).status_code == 201
    assert client.post("/api/cart", json={"product_id": pid}).status_code == 201  # no duplicate
    cart = client.get("/api/cart").json()
    assert len(cart["items"]) == 1 and cart["total"] == cart["items"][0]["product"]["price"]

    assert client.post("/api/orders", json={"product_id": pid, "from_cart": True}).status_code == 201
    assert client.get("/api/cart").json()["items"] == []
    orders = client.get("/api/orders").json()
    assert orders[0]["product"]["id"] == pid and orders[0]["status"] == "pending"
    assert client.delete(f"/api/orders/{orders[0]['id']}").status_code == 200
    assert client.get("/api/orders").json() == []

    assert client.post(f"/api/products/{pid}/reviews", json={"rating": 4, "comment": "Great"}).status_code == 201
    assert client.post(f"/api/products/{pid}/reviews", json={"rating": 9}).status_code == 400
    assert client.get(f"/api/products/{pid}").json()["average_rating"] == 4.0

    before = client.get(f"/api/products/{pid}").json()
    liked = client.post(f"/api/products/{pid}/reaction", json={"action": "like"}).json()
    assert liked["likes"] == before["likes"] + 1 and liked["my_reaction"] == "like"
    swapped = client.post(f"/api/products/{pid}/reaction", json={"action": "dislike"}).json()
    assert swapped["likes"] == before["likes"] and swapped["my_reaction"] == "dislike"

    upd = client.put("/api/profile", data={"name": "Renamed", "interests": "a,b,c,d,e,f,g"})
    assert upd.status_code == 200 and upd.json()["interests"] == ["c", "d", "e", "f", "g"]

    assert client.post("/api/payments/start", json={"product_id": pid}).status_code == 503
    client.post("/api/auth/signout")


def test_contact(client):
    r = client.post("/api/contact", json={"name": "A", "email": "a@b.co", "message": "hi"})
    assert r.status_code == 201


def test_admin_login_and_upload(client, db, fake_storage):
    client.get("/admin/logout")
    r = client.get("/admin/", follow_redirects=False)
    assert r.status_code in (302, 303, 307) and "login" in r.headers["location"]
    bad = client.post("/admin/login", data={"username": ADMIN_USER, "password": "nope"}, follow_redirects=False)
    assert bad.status_code == 400
    ok = client.post("/admin/login", data={"username": ADMIN_USER, "password": ADMIN_PASSWORD}, follow_redirects=False)
    assert ok.status_code == 302

    r = client.post(
        "/admin/category/create",
        data={"cname": "Poetry", "cdate": "2026-10-05"},
        files={"cpic": ("poem.png", PNG, "image/png")},
        follow_redirects=False,
    )
    assert r.status_code == 302, r.text
    cat = db.scalar(select(m.Category).filter_by(cname="Poetry"))
    assert cat.cpic.startswith(f"{STORAGE}/booktown/category/poem-")
    media = db.scalar(select(m.MediaFile).filter_by(public_url=cat.cpic))
    assert media.kind == "category_image" and media.category_id == cat.id

    # Editing without choosing a new file keeps the current image.
    r = client.post(
        f"/admin/category/edit/{cat.id}",
        data={"cname": "Poems", "cdate": "2026-10-05"},
        files={"cpic": ("", b"", "application/octet-stream")},
        follow_redirects=False,
    )
    assert r.status_code == 302
    db.expire_all()
    edited = db.get(m.Category, cat.id)
    assert edited.cname == "Poems" and edited.cpic == cat.cpic
    assert client.get("/admin/media-file/list").status_code == 200


def test_every_admin_form_renders(client, db):
    """Each admin list, create and edit page must render (catches unsupported column types)."""
    client.post("/admin/login", data={"username": ADMIN_USER, "password": ADMIN_PASSWORD})
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy import select as sa_select

    from app.main import app

    admin = app.state.admin
    for view in admin.views:
        base = f"/admin/{view.identity}"
        assert client.get(f"{base}/list").status_code == 200, base
        if view.can_create:
            assert client.get(f"{base}/create").status_code == 200, f"{base}/create"
        row = db.scalars(sa_select(view.model).limit(1)).first()
        if row is not None and view.can_edit:
            pk = sa_inspect(row).identity[0]
            assert client.get(f"{base}/edit/{pk}").status_code == 200, f"{base}/edit/{pk}"
