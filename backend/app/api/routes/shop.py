"""Cart, orders and Razorpay payments."""

import hashlib
import hmac
from datetime import date

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import delete, select

from app.core.config import settings
from app.core.deps import DB, CurrentUser
from app.db.models import CartItem, Order, Product
from app.schemas.catalog import ProductCard
from app.schemas.common import Message
from app.schemas.shop import (
    CartAddIn,
    CartItemOut,
    CartOut,
    OrderCreateIn,
    OrderOut,
    PaymentStartIn,
    PaymentStartOut,
    PaymentVerifyIn,
)
from app.services.interactions import log_interaction, remove_cart_interaction
from app.services.recommender import refresh_async

router = APIRouter(tags=["shop"])

PENDING_PAYMENT_KEY = "pending_payment"


def _product(db: DB, product_id: int) -> Product:
    product = db.get(Product, product_id)
    if product is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Book not found.")
    return product


def _place_order(db: DB, email: str, product_id: int, remarks: str) -> Order:
    order = Order(pid=product_id, userid=email, remarks=remarks, status=True, odate=date.today())
    db.add(order)
    db.execute(delete(CartItem).filter_by(pid=product_id, userid=email))
    db.commit()
    return order


# ---- Cart -------------------------------------------------------------------------------


@router.get("/cart", response_model=CartOut)
def get_cart(db: DB, user: CurrentUser):
    items = []
    for item in db.scalars(select(CartItem).filter_by(userid=user.email).order_by(CartItem.id.desc())):
        product = db.get(Product, item.pid)
        if product:  # skip rows whose book was deleted
            items.append(CartItemOut(id=item.id, added=item.cdate, product=ProductCard.of(product)))
    return CartOut(items=items, total=round(sum(i.product.price for i in items), 2))


@router.post("/cart", response_model=Message, status_code=status.HTTP_201_CREATED)
def add_to_cart(body: CartAddIn, db: DB, user: CurrentUser):
    product = _product(db, body.product_id)
    log_interaction(db, user, product, "add_to_cart")
    exists = db.scalar(select(CartItem).filter_by(pid=product.id, userid=user.email))
    if not exists:
        db.add(CartItem(pid=product.id, userid=user.email, status=True, cdate=date.today()))
        db.commit()
    refresh_async(user.email)
    return Message(message="Added to your cart.")


@router.delete("/cart/{item_id}", response_model=Message)
def remove_from_cart(item_id: int, db: DB, user: CurrentUser):
    item = db.scalar(select(CartItem).filter_by(id=item_id, userid=user.email))
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That item isn't in your cart.")
    db.delete(item)
    db.commit()
    remove_cart_interaction(db, user, item.pid)
    refresh_async(user.email)
    return Message(message="Removed from your cart.")


# ---- Orders -----------------------------------------------------------------------------


@router.get("/orders", response_model=list[OrderOut])
def list_orders(db: DB, user: CurrentUser):
    orders = []
    for o in db.scalars(select(Order).filter_by(userid=user.email).order_by(Order.id.desc())):
        product = db.get(Product, o.pid)
        if product:
            orders.append(OrderOut(id=o.id, status=o.remarks, date=o.odate, product=ProductCard.of(product)))
    return orders


@router.post("/orders", response_model=Message, status_code=status.HTTP_201_CREATED)
def create_order(body: OrderCreateIn, db: DB, user: CurrentUser):
    """Order now, pay on delivery (the original "Buy Now" button)."""
    product = _product(db, body.product_id)
    log_interaction(db, user, product, "add_to_cart")
    _place_order(db, user.email, product.id, "pending")
    refresh_async(user.email)
    return Message(message="Your order has been placed.")


@router.delete("/orders/{order_id}", response_model=Message)
def cancel_order(order_id: int, db: DB, user: CurrentUser):
    order = db.scalar(select(Order).filter_by(id=order_id, userid=user.email))
    if order is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Order not found.")
    db.delete(order)
    db.commit()
    return Message(message="Your order has been cancelled.")


# ---- Razorpay ---------------------------------------------------------------------------


def _razorpay_client():
    if not (settings.razorpay_key_id and settings.razorpay_key_secret):
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Online payment isn't configured on this server."
        )
    import razorpay

    return razorpay.Client(auth=(settings.razorpay_key_id, settings.razorpay_key_secret))


@router.post("/payments/start", response_model=PaymentStartOut)
def start_payment(body: PaymentStartIn, request: Request, db: DB, user: CurrentUser):
    product = _product(db, body.product_id)
    client = _razorpay_client()
    amount = int(round(product.disprice * 100))  # paise
    rp_order = client.order.create({"amount": amount, "currency": "INR", "payment_capture": 1})
    request.session[PENDING_PAYMENT_KEY] = {"order_id": rp_order["id"], "product_id": product.id}
    return PaymentStartOut(
        key_id=settings.razorpay_key_id,
        order_id=rp_order["id"],
        amount=amount,
        currency="INR",
        product=ProductCard.of(product),
    )


@router.post("/payments/verify", response_model=Message)
def verify_payment(body: PaymentVerifyIn, request: Request, db: DB, user: CurrentUser):
    pending = request.session.get(PENDING_PAYMENT_KEY)
    if not pending or pending["order_id"] != body.razorpay_order_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No matching payment in progress.")
    expected = hmac.new(
        settings.razorpay_key_secret.encode(),
        f"{body.razorpay_order_id}|{body.razorpay_payment_id}".encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(expected, body.razorpay_signature):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Payment verification failed.")
    _place_order(db, user.email, pending["product_id"], "paid")
    request.session.pop(PENDING_PAYMENT_KEY, None)
    return Message(message="Payment successful. Your order is confirmed.")
