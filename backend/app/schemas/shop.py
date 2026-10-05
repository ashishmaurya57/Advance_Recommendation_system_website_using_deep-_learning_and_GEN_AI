from datetime import date

from pydantic import BaseModel

from app.schemas.catalog import ProductCard


class CartItemOut(BaseModel):
    id: int
    added: date
    product: ProductCard


class CartOut(BaseModel):
    items: list[CartItemOut]
    total: float


class CartAddIn(BaseModel):
    product_id: int


class OrderOut(BaseModel):
    id: int
    status: str
    date: date
    product: ProductCard


class OrderCreateIn(BaseModel):
    product_id: int
    from_cart: bool = False


class PaymentStartIn(BaseModel):
    product_id: int


class PaymentStartOut(BaseModel):
    key_id: str
    order_id: str
    amount: int
    currency: str
    product: ProductCard


class PaymentVerifyIn(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str


class SearchOut(BaseModel):
    query: str
    products: list[ProductCard]
    message: str | None = None


class TranslatedPage(BaseModel):
    page: int
    text: str
