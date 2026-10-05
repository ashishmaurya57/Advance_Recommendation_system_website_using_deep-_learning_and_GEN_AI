from fastapi import APIRouter, status

from app.core.deps import DB
from app.db.models import Contact
from app.schemas.common import Message
from app.schemas.user import ContactIn

router = APIRouter(tags=["contact"])


@router.post("/contact", response_model=Message, status_code=status.HTTP_201_CREATED)
def contact(body: ContactIn, db: DB):
    db.add(Contact(name=body.name, email=body.email, mobile=body.mobile, message=body.message[:600]))
    db.commit()
    return Message(message="Thanks for reaching out. We'll get back to you soon.")
