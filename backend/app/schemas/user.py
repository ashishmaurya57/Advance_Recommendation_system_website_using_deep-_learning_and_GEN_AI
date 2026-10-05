from datetime import date

from pydantic import BaseModel, EmailStr

from app.db.models import Profile
from app.schemas.common import media_url


class UserOut(BaseModel):
    email: str
    name: str
    dob: date | None
    mobile: str
    address: str
    avatar: str | None
    interests: list[str]

    @classmethod
    def of(cls, u: Profile) -> "UserOut":
        return cls(
            email=u.email,
            name=u.name,
            dob=u.dob,
            mobile=u.mobile,
            address=u.address,
            avatar=media_url(u.ppic),
            interests=[t.name for t in u.interests],
        )


class SignInIn(BaseModel):
    email: EmailStr
    password: str


class ContactIn(BaseModel):
    name: str
    email: EmailStr
    mobile: str = ""
    message: str
