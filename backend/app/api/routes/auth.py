from datetime import date
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from app.core.deps import DB, SESSION_USER_KEY, OptionalUser
from app.core.security import hash_password, is_hashed, verify_password
from app.db.models import Profile
from app.schemas.common import Message
from app.schemas.user import SignInIn, UserOut
from app.services.files import save_upload
from app.services.interactions import set_interests
from app.services.recommender import refresh_async

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def signup(
    request: Request,
    db: DB,
    name: Annotated[str, Form(min_length=1)],
    email: Annotated[str, Form(min_length=3)],
    password: Annotated[str, Form(min_length=6)],
    mobile: Annotated[str, Form()] = "",
    address: Annotated[str, Form()] = "",
    dob: Annotated[date | None, Form()] = None,
    interests: Annotated[str, Form()] = "",
    avatar: Annotated[UploadFile | None, File()] = None,
):
    email = email.strip().lower()
    if db.get(Profile, email):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists.")
    user = Profile(
        email=email,
        name=name.strip(),
        passwd=hash_password(password),
        mobile=mobile,
        address=address,
        dob=dob,
        ppic="",
    )
    db.add(user)
    db.flush()
    if avatar and avatar.filename:
        user.ppic = await save_upload(db, avatar, "avatar", profile_id=user.email)
    set_interests(db, user, interests.split(","))
    request.session[SESSION_USER_KEY] = user.email
    refresh_async(user.email)
    return UserOut.of(user)


@router.post("/signin", response_model=UserOut)
def signin(body: SignInIn, request: Request, db: DB):
    # Old accounts may have mixed-case emails; try exact match first.
    user = db.get(Profile, body.email) or db.get(Profile, body.email.lower())
    if user is None or not verify_password(body.password, user.passwd):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email or password is incorrect.")
    if not is_hashed(user.passwd):
        # Upgrade passwords the old app stored in plain text.
        user.passwd = hash_password(body.password)
        db.commit()
    request.session[SESSION_USER_KEY] = user.email
    refresh_async(user.email)
    return UserOut.of(user)


@router.post("/signout", response_model=Message)
def signout(request: Request):
    request.session.clear()
    return Message(message="Signed out.")


@router.get("/me", response_model=UserOut | None)
def me(user: OptionalUser):
    return UserOut.of(user) if user else None
