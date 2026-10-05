from datetime import date
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.deps import DB, CurrentUser
from app.core.security import hash_password
from app.schemas.user import UserOut
from app.services.files import save_upload
from app.services.interactions import set_interests
from app.services.recommender import refresh_async

router = APIRouter(prefix="/profile", tags=["profile"])


@router.put("", response_model=UserOut)
async def update_profile(
    user: CurrentUser,
    db: DB,
    name: Annotated[str, Form(min_length=1)],
    mobile: Annotated[str, Form()] = "",
    address: Annotated[str, Form()] = "",
    dob: Annotated[date | None, Form()] = None,
    interests: Annotated[str, Form()] = "",
    new_password: Annotated[str, Form()] = "",
    avatar: Annotated[UploadFile | None, File()] = None,
):
    user.name = name.strip()
    user.mobile = mobile
    user.address = address
    user.dob = dob
    if new_password:
        if len(new_password) < 6:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Password must be at least 6 characters.")
        user.passwd = hash_password(new_password)
    if avatar and avatar.filename:
        user.ppic = await save_upload(db, avatar, "avatar", profile_id=user.email)
    set_interests(db, user, interests.split(","))
    refresh_async(user.email)
    return UserOut.of(user)
