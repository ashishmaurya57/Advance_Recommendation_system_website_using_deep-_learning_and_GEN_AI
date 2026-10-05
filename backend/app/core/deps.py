from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.db.models import Profile
from app.db.session import get_db

SESSION_USER_KEY = "userid"

DB = Annotated[Session, Depends(get_db)]


def get_optional_user(request: Request, db: DB) -> Profile | None:
    email = request.session.get(SESSION_USER_KEY)
    if not email:
        return None
    user = db.get(Profile, email)
    if user is None:
        # Account was deleted while the session was still alive.
        request.session.pop(SESSION_USER_KEY, None)
    return user


def get_current_user(user: Annotated[Profile | None, Depends(get_optional_user)]) -> Profile:
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please sign in to continue.")
    return user


CurrentUser = Annotated[Profile, Depends(get_current_user)]
OptionalUser = Annotated[Profile | None, Depends(get_optional_user)]
