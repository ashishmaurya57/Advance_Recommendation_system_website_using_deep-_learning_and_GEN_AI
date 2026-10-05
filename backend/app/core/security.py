"""Password hashing in Django's pbkdf2_sha256 format.

Using the same format means the Django superuser in auth_user can log in to the admin
panel unchanged. Customer passwords in user_profile were stored as plain text by the old
app; they are accepted once and re-hashed on the next successful login.
"""

import base64
import hashlib
import hmac
import secrets

ALGORITHM = "pbkdf2_sha256"
ITERATIONS = 870_000


def hash_password(password: str, *, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_urlsafe(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), iterations)
    return f"{ALGORITHM}${iterations}${salt}${base64.b64encode(digest).decode()}"


def is_hashed(stored: str) -> bool:
    return stored.startswith(f"{ALGORITHM}$")


def verify_password(password: str, stored: str) -> bool:
    if not is_hashed(stored):
        return hmac.compare_digest(password.encode(), stored.encode())
    try:
        _, iterations, salt, expected = stored.split("$", 3)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    except ValueError:
        return False
    return hmac.compare_digest(base64.b64encode(digest).decode(), expected)
