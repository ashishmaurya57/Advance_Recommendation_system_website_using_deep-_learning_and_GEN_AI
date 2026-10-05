from pydantic import BaseModel


def media_url(value: str | None) -> str | None:
    """Image/PDF columns hold public Supabase Storage URLs."""
    return value or None


class Message(BaseModel):
    message: str
