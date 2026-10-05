"""Validating uploads, storing them in Supabase Storage and recording them in media_files."""

import re
import secrets
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import MediaFile
from app.services import storage
from app.services.pdf import compress_pdf

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 100 * 1024 * 1024

# kind -> (bucket, folder inside the bucket)
LOCATIONS = {
    "book_cover": ("media_bucket", "products"),
    "category_image": ("media_bucket", "category"),
    "avatar": ("media_bucket", "profile"),
    "book_pdf": ("pdf_bucket", "pdfs"),
}


def _safe_name(filename: str) -> str:
    stem, suffix = Path(filename or "file").stem, Path(filename or "").suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-")[:60] or "file"
    return f"{stem}-{secrets.token_hex(4)}{suffix}"


def validate(kind: str, data: bytes, content_type: str | None) -> tuple[bytes, str]:
    """Check type and size; returns (bytes to store, content type)."""
    if kind == "book_pdf":
        if not data.startswith(b"%PDF") or len(data) > MAX_PDF_BYTES:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload a PDF under 100 MB.")
        return compress_pdf(data), "application/pdf"
    if content_type not in IMAGE_TYPES or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "Upload a JPG, PNG, WebP or GIF image under 5 MB."
        )
    return data, content_type


def store(kind: str, filename: str, data: bytes, content_type: str | None) -> MediaFile:
    """Upload to Supabase Storage. Returns an unsaved MediaFile; the caller sets the owner and adds it."""
    data, content_type = validate(kind, data, content_type)
    bucket_setting, folder = LOCATIONS[kind]
    bucket = getattr(settings, bucket_setting)
    path = f"{folder}/{_safe_name(filename)}"
    try:
        url = storage.upload(bucket, path, data, content_type)
    except storage.StorageNotConfigured as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    return MediaFile(
        kind=kind,
        bucket=bucket,
        path=path,
        public_url=url,
        content_type=content_type,
        size_bytes=len(data),
        original_name=(filename or "")[:255],
    )


async def save_upload(db: Session, upload: UploadFile, kind: str, **owner) -> str:
    """Store an uploaded file, record it in media_files (owner: product_id / category_id / profile_id)."""
    media = store(kind, upload.filename or "file", await upload.read(), upload.content_type)
    for key, value in owner.items():
        setattr(media, key, value)
    db.add(media)
    return media.public_url
