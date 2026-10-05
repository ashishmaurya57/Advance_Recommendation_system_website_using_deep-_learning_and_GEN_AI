"""Supabase Storage client (REST API, authenticated with the service_role key).

Buckets are public-read so covers and PDFs can be shown directly in the browser;
writes always go through the backend.
"""

import hashlib
import logging
import tempfile
from pathlib import Path

import httpx

from app.core.config import settings

log = logging.getLogger(__name__)

_client: httpx.Client | None = None


class StorageNotConfigured(RuntimeError):
    pass


def _http() -> httpx.Client:
    global _client
    if not settings.supabase_service_role_key:
        raise StorageNotConfigured("Set SUPABASE_SERVICE_ROLE_KEY in backend/.env to enable uploads.")
    if _client is None:
        key = settings.supabase_service_role_key
        _client = httpx.Client(
            base_url=f"{settings.storage_url}/storage/v1",
            headers={"Authorization": f"Bearer {key}", "apikey": key},
            timeout=httpx.Timeout(60, connect=10),
        )
    return _client


def public_url(bucket: str, path: str) -> str:
    return f"{settings.storage_url}/storage/v1/object/public/{bucket}/{path}"


def ensure_buckets() -> None:
    """Make sure the configured buckets exist and are public-read."""
    existing = {b["id"]: b for b in _http().get("/bucket").raise_for_status().json()}
    for bucket in dict.fromkeys((settings.media_bucket, settings.pdf_bucket)):
        if bucket not in existing:
            _http().post("/bucket", json={"id": bucket, "name": bucket, "public": True}).raise_for_status()
            log.info("Created storage bucket %s", bucket)
        elif not existing[bucket]["public"]:
            _http().put(f"/bucket/{bucket}", json={"public": True}).raise_for_status()
            log.info("Made storage bucket %s public", bucket)


def upload(bucket: str, path: str, data: bytes, content_type: str) -> str:
    res = _http().post(
        f"/object/{bucket}/{path}",
        content=data,
        headers={"Content-Type": content_type, "x-upsert": "true", "Cache-Control": "max-age=31536000"},
    )
    if res.status_code >= 400:
        raise RuntimeError(f"Storage upload failed ({res.status_code}): {res.text[:200]}")
    return public_url(bucket, path)


def delete(bucket: str, paths: list[str]) -> None:
    if paths:
        _http().request("DELETE", f"/object/{bucket}", json={"prefixes": paths}).raise_for_status()


_cache_dir = Path(tempfile.gettempdir()) / "booktown-pdf-cache"


def download_cached(url: str) -> Path:
    """Fetch a public file once and keep it on local disk (used to read PDF pages)."""
    _cache_dir.mkdir(exist_ok=True)
    target = _cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:32] + Path(url).suffix)
    if not target.exists():
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as res:
            res.raise_for_status()
            tmp = target.with_suffix(".part")
            with tmp.open("wb") as f:
                for chunk in res.iter_bytes():
                    f.write(chunk)
            tmp.replace(target)
    return target
