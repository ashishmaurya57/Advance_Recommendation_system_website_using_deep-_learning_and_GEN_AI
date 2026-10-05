"""Reading and translating pages of a book's PDF, and compressing uploaded PDFs."""

import io
import logging
from pathlib import Path

from app.core.config import settings
from app.services import storage

log = logging.getLogger(__name__)


def local_copy(pdf_url: str) -> Path | None:
    """Download a book's PDF from our storage once and return the cached local file."""
    if not pdf_url.startswith(settings.storage_url):
        return None  # only fetch files from our own Supabase project
    try:
        return storage.download_cached(pdf_url)
    except Exception as exc:
        log.warning("Couldn't download %s: %s", pdf_url, exc)
        return None


def page_text(pdf_path: Path, page_number: int) -> str | None:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        if not 1 <= page_number <= doc.page_count:
            return None
        return doc.load_page(page_number - 1).get_text()


def translate(text: str, target_language: str) -> str:
    from deep_translator import GoogleTranslator

    try:
        # Google Translate rejects requests over 5000 characters; translate in chunks.
        chunks, current = [], ""
        for line in text.splitlines(keepends=True):
            if len(current) + len(line) > 4500:
                chunks.append(current)
                current = ""
            current += line
        chunks.append(current)
        translator = GoogleTranslator(source="auto", target=target_language)
        return "".join(translator.translate(c) or "" if c.strip() else c for c in chunks)
    except Exception as exc:
        log.warning("Translation to %s failed: %s", target_language, exc)
        return text


def compress_pdf(data: bytes) -> bytes:
    """Recompress a PDF's streams; returns the original bytes if that fails or is larger."""
    import pikepdf

    try:
        with pikepdf.open(io.BytesIO(data)) as pdf:
            out = io.BytesIO()
            pdf.save(out, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
        compressed = out.getvalue()
        return compressed if len(compressed) < len(data) else data
    except Exception as exc:
        log.warning("PDF compression failed: %s", exc)
        return data
