"""One-time copy of the old Django SQLite database (and its uploaded files) into Supabase.

    uv run python -m scripts.migrate_from_sqlite                # rows + files
    uv run python -m scripts.migrate_from_sqlite --replace      # wipe Supabase tables first

Rows keep their ids. Images and PDFs referenced by the old rows are uploaded from
legacy_data/static/ to Supabase Storage, recorded in media_files, and the rows are rewritten to
point at the public URLs. Requires DATABASE_URL and SUPABASE_SERVICE_ROLE_KEY.
"""

import argparse
import mimetypes
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import Boolean, Date, DateTime, text
from sqlalchemy.orm import Session

from app.core.config import BACKEND_DIR
from app.db import models  # noqa: F401
from app.db.models import MediaFile
from app.db.session import Base, engine
from app.services import storage
from app.services.files import store

# Parents before children so foreign keys are satisfied.
TABLES = [
    "auth_user",
    "user_category",
    "user_interesttag",
    "user_profile",
    "user_product",
    "user_contact",
    "user_order",
    "user_addtocart",
    "user_review",
    "user_searchlog",
    "user_userinteraction",
    "user_profile_interests",
    "user_product_tags",
    "user_profile_purchased_products",
    "user_product_liked_users",
    "user_product_disliked_users",
]

# table -> (file column, media kind, MediaFile owner column, row key column)
MEDIA_COLUMNS = {
    "user_category": [("cpic", "category_image", "category_id", "id")],
    "user_product": [("ppic", "book_cover", "product_id", "id"), ("pdf", "book_pdf", "product_id", "id")],
    "user_profile": [("ppic", "avatar", "profile_id", "email")],
}


def convert(value, column):
    if value is None:
        return None
    if isinstance(column.type, Boolean):
        return bool(value)
    if isinstance(column.type, DateTime) and isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    if isinstance(column.type, Date) and isinstance(value, str):
        return date.fromisoformat(value[:10])
    return value


def upload_file(static_dir: Path, old_path: str, kind: str) -> MediaFile | None:
    rel = Path(old_path.replace("\\", "/"))
    if rel.parts and rel.parts[0] == "static":
        rel = Path(*rel.parts[1:])
    local = static_dir / rel
    if not local.is_file():
        print(f"  ! missing file, clearing: {old_path}")
        return None
    content_type = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
    return store(kind, local.name, local.read_bytes(), content_type)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sqlite", type=Path, default=BACKEND_DIR / "legacy_data" / "db.sqlite3")
    parser.add_argument("--static", type=Path, default=BACKEND_DIR / "legacy_data" / "static")
    parser.add_argument("--replace", action="store_true", help="empty the Supabase tables before copying")
    args = parser.parse_args()

    src = sqlite3.connect(args.sqlite)
    src.row_factory = sqlite3.Row
    existing_src = {r[0] for r in src.execute("select name from sqlite_master where type='table'")}

    storage.ensure_buckets()

    with Session(engine) as db:
        filled = [t for t in TABLES if db.execute(text(f'select exists(select 1 from "{t}")')).scalar()]
        if filled and not args.replace:
            print(f"Supabase already has data in {filled}. Re-run with --replace to overwrite.")
            return 1
        if args.replace:
            names = ", ".join(f'"{t}"' for t in TABLES + ["media_files"])
            db.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
            db.commit()

        pending_media: list[MediaFile] = []
        for table_name in TABLES:
            if table_name not in existing_src:
                continue
            table = Base.metadata.tables[table_name]
            rows = [dict(r) for r in src.execute(f'select * from "{table_name}"')]
            out = []
            for row in rows:
                record = {c.name: convert(row.get(c.name), c) for c in table.columns if c.name in row}
                for column, kind, owner_field, key in MEDIA_COLUMNS.get(table_name, []):
                    old = record.get(column)
                    if not old:
                        record[column] = None if column == "pdf" else ""
                        continue
                    media = upload_file(args.static, old, kind)
                    record[column] = media.public_url if media else ("" if column != "pdf" else None)
                    if media:
                        setattr(media, owner_field, record[key])
                        pending_media.append(media)
                out.append(record)
            if out:
                db.execute(table.insert(), out)
            print(f"{table_name}: {len(out)} rows")

        db.add_all(pending_media)
        db.flush()
        print(f"media_files: {len(pending_media)} files uploaded")

        # Continue id sequences after the copied ids.
        for table_name in TABLES + ["media_files"]:
            table = Base.metadata.tables[table_name]
            if "id" in table.columns and table.c.id.autoincrement:
                db.execute(
                    text(
                        f"select setval(pg_get_serial_sequence('\"{table_name}\"', 'id'), "
                        f'coalesce((select max(id) from "{table_name}"), 0) + 1, false)'
                    )
                )
        db.commit()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
