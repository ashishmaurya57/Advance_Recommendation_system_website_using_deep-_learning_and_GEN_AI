"""enable row level security

Supabase exposes the public schema through its REST API. The backend connects as
the postgres role (which bypasses RLS), so enabling RLS with no policies blocks
anon/authenticated API access to every table without affecting the app.

Revision ID: 01099b7200bb
Revises: 2f63606ee597
Create Date: 2026-10-05 14:53:03.013031

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '01099b7200bb'
down_revision: Union[str, Sequence[str], None] = '2f63606ee597'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TABLES = [
    "auth_user", "media_files", "user_addtocart", "user_category", "user_contact",
    "user_interesttag", "user_order", "user_product", "user_product_disliked_users",
    "user_product_liked_users", "user_product_tags", "user_profile", "user_profile_interests",
    "user_profile_purchased_products", "user_review", "user_searchlog", "user_userinteraction",
    "alembic_version",
]


def upgrade() -> None:
    """Upgrade schema."""
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')


def downgrade() -> None:
    """Downgrade schema."""
    for table in TABLES:
        op.execute(f'ALTER TABLE "{table}" DISABLE ROW LEVEL SECURITY')
