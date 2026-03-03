"""add dividend_gap_days to assets

Revision ID: c5d6e7f8a9b0
Revises: a1b2c3d4e5f6
Create Date: 2026-03-03 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5d6e7f8a9b0'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Добавить колонки для хранения статистики дивидендного гэпа."""
    op.add_column(
        'assets',
        sa.Column('dividend_gap_days', sa.Integer(), nullable=True),
    )
    op.add_column(
        'assets',
        sa.Column(
            'dividend_gap_updated_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Удалить колонки статистики дивидендного гэпа."""
    op.drop_column('assets', 'dividend_gap_updated_at')
    op.drop_column('assets', 'dividend_gap_days')
