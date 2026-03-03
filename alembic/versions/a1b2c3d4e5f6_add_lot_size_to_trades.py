"""add lot_size to trades

Revision ID: a1b2c3d4e5f6
Revises: b2c3d4e5f6a1
Create Date: 2026-03-03 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Добавить колонку lot_size в таблицу trades."""
    op.add_column(
        'trades',
        sa.Column('lot_size', sa.Integer(), nullable=False, server_default='1'),
    )


def downgrade() -> None:
    """Удалить колонку lot_size из таблицы trades."""
    op.drop_column('trades', 'lot_size')
