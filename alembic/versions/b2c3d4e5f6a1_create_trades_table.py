"""create trades table

Revision ID: b2c3d4e5f6a1
Revises: 3cbe9cb1ca25
Create Date: 2026-03-03 18:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a1'
down_revision: Union[str, Sequence[str], None] = '3cbe9cb1ca25'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Создать таблицу trades для хранения открытых и закрытых позиций."""
    op.create_table(
        'trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('order_id', sa.String(length=100), nullable=True),
        sa.Column('lots', sa.Integer(), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column('exit_price', sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column('stop_loss_price', sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column('take_profit_price', sa.Numeric(precision=18, scale=6), nullable=False),
        sa.Column('status', sa.String(length=10), nullable=False),
        sa.Column('close_reason', sa.String(length=20), nullable=True),
        sa.Column('pnl', sa.Numeric(precision=18, scale=6), nullable=True),
        sa.Column('opened_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.Column('closed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['assets.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_trades_asset_id'), 'trades', ['asset_id'], unique=False)
    op.create_index(op.f('ix_trades_status'), 'trades', ['status'], unique=False)
    op.create_index('ix_trades_asset_status', 'trades', ['asset_id', 'status'], unique=False)


def downgrade() -> None:
    """Удалить таблицу trades."""
    op.drop_index('ix_trades_asset_status', table_name='trades')
    op.drop_index(op.f('ix_trades_status'), table_name='trades')
    op.drop_index(op.f('ix_trades_asset_id'), table_name='trades')
    op.drop_table('trades')
