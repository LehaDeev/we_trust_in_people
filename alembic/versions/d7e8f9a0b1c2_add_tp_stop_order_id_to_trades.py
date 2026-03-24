"""add_tp_stop_order_id_to_trades

Revision ID: d7e8f9a0b1c2
Revises: 855e365d6518
Create Date: 2026-03-24 12:00:00.000000

Добавляет колонку tp_stop_order_id в таблицу trades.

Переход с лимитного TP-ордера на стоп-ордер типа STOP_ORDER_TYPE_TAKE_PROFIT:
- старые позиции используют tp_order_id (лимитный ордер, блокирует акции)
- новые позиции используют tp_stop_order_id (стоп-ордер, не блокирует акции)
Оба поля живут одновременно для backward compatibility.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7e8f9a0b1c2'
down_revision: Union[str, Sequence[str], None] = '855e365d6518'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Добавить колонку tp_stop_order_id в таблицу trades."""
    op.add_column('trades', sa.Column('tp_stop_order_id', sa.String(length=100), nullable=True))


def downgrade() -> None:
    """Удалить колонку tp_stop_order_id из таблицы trades."""
    op.drop_column('trades', 'tp_stop_order_id')
