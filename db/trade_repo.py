"""
CRUD-функции для таблицы trades.

Все функции принимают session: AsyncSession первым параметром
(паттерн как в candle_repo.py) — управление сессией остаётся на вызывающей стороне.
"""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Trade
from utils.logger import logger


async def save_trade(session: AsyncSession, trade: Trade) -> Trade:
    """
    Сохранить новую сделку в базу данных.

    Аргументы:
        session: активная async-сессия SQLAlchemy
        trade:   объект Trade с заполненными полями

    Возвращает:
        Trade с присвоенным id.
    """
    session.add(trade)
    await session.flush()  # получаем id без коммита
    await session.refresh(trade)
    logger.info(
        "Сделка сохранена",
        trade_id=trade.id,
        asset_id=trade.asset_id,
        status=trade.status,
        entry_price=str(trade.entry_price),
    )
    return trade


async def update_trade(session: AsyncSession, trade: Trade) -> Trade:
    """
    Обновить существующую сделку (статус, exit_price, pnl, closed_at и т.д.).

    Аргументы:
        session: активная async-сессия SQLAlchemy
        trade:   объект Trade с изменёнными полями (должен быть привязан к сессии)

    Возвращает:
        Обновлённый Trade.
    """
    session.add(trade)
    await session.flush()
    logger.info(
        "Сделка обновлена",
        trade_id=trade.id,
        status=trade.status,
        close_reason=trade.close_reason,
        pnl=str(trade.pnl) if trade.pnl is not None else None,
    )
    return trade


async def get_open_trades(session: AsyncSession) -> list[Trade]:
    """
    Получить все открытые позиции (status='OPEN').

    Аргументы:
        session: активная async-сессия SQLAlchemy

    Возвращает:
        Список объектов Trade со статусом OPEN.
    """
    result = await session.execute(
        select(Trade).where(Trade.status == "OPEN").order_by(Trade.opened_at)
    )
    trades = list(result.scalars().all())
    logger.debug("Открытые позиции получены", count=len(trades))
    return trades


async def get_open_trade_by_asset(
    session: AsyncSession, asset_id: int
) -> Trade | None:
    """
    Получить открытую позицию по конкретному активу.

    Аргументы:
        session:  активная async-сессия SQLAlchemy
        asset_id: идентификатор актива

    Возвращает:
        Trade если открытая позиция существует, иначе None.
    """
    # ORDER BY opened_at ASC — правило FIFO: продаём самую раннюю позицию первой
    result = await session.execute(
        select(Trade).where(
            Trade.asset_id == asset_id,
            Trade.status == "OPEN",
        ).order_by(Trade.opened_at.asc())
    )
    return result.scalars().first()


async def get_trade_history(
    session: AsyncSession, limit: int = 50
) -> list[Trade]:
    """
    Получить историю последних закрытых сделок.

    Аргументы:
        session: активная async-сессия SQLAlchemy
        limit:   максимальное количество записей

    Возвращает:
        Список Trade (сначала самые свежие).
    """
    result = await session.execute(
        select(Trade)
        .where(Trade.status == "CLOSED")
        .order_by(Trade.closed_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
