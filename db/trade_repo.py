"""
CRUD-функции для таблицы trades.

Все функции принимают session: AsyncSession первым параметром
(паттерн как в candle_repo.py) — управление сессией остаётся на вызывающей стороне.
"""
from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Asset, Trade
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


async def get_trade_stats(session: AsyncSession) -> dict:
    """
    Агрегированная статистика по всем сделкам.

    Возвращает:
        Словарь с ключами:
            open_count   — количество открытых позиций
            total_closed — количество закрытых сделок
            total_pnl    — суммарный чистый P&L по закрытым сделкам
            wins         — количество прибыльных сделок (pnl > 0)
            losses       — количество убыточных сделок
            win_rate     — процент прибыльных (0.0–100.0)
            avg_pnl      — средний P&L на закрытую сделку
            best         — {"ticker": str, "pnl": Decimal} лучшей сделки или None
            worst        — {"ticker": str, "pnl": Decimal} худшей сделки или None
            by_reason    — {"SELL_SIGNAL": N, "STOP_LOSS": N, "TAKE_PROFIT": N}
    """
    # ── Агрегаты по закрытым сделкам ────────────────────────────────────────
    agg = await session.execute(
        select(
            func.count().label("total"),
            func.sum(Trade.pnl).label("total_pnl"),
            func.sum(case((Trade.pnl > 0, 1), else_=0)).label("wins"),
        ).where(Trade.status == "CLOSED")
    )
    row = agg.one()

    # ── Количество открытых позиций ─────────────────────────────────────────
    open_scalar = await session.execute(
        select(func.count()).where(Trade.status == "OPEN")
    )

    # ── Лучшая сделка ───────────────────────────────────────────────────────
    best_row = (await session.execute(
        select(Trade, Asset.ticker)
        .join(Asset, Trade.asset_id == Asset.id)
        .where(Trade.status == "CLOSED", Trade.pnl.isnot(None))
        .order_by(Trade.pnl.desc())
        .limit(1)
    )).first()

    # ── Худшая сделка ───────────────────────────────────────────────────────
    worst_row = (await session.execute(
        select(Trade, Asset.ticker)
        .join(Asset, Trade.asset_id == Asset.id)
        .where(Trade.status == "CLOSED", Trade.pnl.isnot(None))
        .order_by(Trade.pnl.asc())
        .limit(1)
    )).first()

    # ── Разбивка по причине закрытия ────────────────────────────────────────
    by_reason_rows = (await session.execute(
        select(Trade.close_reason, func.count().label("cnt"))
        .where(Trade.status == "CLOSED")
        .group_by(Trade.close_reason)
    )).all()

    total = row.total or 0
    wins = int(row.wins or 0)
    total_pnl = Decimal(str(row.total_pnl or 0))

    return {
        "open_count":   open_scalar.scalar() or 0,
        "total_closed": total,
        "total_pnl":    total_pnl,
        "wins":         wins,
        "losses":       total - wins,
        "win_rate":     wins / total * 100 if total > 0 else 0.0,
        "avg_pnl":      total_pnl / total if total > 0 else Decimal("0"),
        "best":  {"ticker": best_row[1],  "pnl": best_row[0].pnl}  if best_row  else None,
        "worst": {"ticker": worst_row[1], "pnl": worst_row[0].pnl} if worst_row else None,
        "by_reason": {r.close_reason: r.cnt for r in by_reason_rows},
    }
