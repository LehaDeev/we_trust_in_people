"""
Репозиторий для работы с активами и свечами в БД.
Все операции через async SQLAlchemy session.
"""
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Asset, Candle
from utils.logger import logger


# Маппинг CandleInterval.name -> строка для хранения в БД
INTERVAL_TO_STR: dict[str, str] = {
    "CANDLE_INTERVAL_1_MIN": "1min",
    "CANDLE_INTERVAL_5_MIN": "5min",
    "CANDLE_INTERVAL_15_MIN": "15min",
    "CANDLE_INTERVAL_HOUR": "1h",
    "CANDLE_INTERVAL_DAY": "1d",
    "CANDLE_INTERVAL_WEEK": "1w",
}


async def get_or_create_asset(
    session: AsyncSession,
    figi: str,
    ticker: str,
    name: str,
    currency: str = "RUB",
) -> Asset:
    """
    Получить актив из БД или создать если не существует.

    Args:
        session: async DB session
        figi: FIGI инструмента
        ticker: тикер (SBER, GAZP)
        name: название компании
        currency: валюта (RUB, USD)

    Returns:
        Asset ORM объект
    """
    result = await session.execute(select(Asset).where(Asset.figi == figi))
    asset = result.scalar_one_or_none()

    if asset is None:
        asset = Asset(figi=figi, ticker=ticker, name=name, currency=currency)
        session.add(asset)
        await session.flush()  # получить id без commit
        logger.info("Asset created", ticker=ticker, figi=figi)
    else:
        logger.debug("Asset already exists", ticker=ticker, figi=figi)

    return asset


async def get_last_candle_time(
    session: AsyncSession,
    asset_id: int,
    interval: str,
) -> datetime | None:
    """
    Получить время последней сохранённой свечи для инструмента.
    Используется для инкрементального обновления.

    Args:
        session: async DB session
        asset_id: ID актива в БД
        interval: строка интервала ("1h", "1d")

    Returns:
        datetime последней свечи или None если свечей нет
    """
    from sqlalchemy import func
    result = await session.execute(
        select(func.max(Candle.time)).where(
            Candle.asset_id == asset_id,
            Candle.interval == interval,
        )
    )
    return result.scalar_one_or_none()


async def save_candles(
    session: AsyncSession,
    asset_id: int,
    candles: list[dict],
    interval: str,
    chunk_size: int = 4000,
) -> int:
    """
    Сохранить список свечей в БД. Дубликаты пропускаются (ON CONFLICT DO NOTHING).
    Вставка батчами по chunk_size строк (asyncpg limit: 32767 параметров).

    Args:
        session: async DB session
        asset_id: ID актива в БД
        candles: список словарей {time, open, high, low, close, volume}
        interval: строка интервала ("1h", "1d")
        chunk_size: максимум строк в одном INSERT (default 4000)

    Returns:
        Количество новых сохранённых свечей
    """
    if not candles:
        return 0

    rows = [
        {
            "asset_id": asset_id,
            "time": c["time"],
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
            "interval": interval,
        }
        for c in candles
    ]

    total_saved = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        stmt = insert(Candle).values(chunk)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["asset_id", "time", "interval"]
        )
        result = await session.execute(stmt)
        total_saved += result.rowcount

    logger.info(
        "Candles saved",
        asset_id=asset_id,
        interval=interval,
        total=len(candles),
        new=total_saved,
    )
    return total_saved


async def get_candles_count(
    session: AsyncSession,
    asset_id: int,
    interval: str,
) -> int:
    """Получить количество свечей для актива."""
    from sqlalchemy import func
    result = await session.execute(
        select(func.count()).where(
            Candle.asset_id == asset_id,
            Candle.interval == interval,
        )
    )
    return result.scalar_one()
