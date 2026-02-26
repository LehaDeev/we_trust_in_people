"""
Получение рыночных данных через Tinkoff Invest API.
Свечи, текущие цены, стакан.
"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator

from t_tech.invest.schemas import (
    CandleInterval,
    HistoricCandle,
    LastPrice,
    OrderBook,
)
from t_tech.invest.utils import quotation_to_decimal

from tinkoff.client import get_client
from utils.logger import logger


# Удобные алиасы для часто используемых интервалов
INTERVAL_1MIN = CandleInterval.CANDLE_INTERVAL_1_MIN
INTERVAL_5MIN = CandleInterval.CANDLE_INTERVAL_5_MIN
INTERVAL_15MIN = CandleInterval.CANDLE_INTERVAL_15_MIN
INTERVAL_1HOUR = CandleInterval.CANDLE_INTERVAL_HOUR
INTERVAL_1DAY = CandleInterval.CANDLE_INTERVAL_DAY
INTERVAL_1WEEK = CandleInterval.CANDLE_INTERVAL_WEEK


async def get_candles(
    instrument_id: str,
    from_: datetime,
    to: datetime | None = None,
    interval: CandleInterval = INTERVAL_1HOUR,
) -> AsyncGenerator[HistoricCandle, None]:
    """
    Async-генератор исторических свечей для инструмента.

    Args:
        instrument_id: FIGI или UID инструмента
        from_: начало периода (timezone-aware)
        to: конец периода (None = текущий момент)
        interval: интервал свечи (по умолчанию 1 час)

    Yields:
        HistoricCandle — свеча с полями open, high, low, close, volume, time
    """
    async with get_client() as client:
        async for candle in client.get_all_candles(
            instrument_id=instrument_id,
            from_=from_,
            to=to,
            interval=interval,
        ):
            yield candle


async def fetch_candles_list(
    instrument_id: str,
    from_: datetime,
    to: datetime | None = None,
    interval: CandleInterval = INTERVAL_1HOUR,
    only_complete: bool = True,
) -> list[dict]:
    """
    Загружает свечи и возвращает список словарей (удобно для pandas).

    Args:
        instrument_id: FIGI или UID инструмента
        from_: начало периода
        to: конец периода
        interval: интервал свечи
        only_complete: брать только закрытые свечи (is_complete=True)

    Returns:
        Список словарей с ключами: time, open, high, low, close, volume
    """
    result = []
    async for candle in get_candles(instrument_id, from_, to, interval):
        if only_complete and not candle.is_complete:
            continue
        result.append({
            "time": candle.time,
            "open": quotation_to_decimal(candle.open),
            "high": quotation_to_decimal(candle.high),
            "low": quotation_to_decimal(candle.low),
            "close": quotation_to_decimal(candle.close),
            "volume": candle.volume,
        })

    logger.info(
        "Candles fetched",
        instrument_id=instrument_id,
        count=len(result),
        interval=interval.name,
    )
    return result


async def get_last_prices(instrument_ids: list[str]) -> dict[str, Decimal]:
    """
    Получить текущие цены последней сделки для списка инструментов.

    Args:
        instrument_ids: список FIGI или UID инструментов

    Returns:
        Словарь {instrument_id: цена}
    """
    async with get_client() as client:
        response = await client.market_data.get_last_prices(
            instrument_id=instrument_ids,
        )
    prices = {
        p.instrument_uid: quotation_to_decimal(p.price)
        for p in response.last_prices
    }
    logger.debug("Last prices fetched", count=len(prices))
    return prices


async def get_last_price(instrument_id: str) -> Decimal:
    """
    Получить текущую цену одного инструмента.

    Args:
        instrument_id: FIGI или UID инструмента

    Returns:
        Текущая цена
    """
    prices = await get_last_prices([instrument_id])
    return prices.get(instrument_id, Decimal("0"))


async def get_order_book(
    instrument_id: str,
    depth: int = 10,
) -> OrderBook:
    """
    Получить стакан заявок для инструмента.

    Args:
        instrument_id: FIGI или UID инструмента
        depth: глубина стакана (1-50)

    Returns:
        OrderBook с bids и asks
    """
    async with get_client() as client:
        order_book = await client.market_data.get_order_book(
            instrument_id=instrument_id,
            depth=depth,
        )
    logger.debug(
        "Order book fetched",
        instrument_id=instrument_id,
        bids=len(order_book.bids),
        asks=len(order_book.asks),
    )
    return order_book


async def fetch_multiple_instruments_prices(
    instrument_ids: list[str],
    max_concurrent: int = 5,
) -> dict[str, Decimal]:
    """
    Параллельное получение цен для большого списка инструментов.
    Разбивает на батчи по max_concurrent штук.

    Args:
        instrument_ids: список инструментов
        max_concurrent: максимальный размер одного запроса

    Returns:
        Словарь {instrument_id: цена}
    """
    all_prices: dict[str, Decimal] = {}
    semaphore = asyncio.Semaphore(3)

    async def fetch_batch(batch: list[str]) -> dict[str, Decimal]:
        async with semaphore:
            return await get_last_prices(batch)

    batches = [
        instrument_ids[i:i + max_concurrent]
        for i in range(0, len(instrument_ids), max_concurrent)
    ]

    tasks = [fetch_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error("Batch price fetch failed", error=str(result))
            continue
        all_prices.update(result)

    return all_prices
