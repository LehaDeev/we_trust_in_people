"""
Получение рыночных данных через Tinkoff Invest API.
Свечи, текущие цены, стакан.

Кеширование:
    get_last_prices() кешируется в Redis на REDIS_PRICE_TTL секунд (per instrument).
    Батч-запрос делается только для instrument_id, которых нет в кеше.
    При недоступном Redis — прямой вызов API (graceful degradation).
"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator

from t_tech.invest.schemas import (
    CandleInterval,
    HistoricCandle,
    InstrumentIdType,
    LastPrice,
    OrderBook,
)
from t_tech.invest.utils import quotation_to_decimal

from config.settings import redis_settings
from tinkoff.client import get_client
from utils.logger import logger
from utils.redis_cache import get_redis


# Кеш минимального шага цены (меняется крайне редко, хранится в памяти)
_step_cache: dict[str, Decimal] = {}


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

    Каждая цена кешируется в Redis отдельно (ключ: last_price:{instrument_id})
    на REDIS_PRICE_TTL секунд. API вызывается только для instrument_id без кеша.
    При недоступном Redis — прямой вызов API (graceful degradation).

    Args:
        instrument_ids: список FIGI или UID инструментов

    Returns:
        Словарь {instrument_id: цена}
    """
    prices: dict[str, Decimal] = {}
    missing: list[str] = []

    # ── Redis: проверяем кеш для каждого instrument_id отдельно ─────────────
    redis = await get_redis()
    if redis is not None:
        for uid in instrument_ids:
            try:
                cached = await redis.get(f"last_price:{uid}")
                if cached is not None:
                    prices[uid] = Decimal(cached)
                else:
                    missing.append(uid)
            except Exception as e:
                logger.warning("Redis get error", key=f"last_price:{uid}", error=str(e))
                missing.append(uid)
    else:
        missing = list(instrument_ids)

    # ── API: запрашиваем только те, которых нет в кеше ──────────────────────
    if missing:
        async with get_client() as client:
            response = await client.market_data.get_last_prices(
                instrument_id=missing,
            )
        # Индексируем по instrument_uid И по figi — чтобы lookup работал
        # вне зависимости от того, чем вызвали функцию (FIGI или UID)
        fresh: dict[str, Decimal] = {}
        for p in response.last_prices:
            price = quotation_to_decimal(p.price)
            fresh[p.instrument_uid] = price
            if p.figi:
                fresh[p.figi] = price
        prices.update(fresh)

        # ── Redis: сохраняем новые цены ──────────────────────────────────────
        if redis is not None:
            for key, price in fresh.items():
                try:
                    await redis.setex(
                        f"last_price:{key}",
                        redis_settings.price_ttl,
                        str(price),
                    )
                except Exception as e:
                    logger.warning("Redis setex error", key=f"last_price:{key}", error=str(e))

        logger.debug("Last prices fetched from API", count=len(response.last_prices))

    logger.debug("Last prices total", total=len(prices), from_cache=len(instrument_ids) - len(missing))
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
    semaphore = asyncio.Semaphore(max_concurrent)

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


async def get_min_price_increment(instrument_id: str) -> Decimal:
    """
    Получить минимальный шаг цены для инструмента.

    Результат кешируется в памяти процесса (min_price_increment меняется крайне редко).
    При значении ≤ 0 возвращает фоллбэк 0.01 (копейка).

    Аргументы:
        instrument_id: FIGI инструмента

    Возвращает:
        Минимальный шаг цены (Decimal)
    """
    if instrument_id in _step_cache:
        return _step_cache[instrument_id]

    async with get_client() as client:
        response = await client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI,
            id=instrument_id,
        )

    step = quotation_to_decimal(response.instrument.min_price_increment)
    if step <= Decimal("0"):
        step = Decimal("0.01")

    _step_cache[instrument_id] = step
    logger.debug("Min price increment fetched", instrument_id=instrument_id, step=str(step))
    return step
