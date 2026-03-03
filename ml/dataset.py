"""
Утилиты загрузки данных для ML-pipeline.

Загружает OHLCV свечи из PostgreSQL через async SQLAlchemy
и возвращает pandas DataFrame, готовые для вычисления признаков.

Кеширование:
    load_ticker_data() кеширует результат в Redis на REDIS_CANDLES_TTL секунд.
    При недоступном Redis — прямой запрос к БД (graceful degradation).
    load_all_tickers_dataset() (используется при обучении) не кешируется —
    при обучении нужны свежие данные без TTL-ограничений.
"""
import io

import pandas as pd
from sqlalchemy import select

from config.settings import redis_settings
from db.database import get_session
from db.models import Asset, Candle
from utils.logger import logger
from utils.redis_cache import get_redis


async def load_ticker_data(
    ticker: str,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Загрузить все свечи для тикера из базы данных.

    Результат кешируется в Redis на REDIS_CANDLES_TTL секунд.
    При недоступном Redis — прямой запрос к БД (graceful degradation).

    Аргументы:
        ticker: тикер инструмента (например, "SBER", "GAZP").
        interval: строка интервала свечи как в БД (например, "1h", "1d").

    Возвращает:
        DataFrame с колонками [time, open, high, low, close, volume],
        отсортированный по времени ASC, цены в float64.
        Возвращает пустой DataFrame если тикер не найден.
    """
    # ── Redis: проверяем кеш ─────────────────────────────────────────────────
    cache_key = f"candles:{ticker}:{interval}"
    redis = await get_redis()
    if redis is not None:
        try:
            cached = await redis.get(cache_key)
            if cached:
                logger.debug("Candles cache hit", ticker=ticker, key=cache_key)
                df = pd.read_json(io.StringIO(cached), orient="split")
                # Восстанавливаем типы после JSON-сериализации
                for col in ("open", "high", "low", "close", "volume"):
                    df[col] = df[col].astype(float)
                return df
        except Exception as e:
            logger.warning("Redis get error", key=cache_key, error=str(e))

    # ── Запрос к базе данных ─────────────────────────────────────────────────
    async with get_session() as session:
        stmt = (
            select(
                Candle.time,
                Candle.open,
                Candle.high,
                Candle.low,
                Candle.close,
                Candle.volume,
            )
            .join(Asset, Candle.asset_id == Asset.id)
            .where(Asset.ticker == ticker)
            .where(Candle.interval == interval)
            .order_by(Candle.time.asc())
        )
        result = await session.execute(stmt)
        rows = result.fetchall()

    if not rows:
        logger.warning("No candles found", ticker=ticker, interval=interval)
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])

    # Преобразуем Decimal → float64 для совместимости с TA-Lib
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)

    logger.info(
        "Ticker data loaded",
        ticker=ticker,
        interval=interval,
        rows=len(df),
    )

    # ── Redis: сохраняем в кеш ────────────────────────────────────────────────
    if redis is not None:
        try:
            await redis.setex(
                cache_key,
                redis_settings.candles_ttl,
                df.to_json(orient="split"),
            )
            logger.debug("Candles cached", ticker=ticker, ttl=redis_settings.candles_ttl)
        except Exception as e:
            logger.warning("Redis setex error", key=cache_key, error=str(e))

    return df


async def load_all_tickers_dataset(
    tickers: list[str],
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Загрузить свечи для нескольких тикеров и объединить в один DataFrame.

    Используется при обучении модели — кеш намеренно не применяется,
    чтобы обучение всегда работало с актуальными данными из БД.

    Каждая строка получает колонку "ticker" для группировки при вычислении признаков
    (индикаторы должны считаться отдельно по каждому тикеру, а не по всем данным сразу).

    Аргументы:
        tickers: список тикеров.
        interval: строка интервала свечи (например, "1h").

    Возвращает:
        Объединённый DataFrame с колонками [ticker, time, open, high, low, close, volume].
        Тикеры без данных пропускаются.
    """
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        df = await load_ticker_data(ticker, interval)
        if df.empty:
            logger.warning("Skipping ticker — no data", ticker=ticker)
            continue
        df.insert(0, "ticker", ticker)
        frames.append(df)

    if not frames:
        logger.error("No data loaded for any ticker")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "All tickers loaded",
        tickers=len(frames),
        total_rows=len(combined),
    )
    return combined
