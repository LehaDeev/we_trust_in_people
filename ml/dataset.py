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


async def load_usdrub_data(interval: str = "1h") -> pd.DataFrame:
    """
    Загрузить свечи USD/RUB из БД.

    Возвращает DataFrame с колонками [time, open, high, low, close, volume]
    или пустой DataFrame если данные не собраны.
    """
    return await load_ticker_data("USDRUB", interval)


def merge_usdrub(df: pd.DataFrame, usdrub_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить колонку usdrub_close к DataFrame со свечами тикера.

    Использует merge_asof по времени (backward — последнее известное значение).
    При пустом usdrub_df — graceful degradation: колонка заполняется нулями.

    Аргументы:
        df:        DataFrame тикера с колонкой time.
        usdrub_df: DataFrame USD/RUB с колонками [time, close].

    Возвращает:
        df с добавленной колонкой usdrub_close.
    """
    df = df.copy()

    if usdrub_df.empty:
        df["usdrub_close"] = 0.0
        return df

    def _to_naive_utc(s: pd.Series) -> pd.Series:
        """Привести timestamps к наивному UTC для корректного merge."""
        s = pd.to_datetime(s)
        if s.dt.tz is not None:
            return s.dt.tz_convert("UTC").dt.tz_localize(None)
        return s

    df["_t"] = _to_naive_utc(df["time"])
    usdrub = usdrub_df[["time", "close"]].copy()
    usdrub["_t"] = _to_naive_utc(usdrub["time"])
    usdrub = (
        usdrub[["_t", "close"]]
        .rename(columns={"close": "usdrub_close"})
        .sort_values("_t")
        .drop_duplicates("_t")
    )

    merged = pd.merge_asof(
        df.sort_values("_t"),
        usdrub,
        on="_t",
        direction="backward",
    )
    merged["usdrub_close"] = merged["usdrub_close"].ffill().fillna(0.0)
    merged = merged.drop(columns=["_t"]).sort_values("time").reset_index(drop=True)
    return merged


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

    # Загружаем USD/RUB один раз — мержим к каждому тикеру
    usdrub_df = await load_usdrub_data(interval)
    if usdrub_df.empty:
        logger.warning("USD/RUB данные не найдены — признак usdrub будет нулевым. "
                       "Запустите scripts/collect_candles.py для сбора данных.")

    for ticker in tickers:
        df = await load_ticker_data(ticker, interval)
        if df.empty:
            logger.warning("Skipping ticker — no data", ticker=ticker)
            continue
        df = merge_usdrub(df, usdrub_df)
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
