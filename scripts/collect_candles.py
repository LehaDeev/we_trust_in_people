"""
Скрипт сбора исторических свечей из Tinkoff API и сохранения в PostgreSQL.

Запуск:
    python -m scripts.collect_candles

Что делает:
    1. Ищет инструменты по списку тикеров
    2. Сохраняет активы в таблицу assets
    3. Загружает исторические свечи (с последней сохранённой даты)
    4. Сохраняет свечи в таблицу candles (дубликаты игнорируются)
"""
import asyncio
from datetime import datetime, timedelta, timezone

from db.candle_repo import (
    INTERVAL_TO_STR,
    get_last_candle_time,
    get_or_create_asset,
    save_candles,
)
from db.database import close_db, get_session, init_db
from t_tech.invest.schemas import CandleInterval
from tinkoff.instruments import get_instruments_by_tickers
from tinkoff.market_data import fetch_candles_list
from utils.logger import logger

# ── Настройки сбора данных ──────────────────────────────────────────────────

# Список тикеров для сбора данных (голубые фишки MOEX)
TARGET_TICKERS = [
    "SBER",   # Сбербанк
    "GAZP",   # Газпром
    "LKOH",   # ЛУКОЙЛ
    "YDEX",   # Яндекс (MOEX)
    "NVTK",   # НОВАТЭК
    "GMKN",   # Норникель
    "MGNT",   # Магнит
    "TATN",   # Татнефть
    "ROSN",   # Роснефть
    "MTSS",   # МТС
]

# Интервал свечей
CANDLE_INTERVAL = CandleInterval.CANDLE_INTERVAL_HOUR
INTERVAL_STR = INTERVAL_TO_STR[CANDLE_INTERVAL.name]  # "1h"

# Глубина истории при первом запуске (в днях)
HISTORY_DAYS = 365


# ── Основная логика ─────────────────────────────────────────────────────────

async def collect_for_ticker(
    ticker: str,
    figi: str,
    uid: str,
    name: str,
    currency: str,
) -> None:
    """
    Собрать и сохранить свечи для одного инструмента.

    Если свечи уже есть в БД — загружает только новые (инкрементально).
    Если свечей нет — загружает полную историю за HISTORY_DAYS дней.
    """
    async with get_session() as session:
        asset = await get_or_create_asset(
            session=session,
            figi=figi,
            ticker=ticker,
            name=name,
            currency=currency,
        )

        last_time = await get_last_candle_time(
            session=session,
            asset_id=asset.id,
            interval=INTERVAL_STR,
        )

    # Определяем период загрузки
    now = datetime.now(timezone.utc)
    if last_time is not None:
        # Инкрементальное обновление: от последней свечи + 1 секунда
        from_dt = last_time + timedelta(seconds=1)
        logger.info(
            "Incremental update",
            ticker=ticker,
            from_dt=from_dt.isoformat(),
        )
    else:
        # Первый запуск: полная история
        from_dt = now - timedelta(days=HISTORY_DAYS)
        logger.info(
            "Full history load",
            ticker=ticker,
            days=HISTORY_DAYS,
            from_dt=from_dt.isoformat(),
        )

    if from_dt >= now:
        logger.info("Already up to date", ticker=ticker)
        return

    # Загружаем свечи из API (uid приоритетнее figi)
    instrument_id = uid if uid else figi
    candles = await fetch_candles_list(
        instrument_id=instrument_id,
        from_=from_dt,
        to=now,
        interval=CANDLE_INTERVAL,
        only_complete=True,
    )

    if not candles:
        logger.info("No new candles", ticker=ticker)
        return

    # Сохраняем в БД
    async with get_session() as session:
        # Получаем asset_id (сессия новая — нужно снова запросить)
        from sqlalchemy import select
        from db.models import Asset
        result = await session.execute(select(Asset).where(Asset.figi == figi))
        asset = result.scalar_one()

        saved = await save_candles(
            session=session,
            asset_id=asset.id,
            candles=candles,
            interval=INTERVAL_STR,
        )

    logger.info(
        "Ticker done",
        ticker=ticker,
        fetched=len(candles),
        saved=saved,
    )


async def main() -> None:
    """Запустить сбор данных по всем тикерам из TARGET_TICKERS."""
    logger.info("Starting candle collection", tickers=TARGET_TICKERS)

    await init_db()

    try:
        # Найти инструменты в Tinkoff API
        logger.info("Resolving instruments...")
        instruments = await get_instruments_by_tickers(TARGET_TICKERS)

        if not instruments:
            logger.error("No instruments found, aborting")
            return

        logger.info(
            "Instruments resolved",
            found=len(instruments),
            missing=[t for t in TARGET_TICKERS if t not in instruments],
        )

        # Собираем данные последовательно (не параллельно — ограничения API)
        for ticker, info in instruments.items():
            try:
                await collect_for_ticker(
                    ticker=ticker,
                    figi=info.figi,
                    uid=info.uid,
                    name=info.name,
                    currency=info.currency,
                )
            except Exception as e:
                logger.error(
                    "Failed to collect ticker",
                    ticker=ticker,
                    error=str(e),
                )
                continue

        logger.info("Collection complete", processed=len(instruments))

    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
