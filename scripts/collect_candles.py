"""
Скрипт сбора исторических свечей из Tinkoff API и сохранения в PostgreSQL.

Запуск:
    python -m scripts.collect_candles                  # инкрементальное обновление
    python -m scripts.collect_candles --full-reload    # перезагрузка с DATA_START_DATE / DATA_HISTORY_DAYS

Что делает:
    1. Ищет инструменты по списку тикеров из .env (DATA_TICKERS)
    2. Сохраняет активы в таблицу assets
    3. Загружает исторические свечи (с последней сохранённой даты)
    4. Сохраняет свечи в таблицу candles (дубликаты игнорируются)

Все настройки — в .env:
    DATA_TICKERS         — список тикеров через запятую
    DATA_CANDLE_INTERVAL — интервал свечей (1h, 1d и т.д.)
    DATA_HISTORY_DAYS    — глубина истории при первом запуске
    DATA_START_DATE      — фиксированная дата начала (YYYY-MM-DD), используется при --full-reload
"""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from grpc import RpcError, StatusCode

from config.settings import data_settings
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

# Обратный маппинг: строка интервала ("1h") → имя enum ("CANDLE_INTERVAL_HOUR")
_STR_TO_INTERVAL_NAME: dict[str, str] = {v: k for k, v in INTERVAL_TO_STR.items()}


def _get_candle_interval(interval_str: str) -> CandleInterval:
    """
    Преобразовать строку интервала из настроек в CandleInterval enum.

    Аргументы:
        interval_str: строка из DATA_CANDLE_INTERVAL ("1h", "1d" и т.д.)

    Возвращает:
        Соответствующий элемент CandleInterval.

    Исключения:
        ValueError: если строка не найдена в маппинге.
    """
    enum_name = _STR_TO_INTERVAL_NAME.get(interval_str)
    if enum_name is None:
        valid = list(_STR_TO_INTERVAL_NAME.keys())
        raise ValueError(
            f"Неизвестный интервал '{interval_str}'. Допустимые: {valid}"
        )
    return CandleInterval[enum_name]


async def collect_for_ticker(
    ticker: str,
    figi: str,
    uid: str,
    name: str,
    currency: str,
    candle_interval: CandleInterval,
    interval_str: str,
    history_days: int,
    full_reload: bool = False,
) -> None:
    """
    Собрать и сохранить свечи для одного инструмента.

    Если свечи уже есть в БД — загружает только новые (инкрементально).
    Если свечей нет или full_reload=True — загружает полную историю
    от DATA_START_DATE (или за history_days дней если дата не задана).
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
            interval=interval_str,
        )

    # Определяем период загрузки
    now = datetime.now(timezone.utc)
    if last_time is not None and not full_reload:
        # Инкрементальное обновление: от последней свечи + 1 секунда
        from_dt = last_time + timedelta(seconds=1)
        logger.info(
            "Incremental update",
            ticker=ticker,
            from_dt=from_dt.isoformat(),
        )
    else:
        # Полная история: от фиксированной даты или от history_days
        if data_settings.start_date:
            from_dt = datetime.fromisoformat(data_settings.start_date).replace(
                tzinfo=timezone.utc
            )
            logger.info(
                "Full history load",
                ticker=ticker,
                start_date=data_settings.start_date,
                from_dt=from_dt.isoformat(),
            )
        else:
            from_dt = now - timedelta(days=history_days)
            logger.info(
                "Full history load",
                ticker=ticker,
                days=history_days,
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
        interval=candle_interval,
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
            interval=interval_str,
        )

    logger.info(
        "Ticker done",
        ticker=ticker,
        fetched=len(candles),
        saved=saved,
    )


async def _collect_with_retry(
    *,
    ticker: str,
    figi: str,
    uid: str,
    name: str,
    currency: str,
    candle_interval,
    interval_str: str,
    history_days: int,
    full_reload: bool = False,
    warn_on_error: bool = False,
    max_retries: int = 3,
    retry_delay: int = 60,
) -> None:
    """
    Собрать свечи для тикера с retry при RESOURCE_EXHAUSTED.

    При превышении rate limit ждёт retry_delay секунд и повторяет попытку.
    При warn_on_error=True — логирует как warning, иначе как error.
    """
    for attempt in range(1, max_retries + 1):
        try:
            await collect_for_ticker(
                ticker=ticker,
                figi=figi,
                uid=uid,
                name=name,
                currency=currency,
                candle_interval=candle_interval,
                interval_str=interval_str,
                history_days=history_days,
                full_reload=full_reload,
            )
            return
        except Exception as e:
            # Проверяем RESOURCE_EXHAUSTED по типу и коду gRPC-исключения
            is_rate_limit = isinstance(e, RpcError) and e.code() == StatusCode.RESOURCE_EXHAUSTED
            if is_rate_limit and attempt < max_retries:
                logger.warning(
                    "Rate limit, повтор через %d сек", retry_delay,
                    ticker=ticker, attempt=attempt,
                )
                await asyncio.sleep(retry_delay)
            else:
                if warn_on_error:
                    logger.warning("Не удалось собрать свечи", ticker=ticker, error=str(e))
                else:
                    logger.error("Ошибка сбора тикера", ticker=ticker, error=str(e))
                return


async def run_collection(
    full_reload: bool = False,
    pause_seconds: int | None = None,
) -> None:
    """
    Собрать новые свечи по всем тикерам из настроек.

    В отличие от main() — не управляет жизненным циклом БД (init_db / close_db).
    Предназначена для вызова внутри уже запущенного процесса (бот, планировщик).

    Аргументы:
        full_reload:   загрузить полную историю вместо инкрементального обновления.
        pause_seconds: пауза между тикерами (сек). None = из DATA_COLLECT_PAUSE_SECONDS.
                       Для внутрисессионного обновления передавать малое значение (2–5 сек).
    """
    tickers = data_settings.tickers
    candle_interval = _get_candle_interval(data_settings.candle_interval)
    interval_str = data_settings.candle_interval
    history_days = data_settings.history_days

    logger.info(
        "Инкрементальный сбор свечей",
        tickers=tickers,
        interval=interval_str,
        full_reload=full_reload,
    )

    instruments = await get_instruments_by_tickers(tickers)
    if not instruments:
        raise RuntimeError("Инструменты не найдены в Tinkoff API")

    missing = [t for t in tickers if t not in instruments]
    if missing:
        logger.warning("Тикеры не найдены в API", missing=missing)

    pause = pause_seconds if pause_seconds is not None else data_settings.collect_pause_seconds
    tickers_list = list(instruments.items())

    for i, (ticker, info) in enumerate(tickers_list):
        await _collect_with_retry(
            ticker=ticker,
            figi=info.figi,
            uid=info.uid,
            name=info.name,
            currency=info.currency,
            candle_interval=candle_interval,
            interval_str=interval_str,
            history_days=history_days,
            full_reload=full_reload,
        )
        # Пауза между тикерами кроме последнего — защита от RESOURCE_EXHAUSTED
        if i < len(tickers_list) - 1:
            logger.debug("Пауза между тикерами", seconds=pause)
            await asyncio.sleep(pause)

    logger.info("Сбор свечей завершён", processed=len(instruments))

    # Собираем USD/RUB для ML-признаков (graceful degradation при ошибке)
    await _collect_with_retry(
        ticker="USDRUB",
        figi=data_settings.usdrub_figi,
        uid=data_settings.usdrub_figi,
        name="USD/RUB",
        currency="RUB",
        candle_interval=candle_interval,
        interval_str=interval_str,
        history_days=history_days,
        full_reload=full_reload,
        warn_on_error=True,
    )


async def main() -> None:
    """Запустить сбор данных по всем тикерам из настроек."""
    parser = argparse.ArgumentParser(description="Сбор свечей из Tinkoff API")
    parser.add_argument(
        "--full-reload",
        action="store_true",
        help="Загрузить историю с DATA_START_DATE (или DATA_HISTORY_DAYS) игнорируя уже сохранённые данные",
    )
    args = parser.parse_args()

    logger.info(
        "Starting candle collection",
        tickers=data_settings.tickers,
        interval=data_settings.candle_interval,
        history_days=data_settings.history_days,
        full_reload=args.full_reload,
    )

    await init_db()

    try:
        await run_collection(full_reload=args.full_reload)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
