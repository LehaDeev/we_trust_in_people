"""
Статистика закрытия дивидендного гэпа для инструментов MOEX.

Алгоритм вычисления:
    1. Получить исторические дивиденды инструмента (последние 5 лет)
    2. Для каждого дивиденда (только те, что были >= 90 дней назад):
       - базовая цена = цена закрытия последней свечи ДО экс-даты
       - идём по дневным свечам после экс-даты
       - первый день когда цена >= базовая × 0.98 → записываем кол-во дней
       - если за 90 дней не восстановилась → записываем 90
    3. Возвращаем ceiling(среднее) по всем событиям

Пример (SBER):
    Дивиденды за 5 лет: 5 событий, закрытия: [14, 21, 30, 45, 90]
    Среднее: ceil(40) = 40 дней → dividend_gap_days = 40 для SBER

Хранение:
    Результат сохраняется в PostgreSQL (Asset.dividend_gap_days).
    Пересчёт происходит если значение None или старше GAP_STATS_REFRESH_DAYS дней.
    Ручное переопределение через SQL:
        UPDATE assets SET dividend_gap_days = 45 WHERE ticker = 'SBER';
"""
import asyncio
from datetime import datetime, timedelta, timezone
from math import ceil
from statistics import mean

from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Asset
from tinkoff.client import get_client
from tinkoff.market_data import INTERVAL_1DAY, fetch_candles_list
from utils.logger import logger

# Акция считается "восстановившейся" при достижении этой доли предивидендной цены
_RECOVERY_THRESHOLD = 0.98

# Максимум дней наблюдения за одним гэпом
_MAX_OBSERVATION_DAYS = 90

# Минимум событий для достоверной статистики
_MIN_SAMPLES = 2

# Глубина исторического анализа (лет)
_HISTORY_YEARS = 5

# Дивиденды до которых прошло менее N дней — пропускаем (нет полной истории после)
_MIN_DAYS_PAST = _MAX_OBSERVATION_DAYS

# Пересчитывать статистику если она старше N дней
GAP_STATS_REFRESH_DAYS = 30


async def compute_gap_closure_days(figi: str) -> int | None:
    """
    Вычислить среднее количество дней до закрытия дивидендного гэпа
    на основе исторических данных Tinkoff API.

    Аргументы:
        figi: FIGI инструмента

    Возвращает:
        Среднее кол-во дней (ceiling) или None при нехватке данных.
    """
    today = datetime.now(timezone.utc)
    history_start = today - timedelta(days=365 * _HISTORY_YEARS)

    # ── 1. Исторические дивиденды ─────────────────────────────────────────────
    try:
        async with get_client() as client:
            div_response = await client.instruments.get_dividends(
                figi=figi,
                from_=history_start,
                to=today,
            )
    except Exception as e:
        logger.warning("Ошибка получения дивидендов для статистики", figi=figi, error=str(e))
        return None

    closure_days: list[int] = []

    for div in div_response.dividends:
        if not div.last_buy_date:
            continue

        last_buy = div.last_buy_date
        if hasattr(last_buy, "date"):
            last_buy_date = last_buy.date()
        else:
            last_buy_date = last_buy.replace(tzinfo=timezone.utc).date()

        ex_date = last_buy_date + timedelta(days=1)

        # Пропускаем свежие события — нет достаточной истории после них
        if (today.date() - ex_date).days < _MIN_DAYS_PAST:
            continue

        # ── 2. Дневные свечи вокруг экс-даты ─────────────────────────────────
        from_dt = datetime(
            ex_date.year, ex_date.month, ex_date.day, tzinfo=timezone.utc
        ) - timedelta(days=5)
        to_dt = datetime(
            ex_date.year, ex_date.month, ex_date.day, tzinfo=timezone.utc
        ) + timedelta(days=_MAX_OBSERVATION_DAYS + 2)

        try:
            candles = await fetch_candles_list(
                instrument_id=figi,
                from_=from_dt,
                to=to_dt,
                interval=INTERVAL_1DAY,
            )
        except Exception as e:
            logger.warning(
                "Ошибка свечей для статистики гэпа",
                figi=figi,
                ex_date=ex_date,
                error=str(e),
            )
            continue

        if len(candles) < 3:
            continue

        # Базовая цена: последняя свеча ДО экс-даты
        pre_ex = [c for c in candles if c["time"].date() < ex_date]
        if not pre_ex:
            continue
        baseline = float(pre_ex[-1]["close"])
        recovery_target = baseline * _RECOVERY_THRESHOLD

        # ── 3. Поиск дня восстановления ──────────────────────────────────────
        post_ex = [c for c in candles if c["time"].date() >= ex_date]
        days_to_close = _MAX_OBSERVATION_DAYS

        for candle in post_ex:
            elapsed = (candle["time"].date() - ex_date).days
            if float(candle["close"]) >= recovery_target:
                days_to_close = elapsed
                break

        closure_days.append(days_to_close)
        logger.debug(
            "Гэп проанализирован",
            figi=figi,
            ex_date=ex_date,
            baseline=f"{baseline:.2f}",
            days_to_close=days_to_close,
        )

    if len(closure_days) < _MIN_SAMPLES:
        logger.info(
            "Недостаточно данных для статистики гэпа",
            figi=figi,
            samples=len(closure_days),
            required=_MIN_SAMPLES,
        )
        return None

    result = ceil(mean(closure_days))
    logger.info(
        "Статистика дивидендного гэпа вычислена",
        figi=figi,
        avg_days=result,
        samples=len(closure_days),
        all_values=closure_days,
    )
    return result


async def refresh_gap_days_if_stale(
    asset: Asset,
    session: AsyncSession,
    fallback: int,
) -> int:
    """
    Вернуть актуальное значение окна защиты для актива.

    Если значение в БД не устарело — возвращает его.
    Если устарело или отсутствует — пересчитывает и сохраняет в БД.

    Аргументы:
        asset:    объект Asset (из запроса scheduler'а)
        session:  активная SQLAlchemy-сессия
        fallback: значение при нехватке исторических данных

    Возвращает:
        Кол-во дней защиты (≥ 1).
    """
    now = datetime.now(timezone.utc)
    is_stale = (
        asset.dividend_gap_days is None
        or asset.dividend_gap_updated_at is None
        or (now - asset.dividend_gap_updated_at).days >= GAP_STATS_REFRESH_DAYS
    )

    if not is_stale:
        return asset.dividend_gap_days  # type: ignore[return-value]

    # Нужен пересчёт
    logger.info(
        "Пересчёт статистики дивидендного гэпа",
        ticker=asset.ticker,
        figi=asset.figi,
        current_value=asset.dividend_gap_days,
        last_updated=asset.dividend_gap_updated_at,
    )
    computed = await compute_gap_closure_days(asset.figi)

    if computed is not None:
        asset.dividend_gap_days = computed
        asset.dividend_gap_updated_at = now
        await session.flush()  # сохраняем без отдельного commit (в рамках тика)
        logger.info(
            "dividend_gap_days сохранён в БД",
            ticker=asset.ticker,
            days=computed,
        )
        return computed

    # Данных недостаточно — используем fallback, но не перезаписываем None в БД
    # (попробуем снова при следующем пересчёте через 30 дней)
    if asset.dividend_gap_days is not None:
        return asset.dividend_gap_days
    return fallback


async def get_gap_protection_days_bulk(
    assets: list[Asset],
    session: AsyncSession,
    fallback: int,
) -> dict[str, int]:
    """
    Получить индивидуальные окна защиты SL для списка активов.

    Читает из БД (Asset.dividend_gap_days). Если значение устарело или
    отсутствует — пересчитывает параллельно и сохраняет обратно в БД.

    Аргументы:
        assets:   список объектов Asset с открытыми позициями
        session:  активная SQLAlchemy-сессия
        fallback: глобальное значение по умолчанию (TRADING_DIVIDEND_PROTECTION_DAYS)

    Возвращает:
        Словарь {figi: protection_days}.
    """
    tasks = [
        refresh_gap_days_if_stale(asset, session, fallback)
        for asset in assets
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    per_figi: dict[str, int] = {}
    for asset, result in zip(assets, results):
        if isinstance(result, Exception):
            logger.warning(
                "Ошибка получения gap_protection_days",
                ticker=asset.ticker,
                error=str(result),
            )
            per_figi[asset.figi] = fallback
        else:
            per_figi[asset.figi] = result  # type: ignore[assignment]

    return per_figi
