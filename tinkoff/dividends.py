"""
Получение дивидендных данных через Tinkoff Invest API.

Используется для защиты стоп-лосса от дивидендного гэпа.

Механика гэпа (MOEX):
    - Экс-дивидендная дата = last_buy_date + 1 день
    - В экс-дату акция открывается дешевле примерно на размер дивиденда
    - После закрытия реестра цена восстанавливается — в среднем за 30–90 дней

Стратегия защиты (TRADING_DIVIDEND_PROTECTION_DAYS = N):
    Снижаем эффективный порог SL на размер дивиденда на протяжении N дней,
    начиная с экс-дивидендной даты: ex_div_date ≤ today < ex_div_date + N.
    Это не допускает ложного срабатывания SL из-за предсказуемого гэпа.

    Примеры:
        N=1  — защита только в день открытия гэпа (минимум)
        N=3  — гэп + 2 дня восстановления
        N=7  — неделя защиты для крупных дивидендов (>5%)

Кеширование:
    Ключ: dividend_drop:{figi}:{for_date}:{protection_days}
    TTL: REDIS_DIVIDEND_TTL секунд (по умолчанию 24 часа).
    При недоступном Redis — прямой вызов API (graceful degradation).
"""
import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from t_tech.invest.utils import money_to_decimal

from config.settings import redis_settings
from tinkoff.client import get_client
from utils.logger import logger
from utils.redis_cache import get_redis


async def get_dividend_drop(
    figi: str,
    for_date: date | None = None,
    protection_days: int = 1,
) -> Decimal:
    """
    Вычислить суммарную дивидендную корректировку SL на указанный день.

    Возвращает размер дивиденда на акцию, если for_date попадает в защитное
    окно [ex_div_date, ex_div_date + protection_days). В остальных случаях — 0.

    Экс-дивидендная дата = last_buy_date + 1 день (правило MOEX T+1/T+2).

    Аргументы:
        figi:            FIGI инструмента
        for_date:        проверяемая дата (по умолчанию — сегодня UTC)
        protection_days: ширина защитного окна в торговых днях (≥ 1)

    Возвращает:
        Суммарный дивиденд на акцию (₽). Decimal("0") если окно не активно.
    """
    if for_date is None:
        for_date = datetime.now(timezone.utc).date()

    cache_key = f"dividend_drop:{figi}:{for_date.isoformat()}:{protection_days}"

    # ── Redis: пробуем кеш ────────────────────────────────────────────────────
    redis = await get_redis()
    if redis is not None:
        try:
            cached = await redis.get(cache_key)
            if cached is not None:
                logger.debug("Дивиденд из кеша", figi=figi, date=for_date, amount=cached)
                return Decimal(cached)
        except Exception as e:
            logger.warning("Redis get error", key=cache_key, error=str(e))

    # ── Tinkoff API: запрашиваем дивиденды в диапазоне ───────────────────────
    # Нам нужны все дивиденды, чья экс-дата могла попасть в окно защиты.
    # Окно [for_date - (protection_days - 1), for_date], т.е. смотрим назад
    # на protection_days дней (ex_div_date попала туда ≤ protection_days назад).
    api_lookback = max(7, protection_days)
    from_dt = datetime(
        for_date.year, for_date.month, for_date.day, tzinfo=timezone.utc
    ) - timedelta(days=api_lookback)
    to_dt = datetime(
        for_date.year, for_date.month, for_date.day, tzinfo=timezone.utc
    ) + timedelta(days=2)  # небольшой буфер вперёд

    total_dividend = Decimal("0")
    try:
        async with get_client() as client:
            response = await client.instruments.get_dividends(
                figi=figi,
                from_=from_dt,
                to=to_dt,
            )

        for div in response.dividends:
            if not div.last_buy_date:
                continue

            # Экс-дивидендная дата = last_buy_date + 1 день
            last_buy = div.last_buy_date
            if hasattr(last_buy, "date"):
                last_buy_date = last_buy.date()
            else:
                last_buy_date = last_buy.replace(tzinfo=timezone.utc).date()

            ex_div_date = last_buy_date + timedelta(days=1)

            # Проверяем: попадает ли for_date в защитное окно этого дивиденда
            # Окно: [ex_div_date, ex_div_date + protection_days)
            window_end = ex_div_date + timedelta(days=protection_days)
            if ex_div_date <= for_date < window_end:
                amount = money_to_decimal(div.dividend_net)
                days_since_gap = (for_date - ex_div_date).days
                logger.info(
                    "Дивидендная защита активна",
                    figi=figi,
                    ex_div_date=ex_div_date,
                    for_date=for_date,
                    days_since_gap=days_since_gap,
                    protection_days=protection_days,
                    dividend_per_share=str(amount),
                    dividend_type=div.dividend_type,
                )
                total_dividend += amount

    except Exception as e:
        logger.warning(
            "Не удалось получить дивиденды",
            figi=figi,
            error=str(e),
        )
        # Graceful degradation: возвращаем 0, SL не корректируем
        return Decimal("0")

    # ── Redis: сохраняем результат ────────────────────────────────────────────
    if redis is not None:
        try:
            await redis.setex(cache_key, redis_settings.dividend_ttl, str(total_dividend))
        except Exception as e:
            logger.warning("Redis setex error", key=cache_key, error=str(e))

    logger.debug(
        "Дивидендная корректировка", figi=figi, date=for_date, total=str(total_dividend)
    )
    return total_dividend


async def get_dividend_drops_bulk(
    figis: list[str],
    for_date: date | None = None,
    per_figi_days: dict[str, int] | None = None,
    protection_days: int = 1,
) -> dict[str, Decimal]:
    """
    Получить дивидендные корректировки SL для списка инструментов.

    Если передан per_figi_days — использует индивидуальное окно для каждого
    инструмента. Иначе — единое protection_days для всех.

    Аргументы:
        figis:           список FIGI инструментов
        for_date:        дата для проверки (по умолчанию — сегодня UTC)
        per_figi_days:   индивидуальные окна {figi: days} (из dividend_gap_stats)
        protection_days: запасной дефолт если per_figi_days не передан

    Возвращает:
        Словарь {figi: dividend_per_share}; 0 если окно не активно.
    """
    def _days_for(figi: str) -> int:
        if per_figi_days is not None:
            return per_figi_days.get(figi, protection_days)
        return protection_days

    tasks = [get_dividend_drop(figi, for_date, _days_for(figi)) for figi in figis]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    drops: dict[str, Decimal] = {}
    for figi, result in zip(figis, results):
        if isinstance(result, Exception):
            logger.warning("Ошибка получения дивиденда", figi=figi, error=str(result))
            drops[figi] = Decimal("0")
        else:
            drops[figi] = result  # type: ignore[assignment]

    return drops
