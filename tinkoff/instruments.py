"""
Поиск и получение информации об инструментах через Tinkoff Invest API.
Акции, ETF, облигации — поиск по тикеру или FIGI.
"""
from dataclasses import dataclass, field

from tinkoff.client import get_client
from utils.logger import logger

# Приоритет class_code: TQBR — основной рынок MOEX (Т+2), лучший для ML
MOEX_MAIN_CLASS_CODES = ("TQBR", "TQDE", "TQTF")


@dataclass
class InstrumentInfo:
    """Основная информация об инструменте."""
    figi: str
    uid: str
    ticker: str
    name: str
    currency: str
    instrument_type: str  # "share", "etf", "bond"
    class_code: str = field(default="")


async def find_instrument(query: str) -> list[InstrumentInfo]:
    """
    Поиск инструментов по тикеру или названию.

    Args:
        query: тикер (SBER, GAZP) или название компании

    Returns:
        Список найденных инструментов
    """
    async with get_client() as client:
        response = await client.instruments.find_instrument(query=query)

    result = []
    for item in response.instruments:
        result.append(InstrumentInfo(
            figi=item.figi,
            uid=item.uid,
            ticker=item.ticker,
            name=item.name,
            currency="RUB",  # InstrumentShort не содержит currency
            instrument_type=str(item.instrument_type),
            class_code=getattr(item, "class_code", ""),
        ))

    logger.debug("Instruments found", query=query, count=len(result))
    return result


async def get_instrument_by_ticker(ticker: str) -> InstrumentInfo | None:
    """
    Найти инструмент по точному совпадению тикера.

    Приоритет выбора:
      1. share + TQBR/TQDE/TQTF (основной рынок MOEX)
      2. share + любой class_code
      3. etf + MOEX class_code
      4. первый подходящий

    Args:
        ticker: тикер инструмента (SBER, GAZP, YNDX)

    Returns:
        InstrumentInfo или None если не найден
    """
    instruments = await find_instrument(ticker)

    # Точное совпадение тикера
    exact = [i for i in instruments if i.ticker.upper() == ticker.upper()]
    if not exact:
        logger.warning("Instrument not found by ticker", ticker=ticker)
        return None

    # 1. Акция на основном рынке MOEX
    for item in exact:
        if item.instrument_type == "share" and item.class_code in MOEX_MAIN_CLASS_CODES:
            logger.info(
                "Instrument resolved",
                ticker=ticker,
                figi=item.figi,
                class_code=item.class_code,
                name=item.name,
            )
            return item

    # 2. Любая акция
    for item in exact:
        if item.instrument_type == "share":
            logger.info(
                "Instrument resolved (non-MOEX share)",
                ticker=ticker,
                figi=item.figi,
                class_code=item.class_code,
            )
            return item

    # 3. ETF на MOEX
    for item in exact:
        if item.instrument_type == "etf" and item.class_code in MOEX_MAIN_CLASS_CODES:
            logger.info(
                "Instrument resolved (ETF)",
                ticker=ticker,
                figi=item.figi,
            )
            return item

    logger.warning("No preferred instrument found, using first match", ticker=ticker)
    return exact[0]


async def get_instruments_by_tickers(
    tickers: list[str],
) -> dict[str, InstrumentInfo]:
    """
    Получить информацию по списку тикеров.

    Args:
        tickers: список тикеров

    Returns:
        Словарь {ticker: InstrumentInfo}, пропускает ненайденные
    """
    result: dict[str, InstrumentInfo] = {}
    for ticker in tickers:
        info = await get_instrument_by_ticker(ticker)
        if info:
            result[ticker] = info
        else:
            logger.warning("Skipping ticker — not found", ticker=ticker)
    return result
