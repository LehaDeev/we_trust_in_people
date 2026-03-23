"""
Модуль расчёта размера позиции (Position Sizing).

Поддерживаемые методы (TRADING_POSITION_SIZING):
    fixed_risk  — Fixed Fractional Risk: лоты масштабируются по формуле
                  lots = floor(balance × risk_pct / (sl_pct × price × lot_size)).
                  Гарантирует постоянный рублёвый риск на сделку независимо
                  от цены тикера и волатильности. Рекомендуется.
    fixed_lots  — Всегда lots_per_ticker лотов из настроек.
                  Обратная совместимость со старым поведением.
"""
from decimal import Decimal

from config.settings import trading_settings


def compute_lots(
    balance: Decimal,
    price: Decimal,
    lot_size: int,
    sl_pct: float,
) -> int:
    """
    Рассчитать число лотов по методу Fixed Fractional Risk (фиксированный % риска от депо).

    При TRADING_POSITION_SIZING='fixed_risk':
        риск_на_сделку   = balance × TRADING_RISK_PCT_PER_TRADE
        стоимость_позиции = риск_на_сделку / sl_pct
        лотов             = floor(стоимость_позиции / (price × lot_size))

    Результат зажат в диапазон [1, TRADING_MAX_LOTS_PER_TRADE].
    Метод гарантирует: при срабатывании SL потеря ≈ risk_pct_per_trade × balance,
    независимо от цены тикера и текущей волатильности.

    При TRADING_POSITION_SIZING='fixed_lots' или sl_pct <= 0 возвращает
    TRADING_LOTS_PER_TICKER (обратная совместимость).

    Аргументы:
        balance:  доступный рублёвый баланс (Decimal)
        price:    текущая цена одной бумаги (Decimal)
        lot_size: количество бумаг в одном лоте
        sl_pct:   стоп-лосс как доля от цены входа (например 0.025 = 2.5%)

    Возвращает:
        целое число лотов >= 1
    """
    ts = trading_settings
    if ts.position_sizing == "fixed_lots" or sl_pct <= 0:
        return ts.lots_per_ticker

    lot_value = price * Decimal(lot_size)
    if lot_value <= 0:
        return 1

    risk_rub = balance * Decimal(str(ts.risk_pct_per_trade))
    position_value = risk_rub / Decimal(str(sl_pct))
    raw_lots = int(position_value / lot_value)
    return max(1, min(raw_lots, ts.max_lots_per_trade))
