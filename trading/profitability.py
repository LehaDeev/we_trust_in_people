"""
Расчёт рентабельности сделок с учётом комиссий брокера и НДФЛ.

Формулы:
    entry_total  = entry_price × lots × lot_size
    exit_total   = exit_price  × lots × lot_size
    gross_pnl    = exit_total - entry_total
    buy_comm     = entry_total × commission_pct
    sell_comm    = exit_total  × commission_pct
    profit_pre_tax = gross_pnl - buy_comm - sell_comm
    tax          = max(0, profit_pre_tax × tax_pct)
    net_pnl      = profit_pre_tax - tax

Точка безубыточности (breakeven): минимальный рост цены при котором net_pnl ≥ 0.
    x * (1 - c) = 2c  →  x = 2c / (1 - c)
    (налог не смещает точку безубыточности, только уменьшает прибыль выше неё)
"""
from dataclasses import dataclass
from decimal import Decimal

from config.settings import trading_settings


@dataclass
class PnLBreakdown:
    """Детальный расчёт P&L сделки с учётом комиссий и налога."""

    entry_total: Decimal       # полная стоимость покупки (цена × лоты × размер лота)
    exit_total: Decimal        # полная сумма продажи
    gross_pnl: Decimal         # прибыль без учёта издержек
    buy_commission: Decimal    # комиссия при покупке
    sell_commission: Decimal   # комиссия при продаже
    tax: Decimal               # НДФЛ (только если gross после комиссий > 0)
    net_pnl: Decimal           # чистая прибыль (после всех издержек)
    is_profitable: bool        # True если net_pnl > 0


def calculate_pnl(
    entry_price: Decimal,
    exit_price: Decimal,
    lots: int,
    lot_size: int = 1,
) -> PnLBreakdown:
    """
    Рассчитать детальный P&L сделки.

    Аргументы:
        entry_price: цена покупки (за 1 бумагу)
        exit_price:  цена продажи (за 1 бумагу)
        lots:        количество лотов
        lot_size:    количество бумаг в 1 лоте

    Возвращает:
        PnLBreakdown с детализацией всех составляющих.
    """
    commission_pct = Decimal(str(trading_settings.broker_commission_pct))
    tax_pct = Decimal(str(trading_settings.tax_pct))

    qty = Decimal(lots * lot_size)
    entry_total = entry_price * qty
    exit_total = exit_price * qty
    gross_pnl = exit_total - entry_total

    buy_commission = entry_total * commission_pct
    sell_commission = exit_total * commission_pct
    profit_pre_tax = gross_pnl - buy_commission - sell_commission

    # Налог начисляется только на положительный результат после комиссий
    tax = (
        max(Decimal("0"), profit_pre_tax * tax_pct)
        if profit_pre_tax > Decimal("0")
        else Decimal("0")
    )
    net_pnl = profit_pre_tax - tax

    two = Decimal("0.01")  # округление до копеек
    return PnLBreakdown(
        entry_total=entry_total.quantize(two),
        exit_total=exit_total.quantize(two),
        gross_pnl=gross_pnl.quantize(two),
        buy_commission=buy_commission.quantize(two),
        sell_commission=sell_commission.quantize(two),
        tax=tax.quantize(two),
        net_pnl=net_pnl.quantize(two),
        is_profitable=net_pnl > Decimal("0"),
    )


def breakeven_pct() -> Decimal:
    """
    Минимальный процент роста цены для безубыточной продажи.

    Формула: x = 2c / (1 - c), где c — комиссия брокера.
    При этом уровне роста чистая прибыль = 0 (комиссии покрываются).
    Значение TP должно быть выше этой отметки + желаемая доходность.

    Возвращает:
        Decimal — доля от цены входа (0.006 = 0.6%).
    """
    c = Decimal(str(trading_settings.broker_commission_pct))
    return (2 * c / (1 - c)).quantize(Decimal("0.0001"))


def format_pnl_breakdown(b: PnLBreakdown) -> str:
    """
    Отформатировать PnLBreakdown в читаемый HTML-текст для Telegram.

    Аргументы:
        b: рассчитанный PnLBreakdown

    Возвращает:
        Строка HTML с детализацией.
    """
    sign = "+" if b.gross_pnl >= 0 else ""
    net_sign = "+" if b.net_pnl >= 0 else ""
    icon = "🟢" if b.is_profitable else "🔴"

    return (
        f"📊 <b>Расчёт сделки</b>\n"
        f"Покупка:    <b>{b.entry_total:.2f} ₽</b>\n"
        f"Продажа:    <b>{b.exit_total:.2f} ₽</b>\n"
        f"Gross P&L:  <b>{sign}{b.gross_pnl:.2f} ₽</b>\n"
        f"Комиссии:   <b>−{b.buy_commission + b.sell_commission:.2f} ₽</b>\n"
        f"НДФЛ:       <b>−{b.tax:.2f} ₽</b>\n"
        f"─────────────────────\n"
        f"{icon} Чистая P&L: <b>{net_sign}{b.net_pnl:.2f} ₽</b>"
    )
