"""
Хендлер портфеля: текущие позиции и P&L из Tinkoff Invest API.
"""
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery

from bot.keyboards import back_to_main
from tinkoff.portfolio import get_portfolio_summary
from utils.logger import logger

router = Router(name="portfolio")


def _fmt(value: Decimal | None) -> str:
    """Форматировать Decimal в строку с двумя знаками или '-' если None."""
    if value is None:
        return "-"
    return f"{value:,.2f}"


def _format_portfolio(summary: dict) -> str:
    """
    Форматировать сводку портфеля в читаемый текст.

    Пример:
        💼 Портфель

        Акции:    150 000.00 ₽
        Облигации: 0.00 ₽
        ETF:       0.00 ₽
        Валюта:   5 000.00 ₽

        📌 Позиции:
        SBER  10 шт.  325.50 ₽  P&L: +1 250.00
        GAZP   5 шт.  185.20 ₽  P&L: -300.00
    """
    lines = [
        "💼 <b>Портфель</b>",
        "",
        f"Акции:      <b>{_fmt(summary.get('total_shares'))} ₽</b>",
        f"Облигации:  <b>{_fmt(summary.get('total_bonds'))} ₽</b>",
        f"ETF:        <b>{_fmt(summary.get('total_etf'))} ₽</b>",
        f"Валюта:     <b>{_fmt(summary.get('total_currencies'))} ₽</b>",
    ]

    positions: list[dict] = summary.get("positions", [])
    if positions:
        lines.append("")
        lines.append("📌 <b>Позиции:</b>")
        for pos in positions:
            figi = pos.get("figi", "?")
            qty = pos.get("quantity", 0)
            price = _fmt(pos.get("current_price"))
            pnl = pos.get("expected_yield", Decimal(0))
            pnl_sign = "+" if pnl and pnl >= 0 else ""
            lines.append(
                f"<code>{figi[:12]:<12}</code> {qty} шт.  "
                f"{price} ₽  P&L: {pnl_sign}{_fmt(pnl)}"
            )
    else:
        lines.append("")
        lines.append("Позиций нет.")

    return "\n".join(lines)


@router.callback_query(lambda c: c.data == "menu:portfolio")
async def cb_portfolio(callback: CallbackQuery) -> None:
    """Показать текущий портфель."""
    await callback.answer("⏳ Загружаю портфель...")
    try:
        summary = await get_portfolio_summary()
        text = _format_portfolio(summary)
    except Exception as e:
        logger.error("Portfolio fetch error", error=str(e))
        text = "❌ Не удалось получить данные портфеля.\nПроверьте подключение к Tinkoff API."

    await callback.message.edit_text(
        text, reply_markup=back_to_main(), parse_mode="HTML"
    )
