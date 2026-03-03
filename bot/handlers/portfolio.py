"""
Хендлер портфеля: сводка по счёту и детализация по категориям активов.

Структура:
    menu:portfolio        → сводная страница с итогами и кнопками категорий
    portfolio:shares      → список акций в портфеле
    portfolio:bonds       → список облигаций
    portfolio:etf         → список ETF
    portfolio:currencies  → список валют
"""
import asyncio
from decimal import Decimal

from aiogram import Router
from aiogram.types import CallbackQuery
from sqlalchemy import select

from bot.keyboards import back_to_portfolio, portfolio_menu
from db.database import get_session
from db.models import Asset
from tinkoff.portfolio import get_portfolio_summary, get_rub_balance
from utils.logger import logger

router = Router(name="portfolio")

# Маппинг callback → (ключевое слово в instrument_type, заголовок, иконка)
_CATEGORIES: dict[str, tuple[str, str, str]] = {
    "shares":     ("share",    "Акции",      "📈"),
    "bonds":      ("bond",     "Облигации",  "📄"),
    "etf":        ("etf",      "ETF",        "🏦"),
    "currencies": ("currency", "Валюта",     "💱"),
}


def _fmt(value: Decimal | None) -> str:
    """Форматировать Decimal в строку с двумя знаками или '-' если None."""
    if value is None:
        return "-"
    return f"{value:,.2f}"


def _filter_positions(positions: list[dict], type_keyword: str) -> list[dict]:
    """
    Отфильтровать позиции по типу инструмента.

    Аргументы:
        positions:    список позиций из get_portfolio_summary()
        type_keyword: ключевое слово ("share", "bond", "etf", "currency")
    """
    return [
        p for p in positions
        if type_keyword in str(p.get("instrument_type", "")).lower()
    ]


def _format_overview(summary: dict, rub_balance: Decimal) -> str:
    """
    Сводная страница портфеля: итоги по категориям и свободный баланс.
    """
    total_shares = summary.get("total_shares")
    total_bonds = summary.get("total_bonds")
    total_etf = summary.get("total_etf")
    total_currencies = summary.get("total_currencies")

    # Tinkoff API: total_currencies уже включает свободный рублёвый остаток,
    # поэтому grand_total = сумма по категориям (без дополнительного rub_balance).
    totals = [
        v for v in (total_shares, total_bonds, total_etf, total_currencies)
        if v is not None
    ]
    grand_total = sum(totals, Decimal("0"))

    lines = [
        "💼 <b>Портфель</b>",
        "",
        f"💰 Всего на счёте:  <b>{_fmt(grand_total)} ₽</b>",
        f"💵 Свободно:        <b>{_fmt(rub_balance)} ₽</b>",
        "",
        "📊 <b>По категориям:</b>",
        f"📈 Акции:       <b>{_fmt(total_shares)} ₽</b>",
        f"📄 Облигации:   <b>{_fmt(total_bonds)} ₽</b>",
        f"🏦 ETF:         <b>{_fmt(total_etf)} ₽</b>",
        f"💱 Валюта:      <b>{_fmt(total_currencies)} ₽</b>",
        "",
        "<i>Нажми на категорию чтобы увидеть список активов</i>",
    ]
    return "\n".join(lines)


def _format_category(
    positions: list[dict],
    figi_to_ticker: dict[str, str],
    title: str,
    icon: str,
) -> str:
    """
    Список позиций по одной категории активов.

    Аргументы:
        positions:      отфильтрованный список позиций
        figi_to_ticker: маппинг figi → тикер из БД
        title:          название категории ("Акции", "Облигации" и т.д.)
        icon:           иконка категории
    """
    if not positions:
        return f"{icon} <b>{title}</b>\n\nПозиций нет."

    lines = [f"{icon} <b>{title} ({len(positions)})</b>", ""]

    for pos in positions:
        figi = pos.get("figi", "")
        # Тикер из БД или первые 10 символов FIGI как fallback
        ticker = figi_to_ticker.get(figi) or figi[:10]

        qty = pos.get("quantity", Decimal("0"))
        avg_price = pos.get("average_buy_price")
        current_price = pos.get("current_price")
        pnl = pos.get("expected_yield", Decimal("0"))

        # Форматируем количество: целое если нет дробной части (акции, ETF),
        # два знака если есть (валюта, облигации)
        try:
            qty_decimal = Decimal(str(qty))
            qty_str = (
                str(int(qty_decimal))
                if qty_decimal == qty_decimal.to_integral_value()
                else f"{qty_decimal:.4f}".rstrip("0")
            )
        except Exception:
            qty_str = str(qty)

        pnl_sign = "+" if pnl and Decimal(str(pnl)) >= 0 else ""
        pnl_icon = "🟢" if pnl and Decimal(str(pnl)) >= 0 else "🔴"

        line_parts = [f"<b>{ticker}</b>  {qty_str} шт"]
        if avg_price is not None:
            line_parts.append(f"ср. цена: {_fmt(Decimal(str(avg_price)))} ₽")
        if current_price is not None:
            line_parts.append(f"сейчас: {_fmt(Decimal(str(current_price)))} ₽")
        if pnl:
            line_parts.append(
                f"{pnl_icon} P&L: {pnl_sign}{_fmt(Decimal(str(pnl)))} ₽"
            )

        lines.append("  ".join(line_parts))

    return "\n".join(lines)


@router.callback_query(lambda c: c.data == "menu:portfolio")
async def cb_portfolio(callback: CallbackQuery) -> None:
    """Показать сводку портфеля с кнопками перехода в категории."""
    await callback.answer("⏳ Загружаю портфель...")
    try:
        summary, rub_balance = await asyncio.gather(
            get_portfolio_summary(),
            get_rub_balance(),
        )
        text = _format_overview(summary, rub_balance)
    except Exception as e:
        logger.error("Portfolio fetch error", error=str(e))
        text = (
            "❌ Не удалось получить данные портфеля.\n"
            "Проверьте подключение к Tinkoff API."
        )

    await callback.message.edit_text(
        text, reply_markup=portfolio_menu(), parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data and c.data.startswith("portfolio:"))
async def cb_portfolio_category(callback: CallbackQuery) -> None:
    """Показать список позиций по выбранной категории (акции/облигации/ETF/валюта)."""
    category = callback.data.split(":", 1)[1]

    if category not in _CATEGORIES:
        await callback.answer("Неизвестная категория")
        return

    type_keyword, title, icon = _CATEGORIES[category]
    await callback.answer(f"⏳ Загружаю {title.lower()}...")

    try:
        summary = await get_portfolio_summary()
        all_positions: list[dict] = summary.get("positions", [])
        positions = _filter_positions(all_positions, type_keyword)

        # Получаем тикеры из БД одним запросом
        figis = [p["figi"] for p in positions if p.get("figi")]
        figi_to_ticker: dict[str, str] = {}
        if figis:
            async with get_session() as session:
                result = await session.execute(
                    select(Asset).where(Asset.figi.in_(figis))
                )
                figi_to_ticker = {
                    a.figi: a.ticker for a in result.scalars().all()
                }

        text = _format_category(positions, figi_to_ticker, title, icon)

    except Exception as e:
        logger.error(
            "Portfolio category fetch error", category=category, error=str(e)
        )
        text = f"❌ Не удалось загрузить {title.lower()}."

    await callback.message.edit_text(
        text, reply_markup=back_to_portfolio(), parse_mode="HTML"
    )
