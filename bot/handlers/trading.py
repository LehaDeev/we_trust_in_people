"""
Handlers раздела торговли: меню и переключение режима авто/ручной.

Детальные обработчики вынесены в подмодули:
    - trading_buy.py   — ручная покупка (выбор тикера → превью → подтверждение)
    - trading_sell.py  — ручная продажа (список → P&L → подтверждение)
    - trading_info.py  — позиции, история, статистика, параметры
"""
from aiogram import Router
from aiogram.types import CallbackQuery

from bot.handlers.trading_buy import router as buy_router
from bot.handlers.trading_info import router as info_router
from bot.handlers.trading_sell import router as sell_router
from bot.keyboards import trading_menu
from tinkoff.portfolio import get_rub_balance
from trading import state

router = Router(name="trading")
router.include_router(buy_router)
router.include_router(sell_router)
router.include_router(info_router)


def _mode_text(is_auto: bool) -> str:
    """Строка статуса режима торговли."""
    if is_auto:
        return "🤖 <b>Режим: Авто</b>\nScheduler торгует по ML-сигналам автоматически."
    return "🖐 <b>Режим: Ручной</b>\nScheduler приостановлен. Используй кнопки Купить / Продать."


async def _show_trading_menu(callback: CallbackQuery) -> None:
    """Вспомогательная функция: отрисовать меню торговли с балансом."""
    is_auto = state.is_auto()
    try:
        balance = await get_rub_balance()
        balance_str = f"💵 Свободно: <b>{balance:.2f} ₽</b>"
    except Exception:
        balance_str = "💵 Свободно: <i>н/д</i>"
    text = f"🤖 <b>Торговля</b>\n\n{_mode_text(is_auto)}\n\n{balance_str}"
    await callback.message.edit_text(
        text, reply_markup=trading_menu(is_auto), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:trading")
async def cb_trading_menu(callback: CallbackQuery) -> None:
    """Показать меню раздела торговли."""
    await _show_trading_menu(callback)


@router.callback_query(lambda c: c.data == "trading:toggle")
async def cb_toggle(callback: CallbackQuery) -> None:
    """Переключить режим авто / ручной."""
    new_value = state.toggle()
    mode = "авто" if new_value else "ручной"
    await callback.answer(f"Режим переключён: {mode}")
    await _show_trading_menu(callback)
