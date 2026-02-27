"""
Хендлер /start и навигация главного меню.
"""
from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from bot.keyboards import back_to_main, main_menu, ticker_select
from config.settings import data_settings

router = Router(name="start")

_WELCOME = (
    "👋 <b>We Trust in People</b>\n\n"
    "Торговый бот с ML-сигналами для MOEX.\n"
    "Выберите действие:"
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Ответить на /start главным меню."""
    await message.answer(_WELCOME, reply_markup=main_menu(), parse_mode="HTML")


@router.callback_query(lambda c: c.data == "menu:main")
async def cb_main_menu(callback: CallbackQuery) -> None:
    """Вернуться в главное меню."""
    await callback.message.edit_text(
        _WELCOME, reply_markup=main_menu(), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:signals")
async def cb_signals_menu(callback: CallbackQuery) -> None:
    """Показать список тикеров для выбора сигнала."""
    tickers = data_settings.tickers
    await callback.message.edit_text(
        "📊 <b>Сигналы</b>\n\nВыберите тикер:",
        reply_markup=ticker_select(tickers),
        parse_mode="HTML",
    )
    await callback.answer()
