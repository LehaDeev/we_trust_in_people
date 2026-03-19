"""
Хендлер /start и навигация главного меню.
"""
from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, Message

from bot.keyboards import back_to_main, main_menu, ticker_select
from config.settings import data_settings
from trading.notifier import set_menu_message

router = Router(name="start")

_WELCOME = (
    "👋 <b>We Trust in People</b>\n\n"
    "Торговый бот с ML-сигналами для MOEX.\n"
    "Выберите действие:"
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Ответить на /start главным меню."""
    msg = await message.answer(_WELCOME, reply_markup=main_menu(), parse_mode="HTML")
    await set_menu_message(msg.message_id)


@router.callback_query(lambda c: c.data == "menu:main")
async def cb_main_menu(callback: CallbackQuery) -> None:
    """Вернуться в главное меню."""
    await callback.message.edit_text(
        _WELCOME, reply_markup=main_menu(), parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:signals")
async def cb_signals_menu(callback: CallbackQuery) -> None:
    """Показать список тикеров для выбора сигнала (страница 1)."""
    tickers = data_settings.tickers
    await callback.message.edit_text(
        "📊 <b>Сигналы</b>\n\nВыберите тикер:",
        reply_markup=ticker_select(tickers, page=0),
        parse_mode="HTML",
    )
    await callback.answer()


@router.callback_query(lambda c: c.data and c.data.startswith("signals_page:"))
async def cb_signals_page(callback: CallbackQuery) -> None:
    """Переключить страницу списка тикеров."""
    page = int(callback.data.split(":", 1)[1])
    tickers = data_settings.tickers
    await callback.message.edit_reply_markup(reply_markup=ticker_select(tickers, page=page))
    await callback.answer()


@router.callback_query(lambda c: c.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    """Заглушка для неактивных кнопок (счётчик страниц)."""
    await callback.answer()
