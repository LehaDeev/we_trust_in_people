"""
Inline-клавиатуры Telegram-бота.

Все клавиатуры — только InlineKeyboardMarkup (ReplyKeyboard запрещён).
"""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def main_menu() -> InlineKeyboardMarkup:
    """Главное меню: сигналы и портфель."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="📊 Сигналы", callback_data="menu:signals"),
        InlineKeyboardButton(text="💼 Портфель", callback_data="menu:portfolio"),
    )
    return builder.as_markup()


def ticker_select(tickers: list[str]) -> InlineKeyboardMarkup:
    """
    Выбор тикера для запроса сигнала.

    Аргументы:
        tickers: список тикеров (например, ["SBER", "GAZP", ...]).
    """
    builder = InlineKeyboardBuilder()
    # Кнопки по 3 в ряд
    buttons = [
        InlineKeyboardButton(text=ticker, callback_data=f"signal:{ticker}")
        for ticker in tickers
    ]
    builder.add(*buttons)
    builder.adjust(3)
    builder.row(InlineKeyboardButton(text="◀ Назад", callback_data="menu:main"))
    return builder.as_markup()


def signal_actions(ticker: str) -> InlineKeyboardMarkup:
    """
    Действия после отображения сигнала для тикера.

    Аргументы:
        ticker: тикер инструмента.
    """
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="🔄 Обновить", callback_data=f"signal_refresh:{ticker}"),
        InlineKeyboardButton(text="◀ К тикерам", callback_data="menu:signals"),
    )
    return builder.as_markup()


def back_to_main() -> InlineKeyboardMarkup:
    """Кнопка возврата в главное меню."""
    builder = InlineKeyboardBuilder()
    builder.row(InlineKeyboardButton(text="◀ Главное меню", callback_data="menu:main"))
    return builder.as_markup()
