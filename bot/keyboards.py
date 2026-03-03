"""
Inline-клавиатуры Telegram-бота.

Все клавиатуры — только InlineKeyboardMarkup (ReplyKeyboard запрещён).
"""
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def main_menu() -> InlineKeyboardMarkup:
    """Главное меню: сигналы, портфель, автоторговля."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="📊 Сигналы", callback_data="menu:signals"),
        InlineKeyboardButton(text="💼 Портфель", callback_data="menu:portfolio"),
    )
    builder.row(
        InlineKeyboardButton(text="🤖 Торговля", callback_data="menu:trading"),
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


def trading_menu(is_auto: bool) -> InlineKeyboardMarkup:
    """
    Меню раздела автоторговли.

    Аргументы:
        is_auto: True если сейчас включён авто-режим.
    """
    builder = InlineKeyboardBuilder()
    # Кнопка переключения режима
    if is_auto:
        toggle_btn = InlineKeyboardButton(
            text="⏸ Переключить на ручной", callback_data="trading:toggle"
        )
    else:
        toggle_btn = InlineKeyboardButton(
            text="▶ Переключить на авто", callback_data="trading:toggle"
        )
    builder.row(toggle_btn)
    # В ручном режиме — кнопки ручной торговли
    if not is_auto:
        builder.row(
            InlineKeyboardButton(text="🛒 Купить", callback_data="trading:buy"),
            InlineKeyboardButton(text="💸 Продать", callback_data="trading:sell"),
        )
    builder.row(
        InlineKeyboardButton(text="📋 Позиции", callback_data="trading:positions"),
        InlineKeyboardButton(text="📜 История", callback_data="trading:history"),
    )
    builder.row(
        InlineKeyboardButton(text="ℹ️ Параметры", callback_data="trading:status"),
    )
    builder.row(
        InlineKeyboardButton(text="◀ Главное меню", callback_data="menu:main"),
    )
    return builder.as_markup()


def manual_buy_tickers(tickers: list[str]) -> InlineKeyboardMarkup:
    """
    Список тикеров для ручной покупки.

    Аргументы:
        tickers: список тикеров из data_settings.
    """
    builder = InlineKeyboardBuilder()
    buttons = [
        InlineKeyboardButton(text=ticker, callback_data=f"trading:buy:{ticker}")
        for ticker in tickers
    ]
    builder.add(*buttons)
    builder.adjust(3)
    builder.row(InlineKeyboardButton(text="◀ Назад", callback_data="menu:trading"))
    return builder.as_markup()


def manual_sell_positions(
    positions: list[tuple[int, str, str]],
) -> InlineKeyboardMarkup:
    """
    Список открытых позиций для ручной продажи.

    Аргументы:
        positions: список кортежей (trade_id, ticker, entry_price_str).
    """
    builder = InlineKeyboardBuilder()
    for trade_id, ticker, price_str in positions:
        builder.row(
            InlineKeyboardButton(
                text=f"{ticker}  {price_str} ₽",
                callback_data=f"trading:sell:{trade_id}",
            )
        )
    builder.row(InlineKeyboardButton(text="◀ Назад", callback_data="menu:trading"))
    return builder.as_markup()


def back_to_trading() -> InlineKeyboardMarkup:
    """Кнопка возврата в меню торговли."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="◀ Назад", callback_data="menu:trading"),
    )
    return builder.as_markup()
