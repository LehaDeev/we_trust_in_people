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


_TICKERS_PER_PAGE = 15


def ticker_select(tickers: list[str], page: int = 0) -> InlineKeyboardMarkup:
    """
    Выбор тикера для запроса сигнала с пагинацией.

    Аргументы:
        tickers: список тикеров (например, ["SBER", "GAZP", ...]).
        page: номер страницы (0-based).
    """
    builder = InlineKeyboardBuilder()
    total_pages = max(1, (len(tickers) + _TICKERS_PER_PAGE - 1) // _TICKERS_PER_PAGE)
    page = max(0, min(page, total_pages - 1))

    start = page * _TICKERS_PER_PAGE
    page_tickers = tickers[start : start + _TICKERS_PER_PAGE]

    # Кнопки тикеров по 3 в ряд
    buttons = [
        InlineKeyboardButton(text=ticker, callback_data=f"signal:{ticker}")
        for ticker in page_tickers
    ]
    builder.add(*buttons)
    builder.adjust(5)

    # Навигация по страницам (если больше одной)
    if total_pages > 1:
        nav_buttons: list[InlineKeyboardButton] = []
        if page > 0:
            nav_buttons.append(
                InlineKeyboardButton(text="◀ Пред.", callback_data=f"signals_page:{page - 1}")
            )
        nav_buttons.append(
            InlineKeyboardButton(text=f"{page + 1}/{total_pages}", callback_data="noop")
        )
        if page < total_pages - 1:
            nav_buttons.append(
                InlineKeyboardButton(text="След. ▶", callback_data=f"signals_page:{page + 1}")
            )
        builder.row(*nav_buttons)

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


def portfolio_menu() -> InlineKeyboardMarkup:
    """
    Меню портфеля: кнопки перехода в категории активов.
    """
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="📈 Акции", callback_data="portfolio:shares"),
        InlineKeyboardButton(text="📄 Облигации", callback_data="portfolio:bonds"),
    )
    builder.row(
        InlineKeyboardButton(text="🏦 ETF", callback_data="portfolio:etf"),
        InlineKeyboardButton(text="💱 Валюта", callback_data="portfolio:currencies"),
    )
    builder.row(InlineKeyboardButton(text="◀ Главное меню", callback_data="menu:main"))
    return builder.as_markup()


def back_to_portfolio() -> InlineKeyboardMarkup:
    """Кнопка возврата в сводку портфеля."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="◀ Портфель", callback_data="menu:portfolio"),
    )
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
        InlineKeyboardButton(text="📊 Статистика", callback_data="trading:stats"),
        InlineKeyboardButton(text="ℹ️ Параметры", callback_data="trading:status"),
    )
    builder.row(
        InlineKeyboardButton(text="🧠 ML модели", callback_data="trading:ml_status"),
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
    Кнопка ведёт на превью-расчёт перед подтверждением.

    Аргументы:
        positions: список кортежей (trade_id, ticker, entry_price_str).
    """
    builder = InlineKeyboardBuilder()
    for trade_id, ticker, price_str in positions:
        builder.row(
            InlineKeyboardButton(
                text=f"{ticker}  {price_str} ₽",
                # preview — сначала показываем расчёт P&L, потом confirm
                callback_data=f"trading:sell:preview:{trade_id}",
            )
        )
    builder.row(InlineKeyboardButton(text="◀ Назад", callback_data="menu:trading"))
    return builder.as_markup()


def confirm_buy(ticker: str) -> InlineKeyboardMarkup:
    """
    Клавиатура подтверждения покупки после просмотра превью.

    Аргументы:
        ticker: тикер инструмента для покупки.
    """
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(
            text="✅ Подтвердить покупку",
            callback_data=f"trading:buy:confirm:{ticker}",
        )
    )
    builder.row(
        InlineKeyboardButton(text="◀ Назад к тикерам", callback_data="trading:buy"),
    )
    return builder.as_markup()


def confirm_sell(trade_id: int) -> InlineKeyboardMarkup:
    """
    Клавиатура подтверждения продажи после просмотра расчёта P&L.

    Аргументы:
        trade_id: ID сделки для продажи.
    """
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(
            text="✅ Подтвердить продажу",
            callback_data=f"trading:sell:confirm:{trade_id}",
        )
    )
    builder.row(
        InlineKeyboardButton(text="◀ Назад к позициям", callback_data="trading:sell"),
    )
    return builder.as_markup()


def back_to_trading() -> InlineKeyboardMarkup:
    """Кнопка возврата в меню торговли."""
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="◀ Назад", callback_data="menu:trading"),
    )
    return builder.as_markup()
