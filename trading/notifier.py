"""
Telegram-уведомления о сделках автоторговли.

Использует отдельный экземпляр Bot только для отправки сообщений —
без Dispatcher и полноценного webhook-цикла.

При пустом TRADING_CHAT_ID уведомления логируются как WARNING и пропускаются.
"""
from decimal import Decimal

from aiogram import Bot

from config.settings import telegram_settings, trading_settings
from utils.logger import logger


async def _send(text: str) -> None:
    """
    Отправить сообщение в чат уведомлений.

    Аргументы:
        text: текст сообщения (HTML-разметка поддерживается)
    """
    chat_id = trading_settings.notification_chat_id
    if not chat_id:
        logger.warning("TRADING_CHAT_ID не задан — уведомление пропущено", text=text[:80])
        return

    try:
        bot = Bot(token=telegram_settings.bot_token)
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
        await bot.session.close()
        logger.debug("Уведомление отправлено", chat_id=chat_id)
    except Exception as e:
        logger.error("Ошибка отправки уведомления", error=str(e), text=text[:80])


async def notify_open(
    ticker: str,
    price: Decimal,
    lots: int,
    stop_loss: Decimal,
    take_profit: Decimal,
) -> None:
    """
    Уведомить об открытии позиции.

    Аргументы:
        ticker:      тикер инструмента (например "SBER")
        price:       цена входа
        lots:        количество купленных лотов
        stop_loss:   уровень стоп-лосса
        take_profit: уровень тейк-профита
    """
    text = (
        f"🟢 <b>Открыта позиция {ticker}</b>\n"
        f"Куплено: {lots} лот(ов) по <b>{price:.2f} ₽</b>\n"
        f"Стоп-лосс: {stop_loss:.2f} ₽\n"
        f"Тейк-профит: {take_profit:.2f} ₽"
    )
    await _send(text)


async def notify_close(
    ticker: str,
    entry_price: Decimal,
    exit_price: Decimal,
    reason: str,
    pnl: Decimal,
) -> None:
    """
    Уведомить о закрытии позиции.

    Аргументы:
        ticker:      тикер инструмента
        entry_price: цена входа
        exit_price:  цена выхода
        reason:      причина закрытия ("SELL_SIGNAL" | "STOP_LOSS" | "TAKE_PROFIT")
        pnl:         финансовый результат сделки в рублях
    """
    emoji_map = {
        "STOP_LOSS": "🔴",
        "TAKE_PROFIT": "✅",
        "SELL_SIGNAL": "🔵",
    }
    emoji = emoji_map.get(reason, "⚪")

    reason_label = {
        "STOP_LOSS": "Стоп-лосс",
        "TAKE_PROFIT": "Тейк-профит",
        "SELL_SIGNAL": "Сигнал SELL",
    }.get(reason, reason)

    pnl_sign = "+" if pnl >= 0 else ""
    text = (
        f"{emoji} <b>Закрыта позиция {ticker}</b>\n"
        f"Причина: {reason_label}\n"
        f"Вход: {entry_price:.2f} ₽ → Выход: {exit_price:.2f} ₽\n"
        f"PnL: <b>{pnl_sign}{pnl:.2f} ₽</b>"
    )
    await _send(text)
