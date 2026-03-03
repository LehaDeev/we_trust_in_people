"""
Telegram-уведомления о сделках автоторговли.

Использует Bot-синглтон, установленный при старте в bot/main.py через set_bot().
При пустом TRADING_CHAT_ID уведомления логируются и пропускаются.
"""
from decimal import Decimal

from aiogram import Bot

from config.settings import trading_settings
from utils.logger import logger

# Синглтон бота — устанавливается из bot/main.py при старте
_bot: Bot | None = None


def set_bot(bot: Bot) -> None:
    """
    Зарегистрировать экземпляр бота для отправки уведомлений.

    Вызывать один раз при инициализации в bot/main.py.

    Аргументы:
        bot: запущенный экземпляр aiogram Bot.
    """
    global _bot
    _bot = bot
    logger.debug("Notifier: Bot-синглтон установлен")


async def _send(text: str) -> None:
    """
    Отправить сообщение в чат уведомлений.

    Аргументы:
        text: текст сообщения (поддерживает HTML-разметку).
    """
    chat_id = trading_settings.notification_chat_id
    if not chat_id:
        logger.warning(
            "TRADING_CHAT_ID не задан — уведомление пропущено", text=text[:80]
        )
        return

    if _bot is None:
        logger.warning(
            "Bot-синглтон не инициализирован — уведомление пропущено", text=text[:80]
        )
        return

    try:
        await _bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
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
        f"Стоп-лосс:   {stop_loss:.2f} ₽\n"
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
    reason_label = {
        "STOP_LOSS": "Стоп-лосс",
        "TAKE_PROFIT": "Тейк-профит",
        "SELL_SIGNAL": "Сигнал SELL",
    }

    emoji = emoji_map.get(reason, "⚪")
    label = reason_label.get(reason, reason)
    pnl_sign = "+" if pnl >= 0 else ""

    text = (
        f"{emoji} <b>Закрыта позиция {ticker}</b>\n"
        f"Причина: {label}\n"
        f"Вход: {entry_price:.2f} ₽ → Выход: {exit_price:.2f} ₽\n"
        f"PnL: <b>{pnl_sign}{pnl:.2f} ₽</b>"
    )
    await _send(text)
