"""
Telegram-уведомления о сделках автоторговли и ночном дообучении.

Использует Bot-синглтон, установленный при старте в bot/main.py через set_bot().
При пустом TRADING_CHAT_ID уведомления логируются и пропускаются.
"""
from decimal import Decimal
from pathlib import Path

from aiogram import Bot

from config.settings import trading_settings
from utils.logger import logger
from utils.redis_cache import get_redis

# Синглтон бота — устанавливается из bot/main.py при старте
_bot: Bot | None = None
# ID последнего сообщения с меню (in-memory кеш, персистируется в Redis)
_menu_msg_id: int | None = None
_MENU_MSG_REDIS_KEY = "menu_msg_id"


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


async def set_menu_message(msg_id: int) -> None:
    """
    Сохранить ID текущего сообщения с меню (в памяти и в Redis).

    Вызывать из handlers при отображении главного меню.
    Redis обеспечивает сохранение ID между перезапусками бота.

    Аргументы:
        msg_id: message_id сообщения с inline-меню.
    """
    global _menu_msg_id
    _menu_msg_id = msg_id
    redis = await get_redis()
    if redis is not None:
        await redis.set(_MENU_MSG_REDIS_KEY, str(msg_id))


async def _get_menu_msg_id() -> int | None:
    """Получить ID меню: сначала из памяти, при None — из Redis."""
    if _menu_msg_id is not None:
        return _menu_msg_id
    redis = await get_redis()
    if redis is None:
        return None
    val = await redis.get(_MENU_MSG_REDIS_KEY)
    if val is None:
        return None
    global _menu_msg_id
    _menu_msg_id = int(val)
    return _menu_msg_id


async def _refresh_menu() -> None:
    """Удалить старое меню и отправить новое снизу (после уведомления).

    ID меню читается из Redis — работает корректно после перезапуска бота.
    """
    from bot.keyboards import main_menu  # импорт здесь во избежание circular import

    menu_id = await _get_menu_msg_id()
    if menu_id is None or _bot is None:
        return

    chat_id = trading_settings.notification_chat_id
    if not chat_id:
        return

    # Удаляем старое меню
    try:
        await _bot.delete_message(chat_id=chat_id, message_id=menu_id)
    except Exception:
        pass  # сообщение уже удалено или недоступно

    # Отправляем новое меню снизу
    try:
        msg = await _bot.send_message(
            chat_id=chat_id,
            text="👋 <b>We Trust in People</b>\n\nВыберите действие:",
            reply_markup=main_menu(),
            parse_mode="HTML",
        )
        await set_menu_message(msg.message_id)
    except Exception as e:
        logger.error("Ошибка обновления меню после уведомления", error=str(e))


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
        await _refresh_menu()
    except Exception as e:
        logger.error("Ошибка отправки уведомления", error=str(e), text=text[:80])


async def notify_open(
    ticker: str,
    price: Decimal,
    lots: int,
    lot_size: int,
    stop_loss: Decimal,
    take_profit: Decimal,
) -> None:
    """
    Уведомить об открытии позиции.

    Аргументы:
        ticker:      тикер инструмента (например "SBER")
        price:       цена входа за 1 бумагу
        lots:        количество купленных лотов
        lot_size:    количество бумаг в 1 лоте
        stop_loss:   уровень стоп-лосса
        take_profit: уровень тейк-профита
    """
    total_qty = lots * lot_size
    total_cost = price * total_qty
    text = (
        f"🟢 <b>Открыта позиция {ticker}</b>\n"
        f"Куплено: {lots} лот × {lot_size} шт = {total_qty} бумаг\n"
        f"Цена: <b>{price:.2f} ₽</b>  Сумма: <b>{total_cost:.2f} ₽</b>\n"
        f"Стоп-лосс:   {stop_loss:.2f} ₽\n"
        f"Тейк-профит: {take_profit:.2f} ₽"
    )
    await _send(text)


async def notify_insufficient_balance(
    ticker: str,
    needed: Decimal,
    available: Decimal,
) -> None:
    """
    Уведомить о пропуске BUY-сигнала из-за недостатка средств.

    Аргументы:
        ticker:    тикер инструмента
        needed:    требуемая сумма для покупки
        available: текущий свободный баланс в рублях
    """
    text = (
        f"⚠️ <b>Недостаточно средств: {ticker}</b>\n"
        f"Нужно:    <b>{needed:.2f} ₽</b>\n"
        f"Доступно: <b>{available:.2f} ₽</b>\n"
        f"<i>BUY-сигнал пропущен</i>"
    )
    await _send(text)


async def notify_close(
    ticker: str,
    entry_price: Decimal,
    exit_price: Decimal,
    reason: str,
    net_pnl: Decimal,
    gross_pnl: Decimal,
    commission: Decimal,
    tax: Decimal,
) -> None:
    """
    Уведомить о закрытии позиции с детальным расчётом P&L.

    Аргументы:
        ticker:      тикер инструмента
        entry_price: цена входа за 1 бумагу
        exit_price:  цена выхода за 1 бумагу
        reason:      причина закрытия ("SELL_SIGNAL" | "STOP_LOSS" | "TAKE_PROFIT" | "MANUAL")
        net_pnl:     чистая прибыль после комиссий и НДФЛ
        gross_pnl:   прибыль до комиссий и налога
        commission:  суммарная комиссия (покупка + продажа)
        tax:         удержанный НДФЛ
    """
    emoji_map = {
        "STOP_LOSS": "🔴",
        "TAKE_PROFIT": "✅",
        "SELL_SIGNAL": "🔵",
        "MANUAL": "🖐",
    }
    reason_label = {
        "STOP_LOSS": "Стоп-лосс",
        "TAKE_PROFIT": "Тейк-профит",
        "SELL_SIGNAL": "Сигнал SELL",
        "MANUAL": "Ручная продажа",
    }

    emoji = emoji_map.get(reason, "⚪")
    label = reason_label.get(reason, reason)
    gross_sign = "+" if gross_pnl >= 0 else ""
    net_sign = "+" if net_pnl >= 0 else ""

    tax_line = f"НДФЛ:      −{tax:.2f} ₽\n" if tax > 0 else ""
    text = (
        f"{emoji} <b>Закрыта позиция {ticker}</b>\n"
        f"Причина: {label}\n"
        f"Вход: {entry_price:.2f} ₽ → Выход: {exit_price:.2f} ₽\n"
        f"─────────────────\n"
        f"Gross P&L:  {gross_sign}{gross_pnl:.2f} ₽\n"
        f"Комиссии:  −{commission:.2f} ₽\n"
        f"{tax_line}"
        f"─────────────────\n"
        f"<b>Чистый P&L: {net_sign}{net_pnl:.2f} ₽</b>"
    )
    await _send(text)


async def notify_retrain_done(
    results: dict[str, Path],
    failed: list[str],
) -> None:
    """
    Уведомить об успешном завершении ночного дообучения.

    Аргументы:
        results: словарь {ticker: path} успешно обученных тикеров
        failed:  список тикеров с ошибками
    """
    total = len(results) + len(failed)
    lines = [f"🤖 <b>Ночное дообучение завершено</b>"]
    lines.append(f"Обучено: <b>{len(results)}/{total}</b> тикеров")
    for ticker in results:
        lines.append(f"  ✓ {ticker}")
    for ticker in failed:
        lines.append(f"  ✗ {ticker} — ошибка")
    await _send("\n".join(lines))


async def notify_retrain_error(error: str) -> None:
    """
    Уведомить об ошибке ночного дообучения.

    Аргументы:
        error: текст ошибки
    """
    text = (
        f"🚨 <b>Ошибка ночного дообучения</b>\n"
        f"<code>{error[:400]}</code>"
    )
    await _send(text)
