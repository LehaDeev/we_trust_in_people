"""
Runtime-состояние режима торговли.

Позволяет переключать авто/ручной режим через Telegram-бот без перезапуска.
Инициализируется из TRADING_ENABLED при старте; сбрасывается к нему при перезапуске.

Авто-режим: TradingScheduler торгует по ML-сигналам автоматически.
Ручной режим: Scheduler приостановлен, пользователь торгует кнопками в боте.
"""
from config.settings import trading_settings
from utils.logger import logger

# Текущий режим (инициализируется из .env)
_auto: bool = trading_settings.enabled


def is_auto() -> bool:
    """Возвращает True если включён режим автоматической торговли."""
    return _auto


def set_auto(value: bool) -> None:
    """
    Установить режим торговли.

    Аргументы:
        value: True = авто, False = ручной.
    """
    global _auto
    _auto = value
    mode = "авто" if value else "ручной"
    logger.info("Режим торговли изменён", mode=mode)


def toggle() -> bool:
    """
    Переключить режим торговли.

    Возвращает:
        Новое значение (True = авто, False = ручной).
    """
    global _auto
    _auto = not _auto
    mode = "авто" if _auto else "ручной"
    logger.info("Режим торговли переключён", mode=mode)
    return _auto
