"""
Точка входа для запуска автоматической торговли.

Использование:
    python -m scripts.run_trading

Перед запуском убедись:
    1. TRADING_ENABLED=true в .env (по умолчанию false — безопасный режим)
    2. TRADING_CHAT_ID=<твой chat_id> для получения уведомлений в Telegram
    3. Проверены все TRADING_* параметры в .env
    4. Запущен сборщик рыночных данных (run_collector.py)
    5. Обучена модель (train.py)
"""
import asyncio
import sys

from db.database import init_db
from trading.scheduler import TradingScheduler
from utils.logger import logger
from utils.redis_cache import init_redis


async def main() -> None:
    """Инициализировать зависимости и запустить планировщик торговли."""
    logger.info("Инициализация автоторговли...")

    # Инициализируем БД (создаём таблицы если не существуют)
    await init_db()

    # Инициализируем Redis (graceful: если недоступен — работаем без кеша)
    await init_redis()

    scheduler = TradingScheduler()
    try:
        await scheduler.run()
    except KeyboardInterrupt:
        logger.info("Автоторговля остановлена пользователем")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
