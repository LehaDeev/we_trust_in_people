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
from trading.retrain_scheduler import RetrainScheduler
from trading.scheduler import TradingScheduler
from utils.logger import logger
from utils.redis_cache import init_redis


async def main() -> None:
    """Инициализировать зависимости и запустить планировщики торговли и дообучения."""
    logger.info("Инициализация автоторговли...")

    await init_db()
    await init_redis()

    trading_task = asyncio.create_task(TradingScheduler().run())
    retrain_task = asyncio.create_task(RetrainScheduler().run())

    try:
        await asyncio.gather(trading_task, retrain_task)
    except KeyboardInterrupt:
        logger.info("Автоторговля остановлена пользователем")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
