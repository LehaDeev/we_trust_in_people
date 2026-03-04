"""
Точка входа Telegram-бота.

Инициализирует aiogram Dispatcher, подключает роутеры и запускает polling.
TradingScheduler запускается как фоновая asyncio-задача рядом с polling.

Запуск:
    python -m scripts.run_bot
"""
import asyncio

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers import portfolio, signals, start, trading
from config.settings import telegram_settings
from db.database import close_db, init_db
from trading.notifier import set_bot
from trading.retrain_scheduler import RetrainScheduler
from trading.scheduler import TradingScheduler
from utils.logger import logger
from utils.redis_cache import close_redis, init_redis


async def main() -> None:
    """Инициализировать БД, Redis, запустить бота и планировщик торговли."""
    logger.info("Запуск Telegram-бота...")

    await init_db()
    await init_redis()

    bot = Bot(
        token=telegram_settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )

    # Передаём бот в notifier, чтобы уведомления о сделках шли через него же
    set_bot(bot)

    dp = Dispatcher()
    dp.include_router(start.router)
    dp.include_router(signals.router)
    dp.include_router(portfolio.router)
    dp.include_router(trading.router)

    # Запускаем планировщики как фоновые задачи
    scheduler_task = asyncio.create_task(TradingScheduler().run())
    retrain_task = asyncio.create_task(RetrainScheduler().run())

    logger.info("Бот запущен, начинаю polling...")
    try:
        await dp.start_polling(bot)
    finally:
        for task in (scheduler_task, retrain_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await close_redis()
        await close_db()
        await bot.session.close()
        logger.info("Бот остановлен.")
