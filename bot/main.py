"""
Точка входа Telegram-бота.

Инициализирует aiogram Dispatcher, подключает роутеры и запускает polling.

Запуск:
    python -m scripts.run_bot
"""
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers import portfolio, signals, start
from config.settings import telegram_settings
from db.database import close_db, init_db
from utils.logger import logger
from utils.redis_cache import close_redis, init_redis


async def main() -> None:
    """Инициализировать БД, Redis, запустить бота, закрыть соединения при завершении."""
    logger.info("Запуск Telegram-бота...")

    await init_db()
    await init_redis()

    bot = Bot(
        token=telegram_settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    dp.include_router(start.router)
    dp.include_router(signals.router)
    dp.include_router(portfolio.router)

    logger.info("Бот запущен, начинаю polling...")
    try:
        await dp.start_polling(bot)
    finally:
        await close_redis()
        await close_db()
        await bot.session.close()
        logger.info("Бот остановлен.")
