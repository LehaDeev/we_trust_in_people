"""
Точка входа для запуска Telegram-бота.

Запуск:
    python -m scripts.run_bot
"""
import asyncio

from bot.main import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # нормальная остановка через Ctrl+C
