"""
Планировщик ночного дообучения ML-моделей.

Каждую ночь в RETRAIN_HOUR:RETRAIN_MINUTE (по RETRAIN_TIMEZONE):
    1. Инкрементально собирает новые свечи из Tinkoff API (только новые — от последней в БД)
    2. Переобучает per-ticker ансамбли на всех накопленных данных
       (Optuna HPO пропускается — используются кешированные best_params_*.json)
    3. Сбрасывает in-memory кеш моделей — следующий predict_signal() загрузит новые веса
    4. Отправляет Telegram-уведомление о результате

Запуск:
    Автоматически как фоновый asyncio-task из bot/main.py и scripts/run_trading.py.
    Управляется через .env: RETRAIN_ENABLED, RETRAIN_HOUR, RETRAIN_MINUTE, RETRAIN_TIMEZONE.
"""
import asyncio
import zoneinfo
from datetime import datetime, timedelta

from config.settings import data_settings, retrain_settings
from ml.predict import clear_model_cache
from ml.train import train_model
from scripts.collect_candles import run_collection
from trading.notifier import notify_retrain_done, notify_retrain_error
from utils.logger import logger


class RetrainScheduler:
    """
    Планировщик ночного дообучения.

    Запускается как фоновый asyncio-task рядом с ботом и TradingScheduler.
    При RETRAIN_ENABLED=false — немедленно завершается без действий.
    """

    async def run(self) -> None:
        """Бесконечный цикл: ждёт запланированного времени → дообучает модели."""
        if not retrain_settings.enabled:
            logger.info("Ночное дообучение отключено (RETRAIN_ENABLED=false)")
            return

        logger.info(
            "Планировщик дообучения запущен",
            hour=retrain_settings.hour,
            minute=retrain_settings.minute,
            timezone=retrain_settings.timezone,
        )

        while True:
            await self._wait_until_next_run()
            await self._retrain()

    async def _wait_until_next_run(self) -> None:
        """Рассчитать время до следующего запуска и переждать его."""
        tz = zoneinfo.ZoneInfo(retrain_settings.timezone)
        now = datetime.now(tz)
        next_run = now.replace(
            hour=retrain_settings.hour,
            minute=retrain_settings.minute,
            second=0,
            microsecond=0,
        )
        if next_run <= now:
            next_run += timedelta(days=1)

        wait_seconds = (next_run - now).total_seconds()
        logger.info(
            "Следующее дообучение запланировано",
            at=next_run.strftime("%Y-%m-%d %H:%M %Z"),
            wait_hours=round(wait_seconds / 3600, 1),
        )
        await asyncio.sleep(wait_seconds)

    async def _retrain(self) -> None:
        """Один цикл: сбор свечей → переобучение → сброс кеша → уведомление."""
        logger.info("Запуск ночного дообучения моделей")
        try:
            # Шаг 1: инкрементальный сбор новых свечей (только новые — от последней в БД)
            logger.info("Шаг 1/3: сбор новых свечей из Tinkoff API")
            await run_collection()

            # Шаг 2: переобучение ансамблей (HPO из кеша — быстро, без Optuna)
            logger.info("Шаг 2/3: переобучение per-ticker ансамблей")
            results = await train_model(force_tune=retrain_settings.force_tune)

            # Шаг 3: сброс in-memory кеша — следующий predict загрузит новые веса
            clear_model_cache()
            logger.info("Шаг 3/3: in-memory кеш моделей сброшен")

            failed = [t for t in data_settings.tickers if t not in results]
            await notify_retrain_done(results, failed)

            logger.info(
                "Ночное дообучение завершено",
                trained=list(results.keys()),
                failed=failed,
            )

        except Exception as e:
            logger.error("Ошибка ночного дообучения", error=str(e))
            await notify_retrain_error(str(e))
