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
import json
import zoneinfo
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import data_settings, ml_settings, retrain_settings
from ml.predict import clear_model_cache
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

    def _already_trained_today(self) -> bool:
        """Проверить по last_results.json — было ли обучение сегодня (в часовом поясе RETRAIN_TIMEZONE).

        Работает корректно даже после перезапуска бота: читает файл с диска.
        """
        results_path = Path("ml/weights/last_results.json")
        if not results_path.exists():
            return False
        try:
            with open(results_path) as f:
                data = json.load(f)
            trained_at_str: str = data.get("trained_at", "")
            if not trained_at_str:
                return False
            trained_at = datetime.fromisoformat(trained_at_str)
            tz = zoneinfo.ZoneInfo(retrain_settings.timezone)
            # trained_at может быть naive (UTC) или aware — приводим к tz
            if trained_at.tzinfo is None:
                from datetime import timezone as _tz
                trained_at = trained_at.replace(tzinfo=_tz.utc)
            trained_local = trained_at.astimezone(tz)
            today_local = datetime.now(tz).date()
            return trained_local.date() == today_local
        except Exception:
            return False

    async def _wait_until_next_run(self) -> None:
        """Рассчитать время до следующего запуска и переждать его.

        Если бот запустился после RETRAIN_HOUR, но в пределах RETRAIN_CATCHUP_HOURS —
        запускает дообучение немедленно (наверстывание пропущенного окна).
        Пропускает catchup если last_results.json показывает что обучение уже было сегодня
        (защита от повторного запуска после перезапуска бота).
        """
        tz = zoneinfo.ZoneInfo(retrain_settings.timezone)
        now = datetime.now(tz)
        scheduled_today = now.replace(
            hour=retrain_settings.hour,
            minute=retrain_settings.minute,
            second=0,
            microsecond=0,
        )

        # Наверстывание: если плановое время уже прошло сегодня,
        # но не более чем RETRAIN_CATCHUP_HOURS часов назад — запустить сразу.
        # Пропускаем если уже обучались сегодня (проверяем по файлу — работает после перезапуска).
        missed_seconds = (now - scheduled_today).total_seconds()
        catchup_window = retrain_settings.catchup_hours * 3600
        if 0 < missed_seconds <= catchup_window and not self._already_trained_today():
            logger.info(
                "Пропущенное дообучение: запуск немедленно (catchup)",
                scheduled_at=scheduled_today.strftime("%Y-%m-%d %H:%M %Z"),
                missed_minutes=round(missed_seconds / 60),
            )
            return

        next_run = scheduled_today if scheduled_today > now else scheduled_today + timedelta(days=1)
        wait_seconds = (next_run - now).total_seconds()
        logger.info(
            "Следующее дообучение запланировано",
            at=next_run.strftime("%Y-%m-%d %H:%M %Z"),
            wait_hours=round(wait_seconds / 3600, 1),
        )
        await asyncio.sleep(wait_seconds)

    async def _retrain(self) -> None:
        """Один цикл: сбор свечей → переобучение (subprocess) → сброс кеша → уведомление.

        Обучение запускается отдельным subprocess-ом чтобы изолировать память:
        после завершения subprocess OS полностью освобождает выделенную ему память,
        не нагружая основной процесс бота.
        """
        logger.info("Запуск ночного дообучения моделей")
        try:
            # Шаг 1: инкрементальный сбор новых свечей (только новые — от последней в БД)
            logger.info("Шаг 1/3: сбор новых свечей из Tinkoff API")
            await run_collection()

            # Шаг 2: переобучение в отдельном subprocess для изоляции памяти
            logger.info("Шаг 2/3: переобучение per-ticker ансамблей (subprocess)")
            # nice -n 19: минимальный приоритет CPU (бот и другие процессы получают CPU первыми)
            # ionice -c 3: idle I/O — обучение читает диск только когда никто не ждёт
            cmd = ["nice", "-n", "19", "ionice", "-c", "3",
                   "python", "-m", "scripts.train_model", "--skip-cv"]
            if retrain_settings.force_tune:
                cmd.append("--force-tune")
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                output = stdout.decode(errors="replace") if stdout else ""
                raise RuntimeError(
                    f"train_model завершился с кодом {proc.returncode}:\n{output[-2000:]}"
                )

            # Шаг 3: читаем результаты из last_results.json (записывается subprocess-ом)
            results_path = Path("ml/weights/last_results.json")
            with open(results_path) as f:
                data = json.load(f)
            f1_scores: dict[str, float] = data["f1_scores"]
            failed: list[str] = data.get("failed", [])
            results = {ticker: Path(f"ml/weights/ensemble_{ticker}_{ml_settings.model_version}.pkl")
                       for ticker in f1_scores}

            # Шаг 4: сброс in-memory кеша — следующий predict загрузит новые веса
            clear_model_cache()
            logger.info("Шаг 4/4: in-memory кеш моделей сброшен")

            await notify_retrain_done(results, failed)
            logger.info(
                "Ночное дообучение завершено",
                trained=list(f1_scores.keys()),
                failed=failed,
            )

        except Exception as e:
            logger.error("Ошибка ночного дообучения", error=str(e))
            await notify_retrain_error(str(e))
