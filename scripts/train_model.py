"""
Точка входа для обучения ML-моделей — по одному ансамблю на каждый тикер.

Запуск:
    python -m scripts.train_model               # Optuna только если нет кеша (или ML_FORCE_TUNE=true в .env)
    python -m scripts.train_model --force-tune  # Принудительный повтор Optuna (переопределяет .env)

Приоритет force_tune: CLI-флаг > ML_FORCE_TUNE в .env.

Что делает (для каждого тикера из DATA_TICKERS):
    1. Загружает свежие свечи из PostgreSQL
    2. Вычисляет технические индикаторы (TA-Lib)
    3. Генерирует метки BUY/SELL/HOLD
    4. Подбирает гиперпараметры через Optuna (пропускает если есть кеш):
       - LightGBM     — ML_OPTUNA_TRIALS_LGBM trials
       - XGBoost      — ML_OPTUNA_TRIALS_XGB trials
       - RandomForest — ML_OPTUNA_TRIALS_RF trials
    5. Обучает soft voting ансамбль на данных этого тикера
    6. Сохраняет веса в ml/weights/ensemble_{ticker}_{ML_MODEL_VERSION}.pkl
"""
import argparse
import asyncio
from pathlib import Path

from config.settings import ml_settings
from db.database import close_db, init_db
from ml.train import train_model
from utils.logger import logger


def _parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Обучение ансамбля ML-моделей")
    parser.add_argument(
        "--force-tune",
        action="store_true",
        default=False,
        help="Принудительно запустить Optuna HPO (переопределяет ML_FORCE_TUNE из .env)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Папка с CSV-файлами (для Colab без PostgreSQL). Пример: ml/data",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        default=False,
        help="Пропустить CV-оценку — только финальный фит (быстрее, меньше RAM для ночного переобучения)",
    )
    return parser.parse_args()


async def main(force_tune: bool, data_dir: Path | None, skip_cv: bool = False) -> None:
    """Инициализировать БД (если нет data_dir), обучить ансамбль, закрыть соединение."""
    logger.info(
        "Запуск pipeline обучения ансамбля",
        force_tune=force_tune,
        skip_cv=skip_cv,
        data_dir=str(data_dir) if data_dir else "PostgreSQL",
    )

    if data_dir is None:
        await init_db()

    try:
        results = await train_model(force_tune=force_tune, data_dir=data_dir, skip_cv=skip_cv)
        logger.info(
            "Обучение завершено",
            trained=list(results.keys()),
            paths={t: str(p) for t, p in results.items()},
        )
    finally:
        if data_dir is None:
            await close_db()


if __name__ == "__main__":
    args = _parse_args()
    # CLI-флаг переопределяет значение из .env
    force_tune = args.force_tune or ml_settings.force_tune
    data_dir = Path(args.data_dir) if args.data_dir else None
    asyncio.run(main(force_tune=force_tune, data_dir=data_dir, skip_cv=args.skip_cv))
