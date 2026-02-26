"""
Entry point for ML model training.

Запуск:
    python -m scripts.train_model

Что делает:
    1. Загружает свечи из PostgreSQL для всех тикеров
    2. Вычисляет технические индикаторы (TA-Lib)
    3. Генерирует метки BUY/SELL/HOLD
    4. Обучает LightGBM классификатор
    5. Сохраняет веса в ml/weights/
"""
import asyncio

from db.database import close_db, init_db
from ml.train import train_model
from utils.logger import logger


async def main() -> None:
    """Инициализировать БД, обучить модель, закрыть соединение."""
    logger.info("Starting model training pipeline")

    await init_db()

    try:
        model_path = await train_model()
        logger.info("Training complete", model_path=str(model_path))
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
