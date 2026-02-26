"""
Инференс: генерация сигнала BUY/SELL/HOLD для одного тикера.

Загружает обученную LightGBM модель с диска, получает последние свечи
из базы данных, вычисляет признаки и возвращает словарь с сигналом.

Использование:
    import asyncio
    from ml.predict import predict_signal

    result = asyncio.run(predict_signal("SBER"))
    # {"ticker": "SBER", "signal": "BUY", "confidence": 0.72}
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.dataset import load_ticker_data
from ml.features import compute_features
from ml.labels import LABEL_NAMES
from utils.logger import logger

# Пути по умолчанию (можно переопределить через аргументы)
DEFAULT_WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_MODEL_VERSION = "v1"

# Минимум свечей: 50 прогрев + 200 буфер для надёжных значений индикаторов
MIN_CANDLES = 250


async def predict_signal(
    ticker: str,
    model_version: str = DEFAULT_MODEL_VERSION,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    interval: str = "1h",
) -> dict:
    """
    Сгенерировать сигнал BUY/SELL/HOLD для заданного тикера.

    Аргументы:
        ticker: тикер инструмента (например, "SBER").
        model_version: суффикс версии модели (по умолчанию "v1").
        weights_dir: директория с файлами весов pkl и json.
        interval: интервал свечей для загрузки из БД (по умолчанию "1h").

    Возвращает:
        Словарь с ключами:
            ticker (str): тикер
            signal (str): "BUY", "HOLD" или "SELL"
            confidence (float): вероятность модели для предсказанного класса
            probabilities (dict): {"SELL": p, "HOLD": p, "BUY": p}

    Исключения:
        FileNotFoundError: если веса модели не найдены.
        ValueError: если недостаточно данных свечей.
    """
    model_path = weights_dir / f"lgbm_{model_version}.pkl"
    features_path = weights_dir / f"features_{model_version}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Run: python -m scripts.train_model"
        )

    # Загружаем модель и ожидаемый список признаков
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path) as f:
        feature_columns: list[str] = json.load(f)

    # Загружаем данные свечей
    df = await load_ticker_data(ticker, interval)

    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"Not enough candles for {ticker}: {len(df)} < {MIN_CANDLES} required."
        )

    # Берём только последние свечи для ускорения инференса
    df = df.tail(MIN_CANDLES).reset_index(drop=True)

    # Вычисляем признаки (строки прогрева удаляются внутри)
    feat_df = compute_features(df)

    if feat_df.empty:
        raise ValueError(f"No valid feature rows for {ticker} after indicator warmup.")

    # Берём последнюю строку (самую свежую)
    last_row = feat_df[feature_columns].iloc[[-1]]

    # Предсказание
    proba = model.predict_proba(last_row)[0]  # форма: (3,) — SELL, HOLD, BUY
    predicted_class = int(np.argmax(proba))
    confidence = float(proba[predicted_class])
    signal = LABEL_NAMES[predicted_class]

    result = {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(LABEL_NAMES, proba)
        },
    }

    logger.info(
        "Signal generated",
        ticker=ticker,
        signal=signal,
        confidence=round(confidence, 4),
    )

    return result


async def predict_all(
    tickers: list[str],
    model_version: str = DEFAULT_MODEL_VERSION,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    interval: str = "1h",
) -> list[dict]:
    """
    Сгенерировать сигналы для списка тикеров. При ошибке — пропускает тикер.

    Возвращает:
        Список словарей с сигналами (только для успешно обработанных тикеров).
    """
    results: list[dict] = []
    for ticker in tickers:
        try:
            signal = await predict_signal(ticker, model_version, weights_dir, interval)
            results.append(signal)
        except Exception as e:
            logger.error("Failed to predict signal", ticker=ticker, error=str(e))
    return results
