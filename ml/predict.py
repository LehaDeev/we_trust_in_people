"""
Инференс: генерация сигнала BUY/SELL/HOLD для одного тикера.

Каждый тикер имеет свою модель (ensemble_{ticker}_{version}.pkl),
обученную только на его исторических данных.

Кеширование (двухуровневое):
    1. In-memory (_model_cache): загруженные pickle-модели хранятся в памяти процесса.
       Ключ: "{ticker}_{version}". Исключает повторные disk-reads.
    2. Redis: результат predict_signal() кешируется на REDIS_SIGNAL_TTL секунд.
       При недоступном Redis — graceful degradation (прямой вызов без кеша).

Использование:
    import asyncio
    from ml.predict import predict_signal

    result = asyncio.run(predict_signal("SBER"))
    # {"ticker": "SBER", "signal": "BUY", "confidence": 0.72, "probabilities": {...}}
"""
import json
import pickle
from pathlib import Path

import numpy as np

from config.settings import ml_settings, redis_settings
from ml.dataset import load_ticker_data
from ml.features import compute_features
from ml.labels import LABEL_NAMES
from utils.logger import logger
from utils.redis_cache import get_redis

WEIGHTS_DIR = Path(__file__).parent / "weights"

# In-memory кеш загруженных моделей: "{ticker}_{version}" → (ensemble, feature_columns)
# Каждый тикер хранит свой ансамбль отдельно.
_model_cache: dict[str, tuple] = {}


def _load_model(ticker: str, version: str) -> tuple:
    """
    Загрузить ансамбль тикера и список признаков (из памяти или с диска).

    Аргументы:
        ticker:  тикер инструмента (например, "SBER").
        version: версия модели (например, "v2").

    Возвращает:
        (ensemble, feature_columns): объект VotingClassifier и список имён признаков.

    Исключения:
        FileNotFoundError: если файл весов для тикера не найден.
    """
    cache_key = f"{ticker}_{version}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    ticker_version = f"{ticker}_{version}"
    ensemble_path = WEIGHTS_DIR / f"ensemble_{ticker_version}.pkl"
    features_path = WEIGHTS_DIR / f"features_{ticker_version}.json"

    if not ensemble_path.exists():
        raise FileNotFoundError(
            f"Модель для {ticker} не найдена: {ensemble_path}\n"
            "Запустите: python -m scripts.train_model"
        )

    with open(ensemble_path, "rb") as f:
        ensemble = pickle.load(f)
    with open(features_path) as f:
        feature_columns: list[str] = json.load(f)

    _model_cache[cache_key] = (ensemble, feature_columns)
    logger.info("Модель загружена в память", ticker=ticker, version=version)
    return ensemble, feature_columns


async def predict_signal(
    ticker: str,
    model_version: str | None = None,
    interval: str | None = None,
) -> dict:
    """
    Сгенерировать сигнал BUY/SELL/HOLD для заданного тикера.

    Использует модель, обученную исключительно на данных этого тикера.
    Результат кешируется в Redis на REDIS_SIGNAL_TTL секунд.

    Аргументы:
        ticker:        тикер инструмента (например, "SBER").
        model_version: суффикс версии модели (None = из ml_settings.model_version).
        interval:      интервал свечей (None = из data_settings.candle_interval).

    Возвращает:
        Словарь с ключами:
            ticker (str):         тикер
            signal (str):         "BUY", "HOLD" или "SELL"
            confidence (float):   вероятность для предсказанного класса
            probabilities (dict): {"SELL": p, "HOLD": p, "BUY": p}

    Исключения:
        FileNotFoundError: если веса ансамбля для тикера не найдены.
        ValueError:        если недостаточно данных свечей.
    """
    version = model_version or ml_settings.model_version
    from config.settings import data_settings
    candle_interval = interval or data_settings.candle_interval

    # ── Redis: проверяем кеш ─────────────────────────────────────────────────
    cache_key = f"signal:{ticker}:{version}"
    redis = await get_redis()
    if redis is not None:
        try:
            cached = await redis.get(cache_key)
            if cached:
                logger.debug("Signal cache hit", ticker=ticker, key=cache_key)
                return json.loads(cached)
        except Exception as e:
            logger.warning("Redis get error", key=cache_key, error=str(e))

    # ── Загрузка модели тикера (in-memory кеш) ────────────────────────────────
    ensemble, feature_columns = _load_model(ticker, version)

    # ── Загрузка данных свечей ────────────────────────────────────────────────
    df = await load_ticker_data(ticker, candle_interval)

    if len(df) < ml_settings.min_candles_predict:
        raise ValueError(
            f"Недостаточно свечей для {ticker}: "
            f"{len(df)} < {ml_settings.min_candles_predict}."
        )

    df = df.tail(ml_settings.min_candles_predict).reset_index(drop=True)

    # ── Вычисляем признаки ────────────────────────────────────────────────────
    feat_df = compute_features(df)

    if feat_df.empty:
        raise ValueError(f"Нет валидных строк признаков для {ticker} после прогрева.")

    last_row = feat_df[feature_columns].iloc[[-1]]

    # ── Предсказание: усредняем вероятности трёх моделей ансамбля ────────────
    proba = ensemble.predict_proba(last_row)[0]
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

    # ── Redis: сохраняем в кеш ────────────────────────────────────────────────
    if redis is not None:
        try:
            await redis.setex(cache_key, redis_settings.signal_ttl, json.dumps(result))
            logger.debug("Signal cached", ticker=ticker, ttl=redis_settings.signal_ttl)
        except Exception as e:
            logger.warning("Redis setex error", key=cache_key, error=str(e))

    return result


async def predict_all(
    tickers: list[str] | None = None,
    model_version: str | None = None,
    interval: str | None = None,
) -> list[dict]:
    """
    Сгенерировать сигналы для списка тикеров. При ошибке — пропускает тикер.

    Аргументы:
        tickers:       список тикеров (None = из data_settings.tickers).
        model_version: версия модели (None = из ml_settings).
        interval:      интервал свечей (None = из data_settings).

    Возвращает:
        Список словарей с сигналами (только для успешно обработанных тикеров).
    """
    from config.settings import data_settings
    tickers = tickers or data_settings.tickers

    results: list[dict] = []
    for ticker in tickers:
        try:
            signal = await predict_signal(ticker, model_version, interval)
            results.append(signal)
        except Exception as e:
            logger.error("Ошибка предсказания", ticker=ticker, error=str(e))
    return results
