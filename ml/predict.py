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
    # {"ticker": "SBER", "signal": "BUY", "confidence": 0.0083, "volume_ratio": 1.24}
"""
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np

from config.settings import ml_settings, redis_settings
from ml.dataset import load_ticker_data, load_usdrub_data, merge_usdrub
from ml.features import compute_features
from utils.logger import logger
from utils.redis_cache import get_redis

WEIGHTS_DIR = Path(__file__).parent / "weights"

# LRU-кеш загруженных моделей: "{ticker}_{version}" → (ensemble, feature_columns).
# Размер ограничен ML_MODEL_CACHE_SIZE (по умолчанию 1) — вытесняет старую модель
# при загрузке новой, освобождая память. Тикеры обрабатываются поочерёдно,
# поэтому держать более одной модели в RAM нет смысла.
_model_cache: OrderedDict[str, tuple] = OrderedDict()


def _load_entry_threshold(ticker: str, version: str) -> float:
    """
    Загрузить абсолютный порог входа pnl_pred для тикера из файла весов.

    Порог — абсолютное значение pnl_pred, соответствующее оптимальному персентилю
    на валидационной выборке. Если файл не найден — возвращает ml_settings.threshold.

    Аргументы:
        ticker:  тикер инструмента.
        version: версия модели.

    Возвращает:
        Абсолютный порог (float). Может быть отрицательным при персентильном подходе.
    """
    path = WEIGHTS_DIR / f"best_threshold_{ticker}_{version}.json"
    try:
        with open(path) as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return ml_settings.threshold


def _load_model(ticker: str, version: str) -> tuple:
    """
    Загрузить ансамбль тикера и список признаков (из памяти или с диска).

    Аргументы:
        ticker:  тикер инструмента (например, "SBER").
        version: версия модели (например, "v2").

    Возвращает:
        (ensemble, feature_columns): объект RankEnsemble и список имён признаков.

    Исключения:
        FileNotFoundError: если файл весов для тикера не найден.
    """
    cache_key = f"{ticker}_{version}"
    if cache_key in _model_cache:
        # Переместить в конец — LRU: только что использован
        _model_cache.move_to_end(cache_key)
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

    # Вытеснить самую старую запись если превышен лимит LRU-кеша
    while len(_model_cache) > ml_settings.model_cache_size:
        evicted, _ = _model_cache.popitem(last=False)
        logger.info("Модель вытеснена из LRU-кеша", key=evicted)

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
            confidence (float):   предсказанный net P&L (доля; например 0.0083 = 0.83%).
                                  Положительный → BUY, отрицательный → SELL, ноль → HOLD.
            volume_ratio (float): объём последнего бара / SMA_20 объёма (фильтр подтверждения)

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

    # ── Добавляем USD/RUB как признак ─────────────────────────────────────────
    usdrub_df = await load_usdrub_data(candle_interval)
    df = merge_usdrub(df, usdrub_df)

    # ── Вычисляем признаки ────────────────────────────────────────────────────
    feat_df = compute_features(df)

    if feat_df.empty:
        raise ValueError(f"Нет валидных строк признаков для {ticker} после прогрева.")

    last_row = feat_df[feature_columns].iloc[[-1]]

    # ── Предсказание ──────────────────────────────────────────────────────────
    # Регрессор предсказывает ожидаемый net P&L (доля) для текущего бара.
    # Порог загружается из best_threshold_{ticker}_{version}.json — абсолютное значение
    # pnl_pred, соответствующее оптимальному персентилю (обычно негативное, например -0.003).
    # BUY  если pnl_pred >= threshold (бар в топ-(100-P)% по прогнозу модели).
    # SELL если pnl_pred < 0 (ожидаем убыток — сигнал выхода из позиции).
    # HOLD иначе (между 0 и threshold: выше порога, но прогноз ещё положительный).
    pnl_pred = float(ensemble.predict(last_row)[0])
    entry_threshold = _load_entry_threshold(ticker, version)
    if pnl_pred >= entry_threshold:
        signal = "BUY"
    elif pnl_pred < 0:
        signal = "SELL"
    else:
        signal = "HOLD"

    # volume_ratio последнего бара: используется как фильтр подтверждения в scheduler
    # > 1.0 — объём выше среднего (движение подтверждено), < 1.0 — слабый объём
    last_volume_ratio = float(feat_df["volume_ratio"].iloc[-1]) if "volume_ratio" in feat_df.columns else 1.0

    result = {
        "ticker": ticker,
        "signal": signal,
        # confidence = предсказанный net P&L (доля; например 0.008 = 0.8%).
        # Scheduler сравнивает это значение с per-ticker порогом входа.
        "confidence": round(pnl_pred, 6),
        "volume_ratio": round(last_volume_ratio, 4),
    }

    logger.info(
        "Signal generated",
        ticker=ticker,
        signal=signal,
        confidence=round(pnl_pred, 4),
        volume_ratio=round(last_volume_ratio, 4),
    )

    # ── Redis: сохраняем в кеш ────────────────────────────────────────────────
    if redis is not None:
        try:
            await redis.setex(cache_key, redis_settings.signal_ttl, json.dumps(result))
            logger.debug("Signal cached", ticker=ticker, ttl=redis_settings.signal_ttl)
        except Exception as e:
            logger.warning("Redis setex error", key=cache_key, error=str(e))

    return result


def clear_model_cache() -> None:
    """
    Очистить in-memory кеш загруженных ансамблей.

    Вызывать после переобучения моделей — следующий вызов predict_signal()
    загрузит обновлённые веса с диска.
    """
    _model_cache.clear()
    logger.info("In-memory кеш моделей очищен")


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
