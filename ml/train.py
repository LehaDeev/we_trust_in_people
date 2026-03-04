"""
Обучение ансамблей моделей — по одному на каждый тикер.

Для каждого тикера выполняется отдельный pipeline:
    1. Загрузка свечей из PostgreSQL
    2. Вычисление признаков (TA-Lib индикаторы)
    3. Генерация меток по будущей доходности
    4. Подбор гиперпараметров через Optuna (с кешем best_params_*_{ticker}_{version}.json)
    5. Обучение финального ансамбля (LightGBM + XGBoost + RandomForest)
    6. Сохранение весов в ml/weights/ensemble_{ticker}_{version}.pkl

Запуск:
    python -m scripts.train_model               # Optuna только если нет кеша
    python -m scripts.train_model --force-tune  # Принудительный повтор Optuna

Все настройки — в .env (ML_* и DATA_*).
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from config.settings import data_settings, ml_settings
from ml.dataset import load_all_tickers_dataset
from ml.features import FEATURE_COLUMNS, compute_features
from ml.labels import LABEL_NAMES, create_labels
from ml.tune import tune_lgbm, tune_random_forest, tune_xgboost
from utils.logger import logger

WEIGHTS_DIR = Path(__file__).parent / "weights"


# ── Кеш гиперпараметров ──────────────────────────────────────────────────────

def _load_cached_params(model_name: str, ticker_version: str) -> dict | None:
    """
    Загрузить сохранённые гиперпараметры из JSON-файла, если он существует.

    Аргументы:
        model_name:     имя модели ("lgbm", "xgboost", "rf").
        ticker_version: строка вида "{ticker}_{version}" (например, "SBER_v2").

    Возвращает:
        Словарь параметров или None если кеша нет.
    """
    path = WEIGHTS_DIR / f"best_params_{model_name}_{ticker_version}.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info("Загружены кешированные параметры", model=model_name, path=str(path))
        return params
    return None


def _get_params(
    model_name: str,
    ticker_version: str,
    tune_fn,
    X: pd.DataFrame,
    y: pd.Series,
    force_tune: bool,
) -> dict:
    """
    Вернуть гиперпараметры: из кеша если есть, иначе запустить Optuna.

    Аргументы:
        model_name:     имя модели для поиска кеша.
        ticker_version: строка "{ticker}_{version}" для имён файлов.
        tune_fn:        функция подбора параметров из ml/tune.py.
        X:              DataFrame признаков.
        y:              Series меток.
        force_tune:     если True — игнорировать кеш и запустить Optuna заново.

    Возвращает:
        Словарь гиперпараметров.
    """
    if not force_tune:
        cached = _load_cached_params(model_name, ticker_version)
        if cached is not None:
            return cached

    logger.info("Запуск Optuna HPO", model=model_name, ticker_version=ticker_version)
    return tune_fn(X, y, version=ticker_version)


# ── Сборка датасета для одного тикера ────────────────────────────────────────

def _build_ticker_dataset(
    group: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Вычислить признаки и метки для одного тикера.

    Аргументы:
        group:  DataFrame свечей одного тикера [time, open, high, low, close, volume].
        ticker: тикер (только для логирования).

    Возвращает:
        (X, y): DataFrame признаков и Series меток с согласованными индексами.
    """
    feat_df = compute_features(group)
    labels = create_labels(
        feat_df,
        lookahead=ml_settings.lookahead,
        threshold=ml_settings.threshold,
    )
    feat_df = feat_df.loc[labels.index].copy()

    logger.info(
        "Датасет тикера собран",
        ticker=ticker,
        samples=len(labels),
        buy=int((labels == 2).sum()),
        hold=int((labels == 1).sum()),
        sell=int((labels == 0).sum()),
    )

    return feat_df[FEATURE_COLUMNS], labels


# ── Оценка ансамбля ──────────────────────────────────────────────────────────

def _evaluate_ensemble(
    ensemble: VotingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    ticker: str,
) -> float:
    """
    Оценить ансамбль через кросс-валидацию TimeSeriesSplit.

    Возвращает:
        Среднее f1_macro по всем фолдам.
    """
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)
    scores = cross_val_score(ensemble, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    mean_f1 = float(np.mean(scores))
    logger.info(
        "Оценка ансамбля (CV)",
        ticker=ticker,
        f1_per_fold=[round(s, 4) for s in scores],
        mean_f1=round(mean_f1, 4),
    )
    return mean_f1


# ── Обучение одного тикера ───────────────────────────────────────────────────

def _train_single_ticker(
    ticker: str,
    group: pd.DataFrame,
    force_tune: bool,
) -> Path:
    """
    Полный pipeline обучения для одного тикера.

    Аргументы:
        ticker:     тикер инструмента.
        group:      DataFrame свечей этого тикера.
        force_tune: принудительно запустить Optuna.

    Возвращает:
        Path к сохранённому pkl ансамбля.
    """
    version = ml_settings.model_version
    ticker_version = f"{ticker}_{version}"

    X, y = _build_ticker_dataset(group, ticker)

    if len(X) < ml_settings.n_splits * 2:
        raise RuntimeError(
            f"Недостаточно данных для {ticker}: {len(X)} строк. "
            "Соберите больше свечей."
        )

    # Гиперпараметры: из кеша или через Optuna
    lgbm_params = _get_params("lgbm", ticker_version, tune_lgbm, X, y, force_tune)
    xgb_params = _get_params("xgboost", ticker_version, tune_xgboost, X, y, force_tune)
    rf_params = _get_params("rf", ticker_version, tune_random_forest, X, y, force_tune)

    ensemble = VotingClassifier(
        estimators=[
            ("lgbm", lgb.LGBMClassifier(**lgbm_params)),
            ("xgb", xgb.XGBClassifier(**xgb_params)),
            ("rf", RandomForestClassifier(**rf_params)),
        ],
        voting="soft",
    )

    logger.info("Оценка ансамбля через TimeSeriesSplit CV...", ticker=ticker)
    cv_f1 = _evaluate_ensemble(ensemble, X, y, ticker)

    logger.info("Финальное обучение ансамбля...", ticker=ticker)
    ensemble.fit(X, y)

    ensemble_path = WEIGHTS_DIR / f"ensemble_{ticker_version}.pkl"
    features_path = WEIGHTS_DIR / f"features_{ticker_version}.json"

    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    with open(features_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    logger.info(
        "Ансамбль тикера сохранён",
        ticker=ticker,
        ensemble_path=str(ensemble_path),
        cv_f1_macro=round(cv_f1, 4),
    )

    return ensemble_path


# ── Основная функция обучения ────────────────────────────────────────────────

async def train_model(force_tune: bool = False) -> dict[str, Path]:
    """
    Обучить отдельный ансамбль для каждого тикера из DATA_TICKERS.

    Для каждого тикера свой Optuna HPO (с кешем), свои веса.
    Файлы: ensemble_{ticker}_{version}.pkl, features_{ticker}_{version}.json.

    Аргументы:
        force_tune: если True — принудительно запустить Optuna для всех моделей.

    Возвращает:
        Словарь {ticker: Path} для успешно обученных тикеров.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if force_tune:
        logger.info("Режим force_tune: Optuna будет запущена для всех моделей")

    logger.info("Загрузка данных свечей...", tickers=data_settings.tickers)
    raw = await load_all_tickers_dataset(
        data_settings.tickers,
        interval=data_settings.candle_interval,
    )

    if raw.empty:
        raise RuntimeError("Нет данных. Запустите scripts/collect_candles.py.")

    results: dict[str, Path] = {}

    for ticker, group in raw.groupby("ticker", sort=False):
        ticker = str(ticker)
        group = group.reset_index(drop=True)
        logger.info("Обучение модели", ticker=ticker)
        try:
            path = _train_single_ticker(ticker, group, force_tune)
            results[ticker] = path
        except Exception as e:
            logger.error("Ошибка обучения тикера", ticker=ticker, error=str(e))

    logger.info(
        "Обучение завершено",
        trained=list(results.keys()),
        failed=[t for t in data_settings.tickers if t not in results],
    )

    return results
