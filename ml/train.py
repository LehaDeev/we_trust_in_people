"""
Обучение ансамбля моделей для предсказания сигналов BUY/SELL/HOLD.

Pipeline:
    1. Загрузка свечей из PostgreSQL
    2. Вычисление признаков (TA-Lib индикаторы)
    3. Генерация меток по будущей доходности
    4. Подбор гиперпараметров через Optuna (с кешем из best_params_*.json)
    5. Обучение финальных моделей на полном датасете
    6. Сборка soft voting ансамбля
    7. Сохранение весов в ml/weights/

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

def _load_cached_params(model_name: str) -> dict | None:
    """
    Загрузить сохранённые гиперпараметры из JSON-файла, если он существует.

    Аргументы:
        model_name: имя модели ("lgbm", "xgboost", "rf").

    Возвращает:
        Словарь параметров или None если кеша нет.
    """
    path = WEIGHTS_DIR / f"best_params_{model_name}_{ml_settings.model_version}.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info("Загружены кешированные параметры", model=model_name, path=str(path))
        return params
    return None


def _get_params(
    model_name: str,
    tune_fn,
    X: pd.DataFrame,
    y: pd.Series,
    force_tune: bool,
) -> dict:
    """
    Вернуть гиперпараметры: из кеша если есть, иначе запустить Optuna.

    Аргументы:
        model_name: имя модели для поиска кеша.
        tune_fn:    функция подбора параметров из ml/tune.py.
        X:          DataFrame признаков.
        y:          Series меток.
        force_tune: если True — игнорировать кеш и запустить Optuna заново.

    Возвращает:
        Словарь гиперпараметров.
    """
    if not force_tune:
        cached = _load_cached_params(model_name)
        if cached is not None:
            return cached

    logger.info("Запуск Optuna HPO", model=model_name)
    return tune_fn(X, y)


# ── Сборка датасета ──────────────────────────────────────────────────────────

def _build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Применить вычисление признаков и генерацию меток отдельно по каждому тикеру.

    Аргументы:
        raw: объединённый DataFrame с колонками [ticker, time, open, high, low, close, volume].

    Возвращает:
        (X, y): DataFrame признаков и Series меток с согласованными индексами.
    """
    feature_frames: list[pd.DataFrame] = []
    label_series: list[pd.Series] = []

    for ticker, group in raw.groupby("ticker", sort=False):
        group = group.reset_index(drop=True)

        # Признаки (также удаляет строки прогрева с NaN)
        feat_df = compute_features(group)

        # Метки (удаляет последние lookahead строк)
        labels = create_labels(
            feat_df,
            lookahead=ml_settings.lookahead,
            threshold=ml_settings.threshold,
        )

        # Выравнивание: признаки и метки должны покрывать одни и те же строки
        feat_df = feat_df.loc[labels.index].copy()

        feature_frames.append(feat_df)
        label_series.append(labels)

        logger.info(
            "Ticker dataset built",
            ticker=ticker,
            samples=len(labels),
            buy=int((labels == 2).sum()),
            hold=int((labels == 1).sum()),
            sell=int((labels == 0).sum()),
        )

    if not feature_frames:
        raise RuntimeError("Нет обучающих данных. Запустите scripts/collect_candles.py.")

    combined_features = pd.concat(feature_frames, ignore_index=True)
    combined_labels = pd.concat(label_series, ignore_index=True)

    return combined_features[FEATURE_COLUMNS], combined_labels


# ── Оценка финального ансамбля ───────────────────────────────────────────────

def _evaluate_ensemble(
    ensemble: VotingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
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
        "Ensemble CV evaluation",
        f1_per_fold=[round(s, 4) for s in scores],
        mean_f1=round(mean_f1, 4),
        std_f1=round(float(np.std(scores)), 4),
    )
    return mean_f1


# ── Основная функция обучения ────────────────────────────────────────────────

async def train_model(force_tune: bool = False) -> Path:
    """
    Полный pipeline: загрузка → признаки → метки → HPO (с кешем) → ансамбль → сохранение.

    Гиперпараметры берутся из кеша (best_params_*.json) если файлы существуют.
    При force_tune=True кеш игнорируется и Optuna запускается заново.
    Финальное обучение ансамбля выполняется всегда на свежих данных из БД.

    Аргументы:
        force_tune: если True — принудительно запустить Optuna для всех моделей.

    Возвращает:
        Path к сохранённому файлу ансамбля pkl.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if force_tune:
        logger.info("Режим force_tune: Optuna будет запущена для всех моделей")
    else:
        logger.info("Режим обычного обучения: Optuna пропускается если есть кеш")

    logger.info("Загрузка данных свечей из БД...", tickers=data_settings.tickers)
    raw = await load_all_tickers_dataset(
        data_settings.tickers,
        interval=data_settings.candle_interval,
    )

    if raw.empty:
        raise RuntimeError("Нет данных. Запустите scripts/collect_candles.py.")

    logger.info("Сборка датасета признаков и меток...")
    X, y = _build_dataset(raw)

    logger.info(
        "Датасет готов",
        total_samples=len(X),
        features=len(FEATURE_COLUMNS),
        lookahead=ml_settings.lookahead,
        threshold=ml_settings.threshold,
        class_distribution=y.value_counts().to_dict(),
    )

    # ── Гиперпараметры: из кеша или через Optuna ──────────────────────────────
    lgbm_params = _get_params("lgbm", tune_lgbm, X, y, force_tune)
    xgb_params = _get_params("xgboost", tune_xgboost, X, y, force_tune)
    rf_params = _get_params("rf", tune_random_forest, X, y, force_tune)

    # ── Создание финальных моделей с лучшими параметрами ─────────────────────
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    rf_model = RandomForestClassifier(**rf_params)

    # ── Soft voting ансамбль ──────────────────────────────────────────────────
    # voting='soft' — усредняем вероятности, не голосуем за класс (точнее)
    ensemble = VotingClassifier(
        estimators=[
            ("lgbm", lgbm_model),
            ("xgb", xgb_model),
            ("rf", rf_model),
        ],
        voting="soft",
    )

    # ── Оценка до финального обучения ─────────────────────────────────────────
    logger.info("Оценка ансамбля через TimeSeriesSplit CV...")
    cv_f1 = _evaluate_ensemble(ensemble, X, y)

    # ── Финальное обучение на полном датасете (всегда на свежих данных) ───────
    logger.info("Финальное обучение ансамбля на полном датасете...")
    ensemble.fit(X, y)

    # ── Сохранение ────────────────────────────────────────────────────────────
    version = ml_settings.model_version
    ensemble_path = WEIGHTS_DIR / f"ensemble_{version}.pkl"
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)

    features_path = WEIGHTS_DIR / f"features_{version}.json"
    with open(features_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    logger.info(
        "Ансамбль сохранён",
        ensemble_path=str(ensemble_path),
        features_path=str(features_path),
        cv_f1_macro=round(cv_f1, 4),
    )

    return ensemble_path
