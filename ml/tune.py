"""
Подбор гиперпараметров моделей ансамбля через Optuna.

Для каждой модели создаётся отдельное Optuna-исследование (study).
Оценка производится через кросс-валидацию TimeSeriesSplit(n_splits=ML_N_SPLITS)
с метрикой f1_macro — корректно учитывает дисбаланс классов HOLD/BUY/SELL.

Лучшие параметры сохраняются в ml/weights/best_params_{model}_{version}.json.
"""
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from config.settings import ml_settings
from utils.logger import logger

# Подавляем стандартный вывод Optuna — используем structlog
optuna.logging.set_verbosity(optuna.logging.WARNING)

WEIGHTS_DIR = Path(__file__).parent / "weights"


# ── Вспомогательная функция кросс-валидации ─────────────────────────────────

def _cv_f1_score(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplit,
) -> float:
    """
    Вычислить среднее f1_macro по всем фолдам TimeSeriesSplit.

    Аргументы:
        model: sklearn-совместимый классификатор с fit/predict.
        X:     DataFrame признаков.
        y:     Series меток.
        cv:    экземпляр TimeSeriesSplit.

    Возвращает:
        Среднее значение f1_macro по всем фолдам.
    """
    scores: list[float] = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

    return float(np.mean(scores))


# ── LightGBM ─────────────────────────────────────────────────────────────────

def tune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    version: str | None = None,
) -> dict:
    """
    Подобрать гиперпараметры LightGBM через Optuna.

    Аргументы:
        X:        DataFrame признаков.
        y:        Series меток (0, 1, 2).
        n_trials: количество итераций (None = из ml_settings).
        version:  версия для имени файла (None = из ml_settings).

    Возвращает:
        Словарь лучших гиперпараметров.
    """
    n_trials = n_trials or ml_settings.optuna_trials_lgbm
    version = version or ml_settings.model_version
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "random_state": ml_settings.random_state,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        "objective": "multiclass",
        "num_class": 3,
        "random_state": ml_settings.random_state,
        "verbose": -1,
    })

    logger.info(
        "LightGBM tuning complete",
        best_f1=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_lgbm_{version}.json")
    return best_params


# ── XGBoost ──────────────────────────────────────────────────────────────────

def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    version: str | None = None,
) -> dict:
    """
    Подобрать гиперпараметры XGBoost через Optuna.

    Аргументы:
        X:        DataFrame признаков.
        y:        Series меток (0, 1, 2).
        n_trials: количество итераций (None = из ml_settings).
        version:  версия для имени файла (None = из ml_settings).

    Возвращает:
        Словарь лучших гиперпараметров.
    """
    n_trials = n_trials or ml_settings.optuna_trials_xgb
    version = version or ml_settings.model_version
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": ml_settings.random_state,
            "verbosity": 0,
            "eval_metric": "mlogloss",
        }
        model = xgb.XGBClassifier(**params)
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": ml_settings.random_state,
        "verbosity": 0,
        "eval_metric": "mlogloss",
    })

    logger.info(
        "XGBoost tuning complete",
        best_f1=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_xgboost_{version}.json")
    return best_params


# ── Random Forest ─────────────────────────────────────────────────────────────

def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    version: str | None = None,
) -> dict:
    """
    Подобрать гиперпараметры RandomForest через Optuna.

    Аргументы:
        X:        DataFrame признаков.
        y:        Series меток (0, 1, 2).
        n_trials: количество итераций (None = из ml_settings).
        version:  версия для имени файла (None = из ml_settings).

    Возвращает:
        Словарь лучших гиперпараметров.
    """
    n_trials = n_trials or ml_settings.optuna_trials_rf
    version = version or ml_settings.model_version
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": ml_settings.random_state,
            "n_jobs": -1,
        }
        model = RandomForestClassifier(**params)
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"random_state": ml_settings.random_state, "n_jobs": -1})

    logger.info(
        "RandomForest tuning complete",
        best_f1=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_rf_{version}.json")
    return best_params


# ── Утилита сохранения ───────────────────────────────────────────────────────

def _save_params(params: dict, filename: str) -> None:
    """Сохранить словарь параметров в JSON-файл в директории весов."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / filename
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Best params saved", path=str(path))
