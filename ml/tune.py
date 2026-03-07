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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import ml_settings
from utils.logger import logger

# Подавляем стандартный вывод Optuna — используем собственный прогресс
optuna.logging.set_verbosity(optuna.logging.WARNING)

WEIGHTS_DIR = Path(__file__).parent / "weights"

_BAR_WIDTH = 20


def _make_progress_callback(n_trials: int, label: str):
    """
    Создать callback для Optuna, выводящий прогресс-бар в терминал.

    Аргументы:
        n_trials: общее количество проб.
        label:    название модели для отображения.
    """
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        done = trial.number + 1
        filled = int(_BAR_WIDTH * done / n_trials)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        best = study.best_value if study.best_trial else 0.0
        print(
            f"\r    {label:<14} [{bar}] {done:>3}/{n_trials} | best F1={best:.4f}",
            end="",
            flush=True,
        )
        if done == n_trials:
            print()  # перенос строки после завершения

    return callback


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
            # Балансировка классов: HOLD доминирует (60–80%), без этого модель
            # выучивает "всегда HOLD" и F1_macro падает к случайному уровню ~0.33
            "class_weight": "balanced",
        }
        model = Pipeline([("scaler", StandardScaler()), ("model", lgb.LGBMClassifier(**params))])
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_make_progress_callback(n_trials, "LightGBM HPO")],
    )

    best_params = study.best_params
    best_params.update({
        "objective": "multiclass",
        "num_class": 3,
        "random_state": ml_settings.random_state,
        "verbose": -1,
        "class_weight": "balanced",
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
        model = Pipeline([("scaler", StandardScaler()), ("model", xgb.XGBClassifier(**params))])
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_make_progress_callback(n_trials, "XGBoost HPO")],
    )

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

def tune_extra_trees(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    version: str | None = None,
) -> dict:
    """
    Подобрать гиперпараметры ExtraTreesClassifier через Optuna.

    ExtraTrees использует случайные пороги разбиений (вместо лучших как в RF),
    что даёт низкую корреляцию с LightGBM и реальное разнообразие ансамблю.

    Аргументы:
        X:        DataFrame признаков.
        y:        Series меток (0, 1, 2).
        n_trials: количество итераций (None = из ml_settings).
        version:  версия для имени файла (None = из ml_settings).

    Возвращает:
        Словарь лучших гиперпараметров.
    """
    n_trials = n_trials or ml_settings.optuna_trials_et
    version = version or ml_settings.model_version
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            # max_features: ET особенно чувствителен к этому параметру
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            "random_state": ml_settings.random_state,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        model = Pipeline([("scaler", StandardScaler()), ("model", ExtraTreesClassifier(**params))])
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_make_progress_callback(n_trials, "ExtraTrees HPO")],
    )

    best_params = study.best_params
    best_params.update({"random_state": ml_settings.random_state, "n_jobs": -1, "class_weight": "balanced"})

    logger.info(
        "ExtraTrees tuning complete",
        best_f1=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_et_{version}.json")
    return best_params


# ── CatBoost ──────────────────────────────────────────────────────────────────

def tune_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    version: str | None = None,
) -> dict:
    """
    Подобрать гиперпараметры CatBoost через Optuna.

    Аргументы:
        X:        DataFrame признаков.
        y:        Series меток (0, 1, 2).
        n_trials: количество итераций (None = из ml_settings).
        version:  версия для имени файла (None = из ml_settings).

    Возвращает:
        Словарь лучших гиперпараметров.
    """
    from catboost import CatBoostClassifier  # ленивый импорт — catboost опциональная зависимость

    n_trials = n_trials or ml_settings.optuna_trials_catboost
    version = version or ml_settings.model_version
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            # subsample поддерживается только при bootstrap_type="Bernoulli" или "MVS"
            # (по умолчанию "Bayesian" не поддерживает subsample → CatBoostError)
            "bootstrap_type": "Bernoulli",
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
            "random_seed": ml_settings.random_state,
            "verbose": 0,
            # CatBoost поддерживает балансировку классов нативно
            "auto_class_weights": "Balanced",
            "loss_function": "MultiClass",
        }
        model = Pipeline([("scaler", StandardScaler()), ("model", CatBoostClassifier(**params))])
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_make_progress_callback(n_trials, "CatBoost HPO")],
    )

    best_params = study.best_params
    best_params.update({
        "bootstrap_type": "Bernoulli",
        "random_seed": ml_settings.random_state,
        "verbose": 0,
        "auto_class_weights": "Balanced",
        "loss_function": "MultiClass",
    })

    logger.info(
        "CatBoost tuning complete",
        best_f1=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_catboost_{version}.json")
    return best_params


# ── Утилита сохранения ───────────────────────────────────────────────────────

def _save_params(params: dict, filename: str) -> None:
    """Сохранить словарь параметров в JSON-файл в директории весов."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / filename
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Best params saved", path=str(path))
