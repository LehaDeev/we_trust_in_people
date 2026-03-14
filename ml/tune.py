"""
Подбор гиперпараметров моделей ансамбля через Optuna.

Для каждой модели создаётся отдельное Optuna-исследование (study).
Оценка производится через кросс-валидацию TimeSeriesSplit(n_splits=ML_N_SPLITS)
с метрикой средний P&L на сделку (с учётом SL/TP и комиссии) — напрямую отражает
торговую прибыльность модели. HOLD-сигналы не создают сделок и не влияют на метрику.

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
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import ml_settings, trading_settings
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
        bar = "#" * filled + "." * (_BAR_WIDTH - filled)
        best = study.best_value if study.best_trial else 0.0
        print(
            f"\r    {label:<14} [{bar}] {done:>3}/{n_trials} | best Sharpe={best:.4f}",
            end="",
            flush=True,
        )
        if done == n_trials:
            print()  # перенос строки после завершения

    return callback


# ── P&L-симулятор и CV-оценка ────────────────────────────────────────────────

def _simulate_pnl(
    y_pred: np.ndarray,
    close_window: np.ndarray,
    commission_pct: float,
    sl_pct: float,
    tp_pct: float,
    lookahead: int,
) -> list[float]:
    """
    Симулировать сделки: одна позиция за раз, выход по SL/TP или через lookahead.

    Аргументы:
        y_pred:         предсказанные классы (0=SELL, 1=HOLD, 2=BUY).
        close_window:   матрица цен (N, lookahead+1), close_window[i,j] = close[t_i + j].
        commission_pct: комиссия брокера (доля, например 0.003 = 0.3%).
        sl_pct:         стоп-лосс от цены входа (доля, 0.03 = 3%).
        tp_pct:         тейк-профит от цены входа (доля, 0.05 = 5%).
        lookahead:      максимальное число свечей до выхода.

    Возвращает:
        Список чистых P&L на каждую сделку (доля). Пустой список если сделок не было.
    """
    pnl_list: list[float] = []
    next_available = 0

    for i in range(len(y_pred)):
        if i < next_available:
            continue
        if int(y_pred[i]) != 2:
            continue

        entry = float(close_window[i, 0])
        if entry <= 0.0:
            continue

        sl_price = entry * (1.0 - sl_pct)
        tp_price = entry * (1.0 + tp_pct)
        exit_price = float(close_window[i, lookahead])

        for j in range(1, lookahead + 1):
            c = float(close_window[i, j])
            if c <= sl_price:
                exit_price = sl_price
                break
            if c >= tp_price:
                exit_price = tp_price
                break

        pnl_list.append((exit_price - entry) / entry - 2.0 * commission_pct)
        next_available = i + lookahead

    return pnl_list


def _sharpe_score(pnl_list: list[float], min_trades: int) -> float:
    """
    Вычислить Sharpe ratio по списку P&L сделок.

    Sharpe = mean(P&L) / std(P&L) — предпочитает стабильный доход перед
    случайными всплесками с той же средней доходностью.

    При недостатке сделок (< min_trades) возвращает 0.0 — штраф моделям,
    которые почти не генерируют BUY-сигналы.

    Аргументы:
        pnl_list:   список P&L сделок.
        min_trades: минимальное число сделок для расчёта (иначе 0.0).

    Возвращает:
        Sharpe ratio или 0.0.
    """
    if len(pnl_list) < min_trades:
        return 0.0
    arr = np.array(pnl_list)
    std = float(np.std(arr))
    if std < 1e-9:
        # Все сделки одинаковы — Sharpe не определён, возвращаем mean
        return float(np.mean(arr))
    return float(np.mean(arr) / std)


def _cv_pnl_score(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    close_window: np.ndarray,
    cv: TimeSeriesSplit,
) -> float:
    """
    Кросс-валидация с Sharpe-метрикой: Sharpe по всем сделкам из всех фолдов.

    Собираем все P&L из всех фолдов в один список — это стабильнее чем
    усреднять Sharpe по фолдам (в каждом фолде мало сделок → шумно).

    Аргументы:
        model:        sklearn-совместимый классификатор.
        X:            DataFrame признаков.
        y:            Series меток.
        close_window: матрица цен (N, lookahead+1), выровненная с X/y.
        cv:           экземпляр TimeSeriesSplit.

    Возвращает:
        Sharpe ratio по всем CV-сделкам. 0.0 если сделок меньше min_trades.
    """
    all_pnl: list[float] = []

    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_pred = model.fit(X_train, y_train).predict(X_val)

        all_pnl.extend(_simulate_pnl(
            y_pred=y_pred,
            close_window=close_window[val_idx],
            commission_pct=trading_settings.broker_commission_pct,
            sl_pct=trading_settings.stop_loss_pct,
            tp_pct=trading_settings.take_profit_pct,
            lookahead=ml_settings.lookahead,
        ))

    return _sharpe_score(all_pnl, min_trades=ml_settings.sharpe_min_trades)


# ── LightGBM ─────────────────────────────────────────────────────────────────

def tune_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    close_window: np.ndarray,
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
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        model = Pipeline([("scaler", StandardScaler()), ("model", lgb.LGBMClassifier(**params))])
        return _cv_pnl_score(model, X, y, close_window, cv)

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
        best_sharpe=round(study.best_value, 4),
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
    close_window: np.ndarray,
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
        return _cv_pnl_score(model, X, y, close_window, cv)

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
        best_sharpe=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_et_{version}.json")
    return best_params


def tune_svc(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int | None = None,
    version: str | None = None,
) -> dict:
    """
    Подобрать гиперпараметры SVC через Optuna.

    SVC (RBF-ядро) — принципиально иной алгоритм: максимизирует отступ между
    классами в пространстве признаков, а не строит деревья решений. Даёт
    реальное разнообразие ансамблю с LGBM и ExtraTrees.

    Внимание: probability=True включает Platt scaling (внутренняя CV SVC),
    что существенно замедляет обучение. На 7000+ строк каждый трайл занимает
    ~1–2 минуты — держать ML_OPTUNA_TRIALS_SVC <= 20.

    Аргументы:
        X:        DataFrame признаков.
        y:        Series меток (0, 1, 2).
        n_trials: количество итераций (None = из ml_settings).
        version:  версия для имени файла (None = из ml_settings).

    Возвращает:
        Словарь лучших гиперпараметров.
    """
    n_trials = n_trials or ml_settings.optuna_trials_svc
    version = version or ml_settings.model_version
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            # C: штраф за ошибку классификации; выше → жёстче границы, меньше → шире отступ
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            # gamma: ширина RBF-ядра; scale = 1/(n_features * X.var()) — хорошая отправная точка
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "kernel": "rbf",
            "probability": True,   # нужно для soft voting (predict_proba)
            "class_weight": "balanced",
            "random_state": ml_settings.random_state,
        }
        model = Pipeline([("scaler", StandardScaler()), ("model", SVC(**params))])
        return _cv_f1_score(model, X, y, cv)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[_make_progress_callback(n_trials, "SVC HPO")],
    )

    best_params = study.best_params
    best_params.update({
        "kernel": "rbf",
        "probability": True,
        "class_weight": "balanced",
        "random_state": ml_settings.random_state,
    })

    logger.info(
        "SVC tuning complete",
        best_f1=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_svc_{version}.json")
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
