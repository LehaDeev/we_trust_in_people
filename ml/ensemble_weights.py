"""
Вычисление адаптивных весов ансамбля RankEnsemble по OOS-корреляции Спирмена.

Вынесено из train.py для соблюдения ограничения 800 строк на файл.
Импортируется только из train.py — публичный интерфейс не предназначен для
использования за пределами ML-pipeline.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.base import clone as sk_clone

from config.settings import ml_settings
from utils.logger import logger

if TYPE_CHECKING:
    import pandas as pd

    from ml.ensemble import RankEnsemble


def compute_ensemble_weights(
    ensemble: "RankEnsemble",
    X: "pd.DataFrame",
    y: "pd.Series",
) -> list[float]:
    """
    Вычислить адаптивные веса ансамбля через softmax Spearman-корреляции на OOS-фолде.

    Берётся последний walk-forward фолд — наиболее близкий к реальному OOS периоду.
    Для каждой базовой модели:
        1. Клонируем модель (копируем гиперпараметры, не обученные веса).
        2. Обучаем клон на train_idx — val_idx модель не видела (OOS).
        3. Считаем Spearman(model_clone.predict(X_val), y_val).
    Веса = softmax(spearman_scores / ML_ENSEMBLE_WEIGHT_TEMP).

    Если данных недостаточно для WalkForwardSplit — возвращает равные веса.

    Аргументы:
        ensemble: обученный RankEnsemble (используем estimators_ для клонирования).
        X:        DataFrame признаков (те же, на которых обучен финальный ансамбль).
        y:        Series целевых P&L.

    Возвращает:
        Список весов длиной len(ensemble.estimators_), sum=1.
    """
    from ml.walk_forward import WalkForwardSplit

    n_models = len(ensemble.estimators_)
    equal_weights = [1.0 / n_models] * n_models

    # Получаем последний фолд WalkForwardSplit
    cv = WalkForwardSplit()
    folds = cv.split(X)
    if not folds:
        logger.warning("WalkForwardSplit вернул пустой список — используем равные веса")
        return equal_weights

    train_idx, val_idx = folds[-1]  # последний фолд — ближайший к OOS
    X_train_f = X.iloc[train_idx]
    y_train_f = y.iloc[train_idx]
    X_val_f = X.iloc[val_idx]
    y_val_f = y.iloc[val_idx].values

    temp = ml_settings.ensemble_weight_temp
    model_names = [name for name, _ in ensemble.estimators]
    spearman_scores: list[float] = []

    for idx, (name, _) in enumerate(ensemble.estimators):
        # Клонируем базовую модель из estimators_ (уже обученную на всех данных).
        # Клон копирует только гиперпараметры — затем обучаем его заново на train_idx фолда.
        model_clone = sk_clone(ensemble.estimators_[idx])
        try:
            model_clone.fit(X_train_f, y_train_f)
            preds = model_clone.predict(X_val_f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConstantInputWarning)
                corr, _ = spearmanr(preds, y_val_f)
            score = float(corr) if not np.isnan(corr) else 0.0
        except Exception as exc:
            logger.warning(
                "Ошибка при вычислении весов для модели",
                model=name,
                error=str(exc),
            )
            score = 0.0
        spearman_scores.append(score)
        logger.info("OOS Spearman базовой модели", model=name, spearman=round(score, 4))

    # Численно устойчивый softmax с температурой
    arr = np.array(spearman_scores, dtype=float)
    arr_scaled = arr / max(temp, 1e-9)
    arr_scaled -= arr_scaled.max()  # вычитаем max для численной устойчивости
    exp_arr = np.exp(arr_scaled)
    weights = (exp_arr / exp_arr.sum()).tolist()

    logger.info(
        "Адаптивные веса ансамбля",
        spearman_scores={n: round(s, 4) for n, s in zip(model_names, spearman_scores)},
        weights={n: round(w, 4) for n, w in zip(model_names, weights)},
        temperature=temp,
    )
    return weights
