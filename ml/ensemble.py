"""
Ансамбль регрессоров RankEnsemble с z-score нормализацией и адаптивными весами.

Используется в train.py (обучение) и predict.py (инференс).
Pkl-файл ensemble_{ticker}_{version}.pkl хранит сериализованный RankEnsemble.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class RankEnsemble:
    """
    Ансамбль регрессоров с z-score нормализацией предсказаний и адаптивными весами.

    Решает проблему VotingRegressor: простое среднее подавляется моделью с наибольшим
    диапазоном предсказаний, что снижает Spearman ансамбля ниже уровня лучшей модели.

    Алгоритм predict:
        1. Каждая модель i предсказывает вектор p_i.
        2. z_i = (p_i - train_mean_i) / train_std_i  — нормализация по обучающей статистике.
        3. Результат = weighted_average(z_i, weights=_weights).

    Веса _weights:
        - По умолчанию (после fit без set_weights): равные [1/N, ..., 1/N].
        - После set_weights(): softmax Spearman-корреляции на OOS val-фолде.
          Модели с более высоким Spearman получают больший вес.
        - Отрицательные Spearman обнуляются перед softmax (не штрафуются, но не вредят).

    Backward compatibility: старые pkl без поля _weights работают корректно —
    predict проверяет наличие _weights и при его отсутствии использует равные веса.

    Следствия z-score нормализации:
        - Модели с большим диапазоном не доминируют над моделями с малым.
        - Модели с константными предсказаниями (std≈0) получают нулевой вклад.
        - Spearman ансамбля ≥ среднего Spearman базовых моделей.

    Единицы predict — условные z-score, не P&L. Ранговый порядок сохранён:
    Spearman и перцентильный порог входа (_optimize_threshold) работают корректно.
    """

    def __init__(self, estimators: list[tuple[str, Any]], n_jobs: int = 1) -> None:
        self.estimators = estimators
        self.n_jobs = n_jobs  # для совместимости с VotingRegressor API
        self._fitted: list[Any] = []
        self._means: list[float] = []
        self._stds: list[float] = []
        # Адаптивные веса: пустой список = равные веса (backward compat. со старыми pkl)
        self._weights: list[float] = []

    @property
    def estimators_(self) -> list[Any]:
        """Список обученных базовых моделей (совместимо с VotingRegressor.estimators_)."""
        return self._fitted

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "RankEnsemble":
        """
        Обучить базовые модели и запомнить статистики их предсказаний на обучающей выборке.

        mean и std сохраняются для z-score нормализации в predict — без утечки
        данных валидационной выборки (нормализация по train-статистике).
        Сбрасывает _weights — вызвать set_weights() после fit для адаптивных весов.
        """
        self._fitted = []
        self._means = []
        self._stds = []
        self._weights = []
        for _name, model in self.estimators:
            model.fit(X, y)
            preds = model.predict(X)
            self._means.append(float(np.mean(preds)))
            std = float(np.std(preds))
            # std≈0 → константные предсказания: вклад в ансамбль обнуляется через std=1.0
            self._stds.append(std if std > 1e-9 else 1.0)
            self._fitted.append(model)
        return self

    def set_weights(self, weights: list[float]) -> None:
        """
        Установить адаптивные веса моделей ансамбля.

        Нормирует переданные веса к сумме=1. Отрицательные значения обнуляются.
        При нулевой сумме (все модели с нулевым или отрицательным Spearman)
        устанавливает равные веса — ансамбль не деградирует.

        Аргументы:
            weights: список весов в том же порядке, что self._fitted.
        """
        arr = np.array(weights, dtype=float)
        arr = np.maximum(arr, 0.0)  # отрицательный Spearman → не штрафуем, но и не используем
        total = float(arr.sum())
        n = len(weights)
        if total < 1e-9:
            # Все модели плохие — равные веса как безопасный фолбэк
            self._weights = [1.0 / n] * n
        else:
            self._weights = (arr / total).tolist()

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Вернуть взвешенное среднее z-score предсказаний базовых моделей.

        Веса берутся из _weights. Если _weights пуст (старые pkl / до вызова set_weights)
        — использует равные веса для backward compatibility.
        """
        z_scores = [
            (model.predict(X) - mean) / std
            for model, mean, std in zip(self._fitted, self._means, self._stds)
        ]
        # Backward compat: старые pkl не имеют _weights или имеют пустой список
        weights = getattr(self, "_weights", [])
        if not weights:
            return np.mean(z_scores, axis=0)
        return np.average(z_scores, axis=0, weights=weights)
