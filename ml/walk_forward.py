"""
Роллинговая кросс-валидация WalkForwardSplit для финансовых временных рядов.

Используется в tune.py (HPO) и train.py (CV-оценка и отбор признаков).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import ml_settings


class WalkForwardSplit:
    """
    Роллинговая кросс-валидация для финансовых временных рядов.

    Отличие от sklearn TimeSeriesSplit (расширяющееся окно):
    - Фиксированный размер обучающего окна (rolling, не expanding).
      Старые данные не участвуют в обучении — рыночный режим актуален.
    - Gap между концом train и началом val = ML_LOOKAHEAD.
      Метки последних барóв train рассчитаны по lookahead свечей вперёд —
      без gap они пересекаются с первыми барами val (утечка меток).
    - Embargo после каждого val-окна = ML_WF_EMBARGO барóв.
      Метки последних барóв val зависят от lookahead барóв за пределами val —
      без embargo эти бары войдут в следующее обучающее окно.

    Фолды строятся слева направо с шагом val_size.
    Берутся последние n_splits фолдов (ближайшие к OOS периоду).

    Fallback: если данных меньше (train_size + gap + val_size) — один фолд
    на всём датасете без gap/embargo (для коротких тикеров / ночного обучения).

    Параметры из .env:
        ML_WF_TRAIN_SIZE  — размер обучающего окна (баров)
        ML_WF_VAL_SIZE    — размер val-окна (баров)
        ML_WF_EMBARGO     — embargo после val (баров), рекомендовано >= ML_LOOKAHEAD
        ML_N_SPLITS       — количество последних фолдов для оценки
    """

    def __init__(
        self,
        train_size: int | None = None,
        val_size: int | None = None,
        gap: int | None = None,
        embargo: int | None = None,
        n_splits: int | None = None,
    ) -> None:
        """
        Инициализировать параметры сплиттера из ml_settings (или переданных значений).

        Аргументы:
            train_size: размер обучающего окна (баров). None = из ML_WF_TRAIN_SIZE.
            val_size:   размер val-окна (баров). None = из ML_WF_VAL_SIZE.
            gap:        пропуск между train и val (баров). None = из ML_LOOKAHEAD.
            embargo:    пропуск после val перед следующим train (баров). None = из ML_WF_EMBARGO.
            n_splits:   максимальное число фолдов. None = из ML_N_SPLITS.
        """
        self.train_size = train_size if train_size is not None else ml_settings.wf_train_size
        self.val_size = val_size if val_size is not None else ml_settings.wf_val_size
        self.gap = gap if gap is not None else ml_settings.lookahead
        self.embargo = embargo if embargo is not None else ml_settings.wf_embargo
        self.n_splits = n_splits if n_splits is not None else ml_settings.n_splits

    def split(self, X: pd.DataFrame | np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Сгенерировать индексы (train_idx, val_idx) для каждого фолда.

        Порядок окон (rolling):
            train: [start, start + train_size)
            gap:   [start + train_size, start + train_size + gap)    — исключено из обоих
            val:   [start + train_size + gap, start + train_size + gap + val_size)
            embargo: [val_end, val_end + embargo)                    — исключено из следующего train

        Шаг между фолдами = val_size + embargo (следующий train начинается после embargo).

        Аргументы:
            X: DataFrame или массив признаков (используется только len(X)).

        Возвращает:
            Список кортежей (train_indices, val_indices) для каждого фолда.
        """
        n = len(X)
        min_len = self.train_size + self.gap + self.val_size

        # Fallback: данных недостаточно для хотя бы одного полного фолда
        if n < min_len:
            # Один фолд: 80/20 split без gap и embargo
            split_at = int(n * 0.8)
            return [(
                np.arange(split_at),
                np.arange(split_at, n),
            )]

        # Строим все возможные фолды слева направо
        # Шаг = val_size + embargo: после каждого val-окна пропускаем embargo баров
        step = self.val_size + self.embargo
        all_folds: list[tuple[np.ndarray, np.ndarray]] = []
        start = 0
        while True:
            val_start = start + self.train_size + self.gap
            val_end = val_start + self.val_size
            if val_end > n:
                break
            train_idx = np.arange(start, start + self.train_size)
            val_idx = np.arange(val_start, val_end)
            all_folds.append((train_idx, val_idx))
            start += step

        # Берём последние n_splits фолдов — ближайшие к реальному OOS периоду
        folds = all_folds[-self.n_splits:] if len(all_folds) > self.n_splits else all_folds

        # Гарантированно возвращаем хотя бы один фолд
        if not folds:
            split_at = int(n * 0.8)
            return [(np.arange(split_at), np.arange(split_at, n))]

        return folds

    def get_n_splits(self, X: pd.DataFrame | np.ndarray | None = None) -> int:
        """Вернуть количество фолдов (для совместимости с sklearn API)."""
        if X is None:
            return self.n_splits
        return len(self.split(X))
