"""
Отбор признаков per-ticker для ML-ансамбля.

Поддерживаемые методы (ML_FEATURE_SELECTION_METHOD):
    "permutation" — OOS Permutation Importance на последнем WalkForward-фолде.
                    Для каждого признака измеряется падение Spearman при перемешивании —
                    напрямую отражает вклад в метрику оптимизации HPO.
                    Вычисляется на val_idx → нет утечки обучающих данных.
    "importance"  — Mean Impurity Importance (legacy): нормализованная importance
                    по трём моделям ансамбля. Быстрее, но нестабильнее, bias к
                    высококардинальным признакам.
    "none"        — отбор отключён, используются все признаки.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr

from config.settings import ml_settings
from utils.logger import logger

if TYPE_CHECKING:
    # Импорт только для type hints — избегаем циклического импорта
    from ml.train import RankEnsemble


def _spearman_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычислить корреляцию Спирмена, подавив предупреждение о константных входах.

    Возвращает 0.0 если корреляция не определена (константный вектор или < 4 элементов).
    """
    if len(y_true) < 4:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if np.isfinite(corr) else 0.0


def _select_by_permutation_importance(
    ensemble: "RankEnsemble",
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int,
    random_state: int,
    top_k: int,
    threshold: float,
) -> list[str]:
    """
    Отбор признаков по OOS Permutation Importance.

    Для каждого признака измеряется падение Spearman-корреляции при перемешивании
    его значений в val-выборке. Признаки с нулевым или отрицательным вкладом
    исключаются.

    Вычисляется на X_val (OOS) → нет утечки данных из обучающей выборки.

    Аргументы:
        ensemble:      обученный ансамбль RankEnsemble.
        X_train:       признаки обучающей выборки (используются только для типов, не для фита).
        y_train:       метки обучающей выборки (не используются напрямую).
        X_val:         признаки val-выборки (OOS) для перемешивания.
        y_val:         метки val-выборки для оценки Spearman.
        n_repeats:     количество повторов перемешивания на признак (среднее по n_repeats).
        random_state:  seed генератора случайных чисел.
        top_k:         если > 0 — ограничить топ-N признаков по importance.
                       Если 0 — не ограничивать, использовать порог.
        threshold:     минимальная нормализованная importance (используется если top_k == 0).

    Возвращает:
        Список отобранных имён признаков (порядок из X_val.columns).
        Фолбэк: все признаки если permutation не смог ничего отобрать.
    """
    feature_names = X_val.columns.tolist()
    rng = np.random.RandomState(random_state)

    y_val_arr = y_val.values if hasattr(y_val, "values") else np.array(y_val)

    # Базовый Spearman на неперемешанных данных
    base_pred = ensemble.predict(X_val)
    base_score = _spearman_score(y_val_arr, base_pred)

    # Если базовый Spearman ≈ 0 — модель не предсказывает, отбор бессмысленен
    if abs(base_score) < 1e-6:
        logger.warning(
            "Базовый Spearman ≈ 0 на val-фолде — используем все признаки",
            base_score=base_score,
        )
        return feature_names

    importances: dict[str, float] = {}
    X_val_arr = X_val.values.copy()
    col_indices = {name: i for i, name in enumerate(feature_names)}

    for feat in feature_names:
        col_idx = col_indices[feat]
        scores_shuffled: list[float] = []

        for _ in range(n_repeats):
            # Перемешиваем только одну колонку — остальные остаются неизменными
            X_permuted = X_val_arr.copy()
            X_permuted[:, col_idx] = rng.permutation(X_permuted[:, col_idx])
            X_perm_df = pd.DataFrame(X_permuted, columns=feature_names, index=X_val.index)
            perm_pred = ensemble.predict(X_perm_df)
            scores_shuffled.append(_spearman_score(y_val_arr, perm_pred))

        # Importance = насколько упал Spearman при перемешивании признака
        mean_shuffled = float(np.mean(scores_shuffled))
        importances[feat] = base_score - mean_shuffled

    # Нормализуем к [0, 1] (делим на максимальную importance) для стабильного порога
    max_imp = max(importances.values()) if importances else 0.0
    if max_imp > 1e-9:
        norm_importances = {f: v / max_imp for f, v in importances.items()}
    else:
        # Все importance нулевые — признаки не влияют на Spearman, берём все
        logger.warning("Все permutation importance нулевые — используем все признаки")
        return feature_names

    # Выбираем признаки по top_k или порогу
    if top_k > 0:
        sorted_feats = sorted(norm_importances.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, _ in sorted_feats[:top_k]]
    else:
        selected = [f for f in feature_names if norm_importances[f] >= threshold]

    # Фолбэк: если порог слишком высокий
    if not selected:
        logger.warning(
            "Permutation importance: ни один признак не прошёл порог — используем все",
            threshold=threshold,
            top_k=top_k,
        )
        return feature_names

    # Сохраняем исходный порядок признаков из feature_names
    selected_set = set(selected)
    return [f for f in feature_names if f in selected_set]


def _select_by_threshold(
    ensemble: "RankEnsemble",
    feature_names: list[str],
    threshold: float,
) -> list[str]:
    """
    Legacy-метод: отбор признаков по нормализованной impurity importance.

    LightGBM считает сплиты (сотни), RF — долю Gini (0–1).
    Каждая модель нормализуется к сумме=1 перед усреднением.
    Используется при ML_FEATURE_SELECTION_METHOD=importance.

    Порядок сохраняется из feature_names.
    Если threshold <= 0 или все признаки ниже порога — возвращает все признаки.
    """
    if threshold <= 0.0:
        return list(feature_names)

    raw_per_model: list[dict[str, float]] = []
    for model in ensemble.estimators_:
        if not hasattr(model, "feature_importances_"):
            continue
        raw = dict(zip(feature_names, model.feature_importances_.astype(float)))
        total = sum(raw.values()) or 1.0
        raw_per_model.append({f: v / total for f, v in raw.items()})

    if not raw_per_model:
        return list(feature_names)

    avg_imp = {
        f: sum(m.get(f, 0.0) for m in raw_per_model) / len(raw_per_model)
        for f in feature_names
    }
    selected = [f for f in feature_names if avg_imp[f] >= threshold]
    return selected if selected else list(feature_names)


def select_features(
    ensemble: "RankEnsemble",
    X: pd.DataFrame,
    y: pd.Series,
    last_train_idx: np.ndarray,
    last_val_idx: np.ndarray,
) -> list[str]:
    """
    Отобрать признаки методом, заданным в ML_FEATURE_SELECTION_METHOD.

    Диспетчер: читает ml_settings и вызывает нужный метод.

    Аргументы:
        ensemble:       обученный ансамбль (фит уже выполнен на X.iloc[last_train_idx]).
        X:              полный DataFrame признаков тикера.
        y:              полный Series меток тикера.
        last_train_idx: индексы последнего train-фолда (WalkForwardSplit).
        last_val_idx:   индексы последнего val-фолда (OOS, без утечки).

    Возвращает:
        Список отобранных признаков.
    """
    method = ml_settings.feature_selection_method
    all_features = X.columns.tolist()
    threshold = ml_settings.feature_importance_threshold

    if method == "none" or (threshold <= 0.0 and method == "importance"):
        return all_features

    if method == "permutation":
        X_train = X.iloc[last_train_idx]
        y_train = y.iloc[last_train_idx]
        X_val = X.iloc[last_val_idx]
        y_val = y.iloc[last_val_idx]
        return _select_by_permutation_importance(
            ensemble=ensemble,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_repeats=ml_settings.permutation_n_repeats,
            random_state=ml_settings.random_state,
            top_k=ml_settings.feature_top_k,
            threshold=threshold,
        )

    if method == "importance":
        return _select_by_threshold(ensemble, all_features, threshold)

    # Неизвестный метод — предупреждаем и берём все признаки
    logger.warning(
        "Неизвестный ML_FEATURE_SELECTION_METHOD — используем все признаки",
        method=method,
    )
    return all_features
