"""
Обучение ансамблей моделей — по одному на каждый тикер.

Для каждого тикера выполняется отдельный pipeline:
    1. Загрузка свечей из PostgreSQL
    2. Вычисление признаков (TA-Lib индикаторы)
    3. Генерация меток по будущей доходности
    4. Подбор гиперпараметров через Optuna (с кешем best_params_*_{ticker}_{version}.json)
    5. Обучение ансамбля RankEnsemble — LightGBM(quantile) + ExtraTrees(MSE) + HistGBM(MAE)
    6. Сохранение весов в ml/weights/ensemble_{ticker}_{version}.pkl

Запуск:
    python -m scripts.train_model               # Optuna только если нет кеша
    python -m scripts.train_model --force-tune  # Принудительный повтор Optuna

Все настройки — в .env (ML_* и DATA_*).
"""
from __future__ import annotations

import gc
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from config.settings import data_settings, ml_settings
from ml.dataset import load_all_tickers_from_csv, load_ticker_data, load_usdrub_data, merge_usdrub
from ml.ensemble_weights import compute_ensemble_weights as _compute_ensemble_weights
from ml.features import FEATURE_COLUMNS, compute_features
from ml.labels import compute_pnl_targets
from ml.feature_selection import select_features
from ml.tune import WalkForwardSplit, tune_extra_trees, tune_hist_gbm, tune_lgbm
from utils.logger import logger

WEIGHTS_DIR = Path(__file__).parent / "weights"

_SEP = "=" * 56


# ── Вывод прогресса в терминал ───────────────────────────────────────────────

def _print_ticker_header(ticker: str, index: int, total: int, samples: int) -> None:
    """Вывести заголовок секции обучения тикера."""
    print(f"\n{_SEP}", flush=True)
    print(f"  [{index}/{total}] {ticker}  ({samples} строк)", flush=True)
    print(_SEP, flush=True)


def _print_step(msg: str) -> None:
    """Вывести шаг (без переноса строки — ожидается OK или новая строка)."""
    print(f"  > {msg}", end="", flush=True)


def _print_cached(label: str) -> None:
    """Вывести строку о загрузке из кеша."""
    print(f"    {label:<14} [из кеша]", flush=True)


def _print_ok(extra: str = "") -> None:
    """Вывести ✓ после _print_step."""
    suffix = f"  {extra}" if extra else ""
    print(f" ok{suffix}", flush=True)


def _print_summary(results: dict[str, Path], failed: list[str]) -> None:
    """Вывести итоговую сводку."""
    print(f"\n{_SEP}", flush=True)
    print(f"  ИТОГ: {len(results)}/{len(results) + len(failed)} тикеров обучено", flush=True)
    for ticker, path in results.items():
        print(f"    + {ticker:<8} -> {path.name}", flush=True)
    for ticker in failed:
        print(f"    x {ticker:<8} -- error (see log)", flush=True)
    print(_SEP, flush=True)


# ── Кеш гиперпараметров ──────────────────────────────────────────────────────

def _load_cached_params(model_name: str, ticker_version: str) -> dict | None:
    """
    Загрузить сохранённые гиперпараметры из JSON-файла, если он существует.

    Аргументы:
        model_name:     имя модели ("lgbm", "xgboost", "rf").
        ticker_version: строка вида "{ticker}_{version}" (например, "SBER_v2").

    Возвращает:
        Словарь параметров или None если кеша нет или параметры устарели.
    """
    path = WEIGHTS_DIR / f"best_params_{model_name}_{ticker_version}.json"
    if not path.exists():
        return None
    with open(path) as f:
        params = json.load(f)
    # Валидация совместимости: старые параметры не подходят текущей архитектуре.
    # objective изменён с "regression_l1" на "quantile" — alpha даёт лучшую разделимость.
    if model_name == "lgbm" and params.get("objective") != "quantile":
        logger.info("Кеш LGBM устарел (→quantile objective), переобучаем", path=str(path))
        return None
    # max_depth добавлен в диапазон Optuna — старые кеши без него нужно переоптимизировать.
    if model_name == "lgbm" and "max_depth" not in params:
        logger.info("Кеш LGBM устарел (нет max_depth), переобучаем", path=str(path))
        return None
    # alpha теперь тюнируется Optuna [0.7, 0.95] — кеши с alpha вне диапазона
    # (сгенерированные по positive_rate) могут содержать значения вне этого диапазона,
    # но это безопасно переиспользовать. Инвалидируем только если alpha отсутствует.
    if model_name == "lgbm" and "alpha" not in params:
        logger.info("Кеш LGBM устарел (нет alpha), переобучаем", path=str(path))
        return None
    # ET: n_jobs=-1 вызывает OOM на 2GB сервере — форкает копию датасета на каждое ядро.
    if model_name == "et" and params.get("n_jobs", 1) == -1:
        logger.info("Кеш ET устарел (n_jobs=-1 → OOM), переобучаем", path=str(path))
        return None
    # HistGBM: loss должен быть "absolute_error" — старые кеши без этого поля невалидны.
    if model_name == "hist_gbm" and params.get("loss") != "absolute_error":
        logger.info("Кеш HistGBM устарел (нет/другой loss), переобучаем", path=str(path))
        return None
    logger.info("Загружены кешированные параметры", model=model_name, path=str(path))
    return params


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
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Вычислить признаки, метки и матрицу цен для одного тикера.

    Аргументы:
        group:  DataFrame свечей одного тикера [time, open, high, low, close, volume].
        ticker: тикер (только для логирования).

    Возвращает:
        (X, y, close_window):
            X            — DataFrame признаков,
            y            — Series меток,
            close_window — numpy array (N, lookahead+1):
                           close_window[i, 0] = close[t_i] (цена входа),
                           close_window[i, j] = close[t_i + j], j=1..lookahead.
                           Используется для оптимизации порога входа per-ticker (_optimize_threshold).
    """
    from config.settings import trading_settings as ts

    feat_df = compute_features(group)

    # Строим матрицу close-цен: close_window[i, j] = close[t_i + j], j=0..lookahead.
    # Сначала отфильтровываем строки, для которых lookahead свечей вперёд недоступны
    # (последние lookahead строк feat_df), — иначе close_arr[pos + lookahead] вышел бы за границу.
    lookahead = ml_settings.lookahead
    close_arr = group["close"].values.astype(float)
    group_index_map: dict = {t: pos for pos, t in enumerate(group.index)}

    valid_mask = np.array(
        [group_index_map[t] + lookahead < len(group) for t in feat_df.index],
        dtype=bool,
    )
    feat_df = feat_df[valid_mask].copy()

    entry_positions = np.array([group_index_map[t] for t in feat_df.index], dtype=int)
    close_window = np.stack(
        [close_arr[entry_positions + j] for j in range(lookahead + 1)],
        axis=1,
    )  # shape: (N, lookahead+1)

    # Целевая переменная — непрерывный net P&L для регрессии.
    # Регрессор предсказывает ожидаемый P&L каждого возможного входа (BUY here),
    # что устраняет дискретизацию непрерывного P&L в 3 класса и проблему дисбаланса.
    targets = compute_pnl_targets(
        index=feat_df.index,
        close_window=close_window,
        lookahead=lookahead,
        commission_pct=ts.broker_commission_pct,
        sl_pct=ts.stop_loss_pct,
        tp_pct=ts.take_profit_pct,
        tax_pct=ts.tax_pct,
    )

    logger.info(
        "Датасет тикера собран",
        ticker=ticker,
        samples=len(targets),
        pnl_positive=int((targets > 0).sum()),
        pnl_negative=int((targets < 0).sum()),
        pnl_mean=round(float(targets.mean()), 5),
    )

    return feat_df[FEATURE_COLUMNS], targets, close_window


# ── Оценка ансамбля ──────────────────────────────────────────────────────────

def _evaluate_ensemble(
    ensemble: RankEnsemble,
    X: pd.DataFrame,
    y: pd.Series,
    ticker: str,
) -> float:
    """
    Оценить ансамбль через WalkForwardSplit с метрикой Спирмена.

    Возвращает:
        Средняя корреляция Спирмена по фолдам ([-1, 1]).
    """
    from ml.tune import _cv_spearman_score
    cv = WalkForwardSplit()
    spearman = _cv_spearman_score(ensemble, X, y, cv)
    logger.info(
        "Оценка ансамбля (CV Spearman)",
        ticker=ticker,
        spearman=round(spearman, 4),
    )
    return spearman


# ── Ансамбль с z-score нормализацией ─────────────────────────────────────────

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


# ── Feature importance (вывод для отладки) ────────────────────────────────────

def _avg_importance_for_print(
    ensemble: RankEnsemble,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Вычислить нормализованную impurity importance для вывода таблицы (только для ML_PRINT_FEATURE_IMPORTANCE).

    Нормализует каждую модель к сумме=1 и усредняет — для наглядного вывода в консоль.
    Не используется при отборе признаков (метод задаётся ML_FEATURE_SELECTION_METHOD).
    """
    raw_per_model: list[dict[str, float]] = []
    for model in ensemble.estimators_:
        if not hasattr(model, "feature_importances_"):
            continue
        raw = dict(zip(feature_names, model.feature_importances_.astype(float)))
        total = sum(raw.values()) or 1.0
        raw_per_model.append({f: v / total for f, v in raw.items()})

    if not raw_per_model:
        return {f: 0.0 for f in feature_names}

    return {
        f: sum(m.get(f, 0.0) for m in raw_per_model) / len(raw_per_model)
        for f in feature_names
    }


def _print_feature_importance(
    ensemble: RankEnsemble,
    feature_names: list[str],
    ticker: str,
) -> None:
    """Вывести все признаки по убыванию нормализованной importance."""
    avg_imp = _avg_importance_for_print(ensemble, feature_names)
    sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

    max_imp = sorted_feats[0][1] if sorted_feats else 1.0
    print(f"\n  Признаки [{ticker}] по важности:", flush=True)
    for i, (feat, imp) in enumerate(sorted_feats, 1):
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {i:>2}. {feat:<25} {bar} {imp:.4f}", flush=True)


# ── Оптимизация порога уверенности per-ticker ────────────────────────────────

def _optimize_threshold(
    ensemble: RankEnsemble,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    close_window_val: np.ndarray,
    ticker: str,
) -> float:
    """
    Подобрать оптимальный порог входа в сделку для тикера через Optuna.

    Порог применяется в scheduler: если predicted_pnl < threshold → BUY игнорируется.
    Оптимизация на последних 20% данных максимизирует Sortino ratio сделок.

    Аргументы:
        ensemble:         обученный ансамбль RankEnsemble.
        X_val:            признаки валидационной выборки (последние 20% тикера).
        y_val:            целевые P&L (не используются — симуляция работает с close_window).
        close_window_val: матрица цен (N_val, lookahead+1) для P&L-симуляции.
        ticker:           тикер инструмента (для логирования).

    Возвращает:
        Оптимальный порог в пространстве предсказанного P&L (диапазон [0.0, 0.02]).
    """
    from ml.tune import _simulate_pnl, _sortino_score
    from config.settings import trading_settings as ts

    # Регрессор предсказывает ожидаемый net P&L для каждого бара.
    pnl_pred = ensemble.predict(X_val)
    y_sell = (pnl_pred < 0).astype(np.int8)

    def objective(trial: optuna.Trial) -> float:
        # Ищем в персентильном пространстве: trial подбирает P — какой процент баров
        # считать "хорошими" (топ-(100-P)% предсказаний → BUY).
        # Абсолютный порог [0.0, 0.02] бесполезен: все предсказания отрицательны при
        # 88% отрицательных таргетах, и ни один бар не получает BUY-сигнал.
        signal_pct = trial.suggest_float("signal_pct", 70.0, 95.0)
        threshold_abs = float(np.percentile(pnl_pred, signal_pct))
        preds = np.where(pnl_pred >= threshold_abs, 2, 1).astype(np.int8)
        pnl_list = _simulate_pnl(
            y_pred=preds,
            close_window=close_window_val,
            commission_pct=ts.broker_commission_pct,
            sl_pct=ts.stop_loss_pct,
            tp_pct=ts.take_profit_pct,
            lookahead=ml_settings.lookahead,
            tax_pct=ts.tax_pct,
            y_sell=y_sell,
        )
        return _sortino_score(pnl_list, min_trades=ml_settings.sharpe_min_trades)

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5,
        seed=ml_settings.random_state,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=ml_settings.threshold_n_trials, show_progress_bar=False)

    # Переводим персентиль → абсолютное значение pnl_pred для использования в продакшне.
    # predict_signal и scheduler сравнивают pnl_pred >= threshold напрямую (не по персентилю).
    best_pct = study.best_params["signal_pct"]
    best = float(np.percentile(pnl_pred, best_pct))
    logger.info(
        "Порог входа оптимизирован",
        ticker=ticker,
        signal_pct=round(best_pct, 1),
        threshold=round(best, 6),
        sortino=round(study.best_value, 5),
    )
    return best


# ── Обучение одного тикера ───────────────────────────────────────────────────

def _train_single_ticker(
    ticker: str,
    group: pd.DataFrame,
    force_tune: bool,
    skip_cv: bool = False,
) -> tuple[Path, float]:
    """
    Полный pipeline обучения для одного тикера.

    Аргументы:
        ticker:     тикер инструмента.
        group:      DataFrame свечей этого тикера.
        force_tune: принудительно запустить Optuna.
        skip_cv:    пропустить CV-оценку (для ночного переобучения — экономит RAM и время).

    Возвращает:
        (Path к сохранённому pkl ансамбля, F1 из CV или 0.0 если skip_cv=True).
    """
    version = ml_settings.model_version
    ticker_version = f"{ticker}_{version}"

    X, y, close_window = _build_ticker_dataset(group, ticker)

    if len(X) < ml_settings.n_splits * 2:
        raise RuntimeError(
            f"Недостаточно данных для {ticker}: {len(X)} строк. "
            "Соберите больше свечей."
        )

    # ── HPO на всех признаках ────────────────────────────────────────────────
    # HPO работает на полном X (все 54 признака): деревья сами игнорируют слабые
    # признаки через регуляризацию (min_split_gain, min_impurity_decrease, l2).
    # Отбор признаков выполняется после — по importance уже обученных моделей.
    def _get_with_display(model_name: str, tune_fn) -> dict:
        cached = None if force_tune else _load_cached_params(model_name, ticker_version)
        if cached is not None:
            _print_cached(f"{model_name.upper()} HPO")
            return cached
        return _get_params(model_name, ticker_version, tune_fn, X, y, force_tune)

    lgbm_params     = _get_with_display("lgbm", tune_lgbm)
    et_params       = _get_with_display("et", tune_extra_trees)
    hist_gbm_params = _get_with_display("hist_gbm", tune_hist_gbm)

    def _make_ensemble() -> RankEnsemble:
        return RankEnsemble(
            estimators=[
                ("lgbm", lgb.LGBMRegressor(**lgbm_params)),
                ("et", ExtraTreesRegressor(**et_params)),
                ("hist_gbm", HistGradientBoostingRegressor(**hist_gbm_params)),
            ],
            n_jobs=1,
        )

    all_features = X.columns.tolist()
    method = ml_settings.feature_selection_method
    threshold = ml_settings.feature_importance_threshold

    # ── Проход 1: отбор признаков ─────────────────────────────────────────────
    # При method="permutation": зондовый ансамбль обучается на последнем train-фолде,
    # permutation importance вычисляется на последнем val-фолде (OOS) → нет утечки.
    # При method="importance": зондовый фит на полных данных, нормализованная impurity importance.
    # При method="none" или threshold=0: отбор пропускается.
    features_path = WEIGHTS_DIR / f"features_{ticker_version}.json"
    do_selection = method != "none" and (threshold > 0.0 or ml_settings.feature_top_k > 0)

    if do_selection:
        cached_features: list[str] | None = None
        if not force_tune and features_path.exists():
            try:
                with open(features_path) as f:
                    cached_features = json.load(f)
                cached_features = [c for c in cached_features if c in all_features]
            except Exception:
                cached_features = None

        if cached_features is not None:
            _print_cached("Отбор признаков")
            selected_features = cached_features
        else:
            _print_step(
                f"Отбор признаков ({method}, проход 1 из 2)..."
            )
            # Получаем последний WalkForward-фолд для OOS-оценки при permutation
            wf_splits = list(WalkForwardSplit().split(X))
            last_train_idx, last_val_idx = wf_splits[-1]

            # Зондовый ансамбль: при permutation — только на train-фолде;
            # при importance — на всех данных (как раньше, допустима "утечка" для отбора)
            probe = _make_ensemble()
            if method == "permutation":
                probe.fit(X.iloc[last_train_idx], y.iloc[last_train_idx])
            else:
                probe.fit(X, y)

            selected_features = select_features(
                ensemble=probe,
                X=X,
                y=y,
                last_train_idx=last_train_idx,
                last_val_idx=last_val_idx,
            )
            dropped = len(all_features) - len(selected_features)
            _print_ok(f"{len(selected_features)} из {len(all_features)} признаков (-{dropped})")
            del probe
            gc.collect()
        X_final = X[selected_features]
    else:
        selected_features = all_features
        X_final = X

    # ── Проход 2: CV (опционально) + финальный фит на отобранных признаках ──
    ensemble = _make_ensemble()

    if skip_cv:
        print("    CV оценка   [пропущена — skip_cv]", flush=True)
        cv_f1 = 0.0
    else:
        _print_step("CV оценка ансамбля...")
        cv_f1 = _evaluate_ensemble(ensemble, X_final, y, ticker)
        _print_ok(f"Spearman={cv_f1:.4f}")

    # ── Оптимизация порога уверенности per-ticker ─────────────────────────────
    # Порог оптимизируется ДО финального фита: временный ансамбль обучается на
    # первых 80% данных, holdout (последние 20%) модель ещё не видела → нет утечки.
    # Финальный ансамбль затем переобучается на всех данных.
    _print_step("Оптимизация порога уверенности (Optuna)...")
    val_size = max(ml_settings.sharpe_min_trades * 2, int(len(X_final) * 0.2))
    X_th_train = X_final.iloc[:-val_size]
    y_th_train = y.iloc[:-val_size]
    X_th_val   = X_final.iloc[-val_size:]
    y_th_val   = y.values[-val_size:]
    cw_val     = close_window[-val_size:]
    th_ensemble = _make_ensemble()
    th_ensemble.fit(X_th_train, y_th_train)
    best_threshold = _optimize_threshold(th_ensemble, X_th_val, y_th_val, cw_val, ticker)
    del th_ensemble
    gc.collect()
    _print_ok(f"threshold={best_threshold:.4f}")

    _print_step("Финальное обучение ансамбля...")
    ensemble.fit(X_final, y)
    _print_ok()

    # ── Адаптивные веса: Spearman каждой модели на последнем OOS-фолде ────────
    # Вычисляется ПОСЛЕ финального fit, но каждая базовая модель клонируется
    # и обучается заново на train_idx фолда — нет утечки финального фита.
    _print_step("Вычисление адаптивных весов ансамбля (OOS Spearman)...")
    try:
        adaptive_weights = _compute_ensemble_weights(ensemble, X_final, y)
        ensemble.set_weights(adaptive_weights)
        weights_str = ", ".join(f"{w:.3f}" for w in adaptive_weights)
        _print_ok(f"weights=[{weights_str}]")
    except Exception as exc:
        logger.warning("Ошибка вычисления весов — используем равные", error=str(exc))
        print(f" warn: {exc}", flush=True)

    if ml_settings.print_feature_importance:
        _print_feature_importance(ensemble, selected_features, ticker)

    threshold_path = WEIGHTS_DIR / f"best_threshold_{ticker_version}.json"
    with open(threshold_path, "w") as f:
        json.dump({"ticker": ticker, "threshold": round(best_threshold, 4)}, f)

    ensemble_path = WEIGHTS_DIR / f"ensemble_{ticker_version}.pkl"

    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    # Сохраняем per-ticker список признаков (не глобальный FEATURE_COLUMNS).
    # predict.py загружает именно этот файл → инференс автоматически использует
    # тикерный набор без каких-либо изменений в коде инференса.
    with open(features_path, "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"  Сохранено: {ensemble_path.name}", flush=True)

    return ensemble_path, cv_f1


# ── Основная функция обучения ────────────────────────────────────────────────

async def train_model(
    force_tune: bool = False,
    data_dir: Optional[Path] = None,
    skip_cv: bool = False,
) -> dict[str, Path]:
    """
    Обучить отдельный ансамбль для каждого тикера из DATA_TICKERS.

    Для каждого тикера свой Optuna HPO (с кешем), свои веса.
    Файлы: ensemble_{ticker}_{version}.pkl, features_{ticker}_{version}.json.

    Аргументы:
        force_tune: если True — принудительно запустить Optuna для всех моделей.
        data_dir:   папка с CSV-файлами (для Colab без PostgreSQL).
                    Если None — данные загружаются из PostgreSQL.
        skip_cv:    пропустить CV-оценку (флаг --skip-cv для ночного переобучения).

    Возвращает:
        Словарь {ticker: Path} для успешно обученных тикеров.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if force_tune:
        logger.info("Режим force_tune: Optuna будет запущена для всех моделей")

    tickers_list = data_settings.tickers
    total = len(tickers_list)
    results: dict[str, Path] = {}
    spearman_scores: dict[str, float] = {}

    if data_dir is not None:
        # Режим CSV (Colab): загружаем всё сразу — файлы уже на диске
        logger.info("Режим CSV: загрузка из папки", data_dir=str(data_dir))
        raw = load_all_tickers_from_csv(
            data_dir,
            tickers=tickers_list,
            interval=data_settings.candle_interval,
        )
        if raw.empty:
            raise RuntimeError("Нет данных в CSV. Запустите scripts/export_candles_csv.py.")
        ticker_groups = {t: g.reset_index(drop=True) for t, g in raw.groupby("ticker", sort=False)}
        del raw
        gc.collect()
    else:
        # Режим PostgreSQL: загружаем тикеры по одному чтобы не держать всё в RAM
        logger.info("Загрузка свечей из PostgreSQL по одному тикеру...")
        ticker_groups = None  # будем загружать в цикле

    # Загружаем USD/RUB один раз — нужен для merge в режиме PostgreSQL
    usdrub_df: pd.DataFrame | None = None
    if data_dir is None:
        usdrub_df = await load_usdrub_data(data_settings.candle_interval)
        if usdrub_df.empty:
            logger.warning("USD/RUB данные не найдены — признак usdrub будет нулевым")

    for i, ticker in enumerate(tickers_list, 1):
        if ticker_groups is not None:
            # CSV-режим: группа уже в памяти
            group = ticker_groups.get(ticker)
            if group is None:
                logger.warning("Нет данных для тикера в CSV, пропускаем", ticker=ticker)
                continue
        else:
            # PostgreSQL-режим: загружаем только этот тикер
            df = await load_ticker_data(ticker, data_settings.candle_interval)
            if df.empty:
                logger.warning("Нет данных для тикера в БД, пропускаем", ticker=ticker)
                continue
            group = merge_usdrub(df, usdrub_df)
            group.insert(0, "ticker", ticker)
            group = group.reset_index(drop=True)
            del df

        _print_ticker_header(ticker, i, total, len(group))
        logger.info("Обучение модели", ticker=ticker)
        try:
            path, cv_spearman = _train_single_ticker(ticker, group, force_tune, skip_cv=skip_cv)
            results[ticker] = path
            spearman_scores[ticker] = round(cv_spearman, 4)
        except Exception as e:
            logger.error("Ошибка обучения тикера", ticker=ticker, error=str(e))
            print(f"  x Error: {e}", flush=True)
        finally:
            del group
            gc.collect()

    failed = [t for t in data_settings.tickers if t not in results]
    _print_summary(results, failed)

    # Сохраняем метрики последнего обучения для быстрого просмотра
    results_path = WEIGHTS_DIR / "last_results.json"
    prev_path = WEIGHTS_DIR / "last_results_prev.json"
    # Перед перезаписью сохраняем предыдущий результат для сравнения дельты в боте
    if results_path.exists():
        import shutil
        shutil.copy2(results_path, prev_path)
    # При skip_cv Spearman не вычислялся — берём сохранённые значения из предыдущего обучения
    if skip_cv and prev_path.exists():
        try:
            with open(prev_path) as f_prev:
                prev_data = json.load(f_prev)
            prev_spearman = (
                prev_data.get("spearman_scores")
                or prev_data.get("sortino_scores")
                or {}
            )
            for ticker in spearman_scores:
                if spearman_scores[ticker] == 0.0 and ticker in prev_spearman:
                    spearman_scores[ticker] = prev_spearman[ticker]
        except Exception:
            pass
    with open(results_path, "w") as f:
        json.dump(
            {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "force_tune": force_tune,
                "skip_cv": skip_cv,
                "spearman_scores": spearman_scores,
                "failed": failed,
            },
            f,
            indent=2,
        )
    logger.info("Метрики сохранены", path=str(results_path))

    logger.info(
        "Обучение завершено",
        trained=list(results.keys()),
        failed=failed,
    )

    return results
