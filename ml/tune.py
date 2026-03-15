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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import TimeSeriesSplit

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
            f"\r    {label:<14} [{bar}] {done:>3}/{n_trials} | best Sortino={best:.4f}",
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
    tax_pct: float = 0.0,
    y_sell: np.ndarray | None = None,
) -> list[float]:
    """
    Симулировать сделки: одна позиция за раз, выход по SELL-сигналу / SL / TP / lookahead.

    Порядок выхода по приоритету (соответствует реальной торговой логике бота):
        1. SL — аварийный стоп (жёсткий, не перебивается SELL-сигналом)
        2. TP — фиксация прибыли (жёсткий)
        3. SELL-сигнал модели — основной выход при нормальной работе
        4. конец окна lookahead — принудительный выход

    P&L рассчитывается как чистый результат после комиссии и НДФЛ:
        gross = (exit - entry) / entry
        commission = 2 × commission_pct  (вход + выход)
        tax = max(0, (gross - commission) × tax_pct)  — только на прибыль
        net = gross - commission - tax

    Аргументы:
        y_pred:         предсказанные классы (0=SELL, 1=HOLD, 2=BUY). Используется для входа.
        close_window:   матрица цен (N, lookahead+1), close_window[i,j] = close[t_i + j].
        commission_pct: комиссия брокера (доля, например 0.003 = 0.3%).
        sl_pct:         стоп-лосс от цены входа (доля, 0.03 = 3%).
        tp_pct:         тейк-профит от цены входа (доля, 0.05 = 5%).
        lookahead:      максимальное число свечей до выхода.
        tax_pct:        ставка НДФЛ на прибыль (0.13 = 13%). 0.0 = не учитывать.
        y_sell:         массив SELL-флагов (1 = SELL-сигнал, 0 = нет) той же длины что y_pred.
                        None = не использовать SELL-выход (только SL/TP/lookahead).

    Возвращает:
        Список чистых P&L на каждую сделку (доля). Пустой список если сделок не было.
    """
    pnl_list: list[float] = []
    next_available = 0
    n = len(y_pred)

    for i in range(n):
        if i < next_available:
            continue
        if int(y_pred[i]) != 2:
            continue

        entry = float(close_window[i, 0])
        if entry <= 0.0:
            continue

        # Цены триггера рассчитаны как NET SL/TP — совпадают с adjusted_sl_price / adjusted_tp_price
        # из trading/profitability.py. Это обеспечивает согласованность симуляции с реальной торговлей:
        #   SL: выход при net_loss = sl_pct  → gross trigger ≈ sl_pct - 2×commission  (меньше sl_pct)
        #   TP: выход при net_profit = tp_pct → gross trigger ≈ tp_pct + 2×commission + tax (больше tp_pct)
        c = commission_pct
        t = tax_pct
        sl_price = entry * (1.0 + c - sl_pct) / (1.0 - c)
        tp_price = entry * ((1.0 + c) + tp_pct / (max(1.0 - t, 1e-9))) / (1.0 - c)
        exit_price = float(close_window[i, lookahead])

        for j in range(1, lookahead + 1):
            c = float(close_window[i, j])
            # SL и TP — жёсткие стопы, имеют приоритет над SELL-сигналом
            if c <= sl_price:
                exit_price = sl_price
                break
            if c >= tp_price:
                exit_price = tp_price
                break
            # SELL-сигнал модели: основной выход, если SL/TP не сработали
            if y_sell is not None and i + j < n and y_sell[i + j]:
                exit_price = c
                break

        gross = (exit_price - entry) / entry
        commission = 2.0 * commission_pct
        tax = max(0.0, (gross - commission) * tax_pct)
        pnl_list.append(gross - commission - tax)
        next_available = i + lookahead

    return pnl_list


def _sortino_score(pnl_list: list[float], min_trades: int) -> float:
    """
    Вычислить Sortino ratio по списку P&L сделок.

    Sortino = mean(P&L) / downside_std — в знаменателе только убыточные сделки.
    В отличие от Sharpe не штрафует за высокую прибыль: случайный большой выигрыш
    не снижает оценку. Для торговой системы с ограниченными потерями (SL) и
    несимметричными выигрышами (TP) теоретически корректнее Sharpe.

    При недостатке сделок (< min_trades) возвращает 0.0 — штраф моделям,
    которые почти не генерируют BUY-сигналы.
    При отсутствии убытков (все сделки прибыльны) возвращает mean(P&L).

    Аргументы:
        pnl_list:   список P&L сделок.
        min_trades: минимальное число сделок для расчёта (иначе 0.0).

    Возвращает:
        Sortino ratio или 0.0.
    """
    if len(pnl_list) < min_trades:
        return 0.0
    arr = np.array(pnl_list)
    mean = float(np.mean(arr))
    losses = arr[arr < 0.0]
    if len(losses) == 0:
        # Все сделки прибыльны — downside не определён, возвращаем mean
        return mean
    downside_std = float(np.std(losses))
    if downside_std < 1e-9:
        return mean
    return mean / downside_std


def _cv_pnl_score(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    close_window: np.ndarray,
    cv: TimeSeriesSplit,
    trial: optuna.trial.Trial | None = None,
) -> float:
    """
    Кросс-валидация с Sortino-метрикой: Sortino по всем сделкам из всех фолдов.

    Собираем все P&L из всех фолдов в один список — это стабильнее чем
    усреднять Sortino по фолдам (в каждом фолде мало сделок → шумно).
    P&L включает комиссию и НДФЛ — точная реплика торговой логики scheduler.

    При передаче trial — включается pruning: после каждого фолда промежуточный
    Sortino репортится в Optuna. Если trial признаётся плохим (ниже медианы
    завершённых trials на том же шаге) — поднимается TrialPruned и фолды
    не досчитываются. Экономия: ~30% времени HPO на отброшенных trials.

    Аргументы:
        model:        sklearn-совместимый классификатор.
        X:            DataFrame признаков.
        y:            Series меток.
        close_window: матрица цен (N, lookahead+1), выровненная с X/y.
        cv:           экземпляр TimeSeriesSplit.
        trial:        Optuna trial для pruning (None = без pruning).

    Возвращает:
        Sortino ratio по всем CV-сделкам. 0.0 если сделок меньше min_trades.
    """
    all_pnl: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        # predict_proba + порог 0.5 вместо argmax для входа в BUY:
        # argmax с balanced классами → BUY на ~33% баров (слишком агрессивно),
        # порог 0.5 → BUY только когда модель уверена сильнее случайного выбора.
        proba = model.fit(X_train, y_train).predict_proba(X_val)
        y_pred = np.where(proba[:, 2] >= 0.5, 2, 1)
        # SELL-сигнал для выхода из позиции: argmax == 0 (SELL — наиболее вероятный класс).
        # Это зеркалит реальную логику бота: scheduler выходит по SELL-сигналу модели,
        # а не ждёт конца lookahead-окна. SL/TP остаются жёсткими стопами с приоритетом.
        y_sell = (np.argmax(proba, axis=1) == 0).astype(np.int8)

        all_pnl.extend(_simulate_pnl(
            y_pred=y_pred,
            close_window=close_window[val_idx],
            commission_pct=trading_settings.broker_commission_pct,
            sl_pct=trading_settings.stop_loss_pct,
            tp_pct=trading_settings.take_profit_pct,
            lookahead=ml_settings.lookahead,
            tax_pct=trading_settings.tax_pct,
            y_sell=y_sell,
        ))

        # Pruning: репортим промежуточный Sortino после каждого фолда.
        # MedianPruner сравнивает с медианой завершённых trials на том же шаге —
        # явно плохие trials останавливаются не дожидаясь последнего фолда.
        if trial is not None:
            intermediate = _sortino_score(all_pnl, min_trades=ml_settings.sharpe_min_trades)
            trial.report(intermediate, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return _sortino_score(all_pnl, min_trades=ml_settings.sharpe_min_trades)


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
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            # subsample_freq > 0 обязателен — иначе subsample игнорируется LightGBM
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            # Регуляризация — ключевые параметры для шумных финансовых данных
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),       # L1: обнуляет слабые признаки
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),     # L2: сглаживает веса
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),  # минимальный прирост для разбиения
            "random_state": ml_settings.random_state,
            "verbose": -1,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        # LightGBM — дерево решений: StandardScaler не нужен (сплиты монотонны к масштабу)
        model = lgb.LGBMClassifier(**params)
        return _cv_pnl_score(model, X, y, close_window, cv, trial=trial)

    # n_startup_trials=20: LightGBM имеет 9 параметров → нужно ~2–3x random trials
    # перед тем как TPE начнёт строить суррогатную модель (дефолт 10 — слишком мало)
    # seed: воспроизводимость HPO между запусками (--force-tune даёт те же результаты)
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=20,
        seed=ml_settings.random_state,
    )
    # MedianPruner: останавливает trial если его промежуточный Sortino
    # хуже медианы завершённых trials на том же фолде.
    # n_startup_trials=5: не прунить пока нет хотя бы 5 завершённых для медианы.
    # n_warmup_steps=2: не прунить до фолда №2 (первые 2 фолда — нужно накопить P&L).
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
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
        "subsample_freq": 1,  # обязателен чтобы subsample реально применялся
        "random_state": ml_settings.random_state,
        "verbose": -1,
        "n_jobs": -1,
        "class_weight": "balanced",
    })

    logger.info(
        "LightGBM tuning complete",
        best_sortino=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_lgbm_{version}.json")
    return best_params


# ── ExtraTrees ────────────────────────────────────────────────────────────────

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
            # log2(8000 строк) ≈ 13 — деревья не вырастают глубже 13-15 на наших данных;
            # max_depth > 18 даёт идентичный результат и тратит trials Optuna впустую
            "max_depth": trial.suggest_int("max_depth", 5, 18),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            # max_features: ET особенно чувствителен к этому параметру
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            # Разрезать узел только если снижение примеси >= порога — отсекает шумные разбиения
            "min_impurity_decrease": trial.suggest_float("min_impurity_decrease", 0.0, 0.01),
            "random_state": ml_settings.random_state,
            "n_jobs": -1,
            "class_weight": "balanced",
        }
        # ExtraTrees — дерево решений: StandardScaler не влияет (сплиты монотонны к масштабу)
        model = ExtraTreesClassifier(**params)
        return _cv_pnl_score(model, X, y, close_window, cv, trial=trial)

    # n_startup_trials=15: ExtraTrees имеет 6 параметров → нужно ~2–3x random trials
    # seed: воспроизводимость HPO
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=15,
        seed=ml_settings.random_state,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
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
        best_sortino=round(study.best_value, 4),
        best_params=best_params,
    )

    _save_params(best_params, f"best_params_et_{version}.json")
    return best_params


# ── Утилита сохранения ───────────────────────────────────────────────────────

def _save_params(params: dict, filename: str) -> None:
    """Сохранить словарь параметров в JSON-файл в директории весов."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / filename
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Best params saved", path=str(path))
