"""
Обучение ансамблей моделей — по одному на каждый тикер.

Для каждого тикера выполняется отдельный pipeline:
    1. Загрузка свечей из PostgreSQL
    2. Вычисление признаков (TA-Lib индикаторы)
    3. Генерация меток по будущей доходности
    4. Подбор гиперпараметров через Optuna (с кешем best_params_*_{ticker}_{version}.json)
    5. Обучение ансамбля VotingClassifier(soft) — LightGBM + ExtraTrees
    6. Сохранение весов в ml/weights/ensemble_{ticker}_{version}.pkl

Запуск:
    python -m scripts.train_model               # Optuna только если нет кеша
    python -m scripts.train_model --force-tune  # Принудительный повтор Optuna

Все настройки — в .env (ML_* и DATA_*).
"""
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
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config.settings import data_settings, ml_settings
from ml.dataset import load_all_tickers_from_csv, load_ticker_data, load_usdrub_data, merge_usdrub
from ml.features import FEATURE_COLUMNS, compute_features
from ml.labels import LABEL_NAMES, create_labels_sim
from ml.tune import tune_extra_trees, tune_lgbm
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
        Словарь параметров или None если кеша нет.
    """
    path = WEIGHTS_DIR / f"best_params_{model_name}_{ticker_version}.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info("Загружены кешированные параметры", model=model_name, path=str(path))
        return params
    return None


def _get_params(
    model_name: str,
    ticker_version: str,
    tune_fn,
    X: pd.DataFrame,
    y: pd.Series,
    close_window: np.ndarray,
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
        close_window:   матрица цен (N, lookahead+1) для P&L-метрики Optuna.
        force_tune:     если True — игнорировать кеш и запустить Optuna заново.

    Возвращает:
        Словарь гиперпараметров.
    """
    if not force_tune:
        cached = _load_cached_params(model_name, ticker_version)
        if cached is not None:
            return cached

    logger.info("Запуск Optuna HPO", model=model_name, ticker_version=ticker_version)
    return tune_fn(X, y, close_window, version=ticker_version)


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
                           Используется для P&L-симуляции при HPO и оптимизации порога.
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

    # Генерируем метки на основе SL/TP-симуляции — цель обучения совпадает с метрикой HPO.
    # BUY  если чистый P&L > threshold (выгодный вход),
    # SELL если чистый P&L < -threshold (невыгодный вход),
    # HOLD иначе.
    labels = create_labels_sim(
        index=feat_df.index,
        close_window=close_window,
        lookahead=lookahead,
        threshold=ml_settings.threshold,
        commission_pct=ts.broker_commission_pct,
        sl_pct=ts.stop_loss_pct,
        tp_pct=ts.take_profit_pct,
        tax_pct=ts.tax_pct,
    )

    logger.info(
        "Датасет тикера собран",
        ticker=ticker,
        samples=len(labels),
        buy=int((labels == 2).sum()),
        hold=int((labels == 1).sum()),
        sell=int((labels == 0).sum()),
    )

    return feat_df[FEATURE_COLUMNS], labels, close_window


# ── Оценка ансамбля ──────────────────────────────────────────────────────────

def _evaluate_ensemble(
    ensemble: VotingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    close_window: np.ndarray,
    ticker: str,
) -> float:
    """
    Оценить ансамбль через кросс-валидацию TimeSeriesSplit с P&L-метрикой.

    Возвращает:
        Средний P&L на сделку по фолдам (доля от вложений).
        Для отображения в боте конвертируется в псевдо-F1 (сдвиг + масштаб).
    """
    from ml.tune import _cv_pnl_score
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)
    mean_pnl = _cv_pnl_score(ensemble, X, y, close_window, cv)
    logger.info(
        "Оценка ансамбля (CV P&L)",
        ticker=ticker,
        mean_pnl=round(mean_pnl, 5),
    )
    return mean_pnl


# ── Feature importance ───────────────────────────────────────────────────────

def _avg_importance(
    ensemble: VotingClassifier,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Вычислить нормализованную важность признаков — среднее по всем моделям ансамбля.

    LightGBM считает сплиты (сотни), RF — долю Gini (0–1).
    Каждая модель нормализуется к сумме=1 перед усреднением,
    чтобы LightGBM не доминировал из-за больших абсолютных значений.

    Возвращает:
        Словарь {feature_name: avg_importance}.
    """
    raw_per_model: list[dict[str, float]] = []
    for fitted_pipeline in ensemble.estimators_:
        model = fitted_pipeline.named_steps["model"]
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


def _select_by_threshold(
    ensemble: VotingClassifier,
    feature_names: list[str],
    threshold: float,
) -> list[str]:
    """
    Выбрать признаки с нормализованной importance >= threshold.

    Порядок сохраняется из feature_names (по убыванию importance внутри).
    Если threshold <= 0 или все признаки ниже порога — вернуть все признаки.
    """
    if threshold <= 0.0:
        return list(feature_names)

    avg_imp = _avg_importance(ensemble, feature_names)
    selected = [f for f in feature_names if avg_imp[f] >= threshold]
    # Фолбэк: если порог слишком высокий и отфильтровал всё — берём все
    return selected if selected else list(feature_names)


def _print_feature_importance(
    ensemble: VotingClassifier,
    feature_names: list[str],
    ticker: str,
) -> None:
    """Вывести все признаки по убыванию нормализованной importance."""
    avg_imp = _avg_importance(ensemble, feature_names)
    sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

    max_imp = sorted_feats[0][1] if sorted_feats else 1.0
    print(f"\n  Признаки [{ticker}] по важности:", flush=True)
    for i, (feat, imp) in enumerate(sorted_feats, 1):
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {i:>2}. {feat:<25} {bar} {imp:.4f}", flush=True)


# ── Оптимизация порога уверенности per-ticker ────────────────────────────────

def _optimize_threshold(
    ensemble: VotingClassifier,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    close_window_val: np.ndarray,
    ticker: str,
) -> float:
    """
    Подобрать оптимальный порог уверенности для тикера через Optuna.

    Порог применяется в scheduler: если max(proba) < threshold → сигнал игнорируется.
    Оптимизация на последних 20% данных максимизирует средний P&L на сделку.

    Аргументы:
        ensemble:         обученный ансамбль VotingClassifier.
        X_val:            признаки валидационной выборки (последние 20% тикера).
        y_val:            метки валидационной выборки.
        close_window_val: матрица цен (N_val, lookahead+1) для P&L-симуляции.
        ticker:           тикер инструмента (для логирования).

    Возвращает:
        Оптимальный порог уверенности в диапазоне [0.3, 0.9].
    """
    from ml.tune import _simulate_pnl, _sortino_score
    from config.settings import trading_settings as ts

    proba = ensemble.predict_proba(X_val)  # shape: (n, 3)

    def objective(trial: optuna.Trial) -> float:
        threshold = trial.suggest_float("threshold", 0.3, 0.9)
        preds = np.where(
            proba.max(axis=1) >= threshold,
            proba.argmax(axis=1),
            1,  # HOLD
        )
        pnl_list = _simulate_pnl(
            y_pred=preds,
            close_window=close_window_val,
            commission_pct=ts.broker_commission_pct,
            sl_pct=ts.stop_loss_pct,
            tp_pct=ts.take_profit_pct,
            lookahead=ml_settings.lookahead,
            tax_pct=ts.tax_pct,
        )
        return _sortino_score(pnl_list, min_trades=ml_settings.sharpe_min_trades)

    study = optuna.create_study(direction="maximize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=ml_settings.threshold_n_trials, show_progress_bar=False)

    best = study.best_params["threshold"]
    logger.info(
        "Порог уверенности оптимизирован",
        ticker=ticker,
        threshold=round(best, 4),
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

    # Гиперпараметры: из кеша или через Optuna (с прогрессом в терминале)
    def _get_with_display(model_name: str, tune_fn) -> dict:
        cached = None if force_tune else _load_cached_params(model_name, ticker_version)
        if cached is not None:
            _print_cached(f"{model_name.upper()} HPO")
            return cached
        return _get_params(model_name, ticker_version, tune_fn, X, y, close_window, force_tune)

    lgbm_params = _get_with_display("lgbm", tune_lgbm)
    et_params   = _get_with_display("et", tune_extra_trees)

    # Балансировка классов — фиксированный параметр, не из HPO-кеша.
    # Без него модели выучивают "всегда HOLD" → F1_macro ≈ 0.33 (случайный уровень).
    lgbm_params["class_weight"] = "balanced"
    et_params["class_weight"]   = "balanced"

    # Каждая базовая модель обёрнута в Pipeline со StandardScaler.
    # VotingClassifier(voting='soft') усредняет предсказанные вероятности базовых моделей.
    # Два принципиально разных алгоритма:
    #   LGBM       — градиентный бустинг деревьев
    #   ExtraTrees — рандомизированные деревья (случайные пороги), низкая корреляция с LGBM
    # SVC был исключён: его F1=0.3831 < F1 деревьев (~0.405), ансамбль стал хуже обеих моделей.
    # StackingClassifier не подходит для временных рядов: его внутренний cv=StratifiedKFold
    # перемешивает данные → утечка будущего в OOF → мета-модель деградирует на TimeSeriesSplit.
    def _make_ensemble() -> VotingClassifier:
        def _scaled(name: str, model) -> tuple:
            return (name, Pipeline([("scaler", StandardScaler()), ("model", model)]))

        return VotingClassifier(
            estimators=[
                _scaled("lgbm", lgb.LGBMClassifier(**lgbm_params)),
                _scaled("et", ExtraTreesClassifier(**et_params)),
            ],
            voting="soft",
            n_jobs=1,  # последовательный фит — меньше RAM на слабом сервере
        )

    all_features = X.columns.tolist()
    threshold = ml_settings.feature_importance_threshold

    # ── Проход 1: быстрый фит на всех признаках → отбор по порогу per-ticker ──
    # Один фит (без CV) чтобы получить feature importance и отбросить слабые.
    # Для каждого тикера свой порог отбора — разные признаки могут быть важны
    # для банков (SBER), нефтяников (LKOH), IT-компаний (YDEX) и т.д.
    # CV-оценка и финальный фит выполняются уже на отобранных признаках.
    # Кеш признаков: если features_{ticker_version}.json уже есть и force_tune=False
    # — пропускаем первый фит, загружаем готовый список.
    features_path = WEIGHTS_DIR / f"features_{ticker_version}.json"
    if threshold > 0.0:
        cached_features: list[str] | None = None
        if not force_tune and features_path.exists():
            try:
                with open(features_path) as f:
                    cached_features = json.load(f)
                # Проверяем что кешированные признаки — подмножество текущих
                cached_features = [c for c in cached_features if c in all_features]
            except Exception:
                cached_features = None

        if cached_features is not None:
            _print_cached("Отбор признаков")
            selected_features = cached_features
        else:
            _print_step(f"Отбор признаков (проход 1 из 2, порог >={threshold})...")
            probe = _make_ensemble()
            probe.fit(X, y)
            selected_features = _select_by_threshold(probe, all_features, threshold)
            dropped = len(all_features) - len(selected_features)
            _print_ok(f"{len(selected_features)} из {len(all_features)} признаков (-{dropped})")
            del probe
            gc.collect()
        X_final = X[selected_features]
    else:
        # Отбор отключён — берём все признаки
        selected_features = all_features
        X_final = X

    # ── Проход 2: CV (опционально) + финальный фит на отобранных признаках ──
    ensemble = _make_ensemble()

    if skip_cv:
        print("    CV оценка   [пропущена — skip_cv]", flush=True)
        cv_f1 = 0.0
    else:
        _print_step("CV оценка ансамбля...")
        cv_f1 = _evaluate_ensemble(ensemble, X_final, y, close_window, ticker)
        _print_ok(f"F1={cv_f1:.4f}")

    _print_step("Финальное обучение ансамбля...")
    ensemble.fit(X_final, y)
    _print_ok()

    if ml_settings.print_feature_importance:
        _print_feature_importance(ensemble, selected_features, ticker)

    # ── Оптимизация порога уверенности per-ticker ─────────────────────────────
    # Используем последние 20% данных как holdout для подбора threshold.
    # Небольшая утечка допустима: порог — скаляр, не веса модели.
    _print_step("Оптимизация порога уверенности (Optuna)...")
    val_size = max(1, int(len(X_final) * 0.2))
    X_val = X_final.iloc[-val_size:]
    y_val = y.values[-val_size:]
    cw_val = close_window[-val_size:]
    best_threshold = _optimize_threshold(ensemble, X_val, y_val, cw_val, ticker)
    _print_ok(f"threshold={best_threshold:.4f}")

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
    sortino_scores: dict[str, float] = {}

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
            path, cv_sortino = _train_single_ticker(ticker, group, force_tune, skip_cv=skip_cv)
            results[ticker] = path
            sortino_scores[ticker] = round(cv_sortino, 4)
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
    # При skip_cv Sortino не вычислялся — берём сохранённые значения из предыдущего обучения
    if skip_cv and prev_path.exists():
        try:
            with open(prev_path) as f_prev:
                prev_data = json.load(f_prev)
            prev_sortino = prev_data.get("sortino_scores", {})
            for ticker in sortino_scores:
                if sortino_scores[ticker] == 0.0 and ticker in prev_sortino:
                    sortino_scores[ticker] = prev_sortino[ticker]
        except Exception:
            pass
    with open(results_path, "w") as f:
        json.dump(
            {
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "force_tune": force_tune,
                "skip_cv": skip_cv,
                "sortino_scores": sortino_scores,
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
