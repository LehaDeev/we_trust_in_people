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
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config.settings import data_settings, ml_settings
from ml.dataset import load_all_tickers_dataset
from ml.features import FEATURE_COLUMNS, compute_features
from ml.labels import LABEL_NAMES, create_labels
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
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Вычислить признаки и метки для одного тикера.

    Аргументы:
        group:  DataFrame свечей одного тикера [time, open, high, low, close, volume].
        ticker: тикер (только для логирования).

    Возвращает:
        (X, y): DataFrame признаков и Series меток с согласованными индексами.
    """
    feat_df = compute_features(group)
    labels = create_labels(
        feat_df,
        lookahead=ml_settings.lookahead,
        threshold=ml_settings.threshold,
    )
    feat_df = feat_df.loc[labels.index].copy()

    logger.info(
        "Датасет тикера собран",
        ticker=ticker,
        samples=len(labels),
        buy=int((labels == 2).sum()),
        hold=int((labels == 1).sum()),
        sell=int((labels == 0).sum()),
    )

    return feat_df[FEATURE_COLUMNS], labels


# ── Оценка ансамбля ──────────────────────────────────────────────────────────

def _evaluate_ensemble(
    ensemble: VotingClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    ticker: str,
) -> float:
    """
    Оценить ансамбль через кросс-валидацию TimeSeriesSplit.

    Возвращает:
        Среднее f1_macro по всем фолдам.
    """
    cv = TimeSeriesSplit(n_splits=ml_settings.n_splits)
    scores = cross_val_score(ensemble, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    mean_f1 = float(np.mean(scores))
    logger.info(
        "Оценка ансамбля (CV)",
        ticker=ticker,
        f1_per_fold=[round(s, 4) for s in scores],
        mean_f1=round(mean_f1, 4),
    )
    return mean_f1


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


# ── Обучение одного тикера ───────────────────────────────────────────────────

def _train_single_ticker(
    ticker: str,
    group: pd.DataFrame,
    force_tune: bool,
) -> Path:
    """
    Полный pipeline обучения для одного тикера.

    Аргументы:
        ticker:     тикер инструмента.
        group:      DataFrame свечей этого тикера.
        force_tune: принудительно запустить Optuna.

    Возвращает:
        Path к сохранённому pkl ансамбля.
    """
    version = ml_settings.model_version
    ticker_version = f"{ticker}_{version}"

    X, y = _build_ticker_dataset(group, ticker)

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
        return _get_params(model_name, ticker_version, tune_fn, X, y, force_tune)

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
        )

    all_features = X.columns.tolist()
    threshold = ml_settings.feature_importance_threshold

    # ── Проход 1: быстрый фит на всех признаках → отбор по порогу per-ticker ──
    # Один фит (без CV) чтобы получить feature importance и отбросить слабые.
    # Для каждого тикера свой порог отбора — разные признаки могут быть важны
    # для банков (SBER), нефтяников (LKOH), IT-компаний (YDEX) и т.д.
    # CV-оценка и финальный фит выполняются уже на отобранных признаках.
    if threshold > 0.0:
        _print_step(f"Отбор признаков (проход 1 из 2, порог >={threshold})...")
        probe = _make_ensemble()
        probe.fit(X, y)
        selected_features = _select_by_threshold(probe, all_features, threshold)
        dropped = len(all_features) - len(selected_features)
        _print_ok(f"{len(selected_features)} из {len(all_features)} признаков (-{dropped})")
        X_final = X[selected_features]
    else:
        # Отбор отключён — берём все признаки
        selected_features = all_features
        X_final = X

    # ── Проход 2: CV + финальный фит на отобранных признаках ─────────────────
    ensemble = _make_ensemble()

    _print_step("CV оценка ансамбля...")
    cv_f1 = _evaluate_ensemble(ensemble, X_final, y, ticker)
    _print_ok(f"F1={cv_f1:.4f}")

    _print_step("Финальное обучение ансамбля...")
    ensemble.fit(X_final, y)
    _print_ok()

    if ml_settings.print_feature_importance:
        _print_feature_importance(ensemble, selected_features, ticker)

    ensemble_path = WEIGHTS_DIR / f"ensemble_{ticker_version}.pkl"
    features_path = WEIGHTS_DIR / f"features_{ticker_version}.json"

    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    # Сохраняем per-ticker список признаков (не глобальный FEATURE_COLUMNS).
    # predict.py загружает именно этот файл → инференс автоматически использует
    # тикерный набор без каких-либо изменений в коде инференса.
    with open(features_path, "w") as f:
        json.dump(selected_features, f, indent=2)

    print(f"  Сохранено: {ensemble_path.name}", flush=True)

    return ensemble_path


# ── Основная функция обучения ────────────────────────────────────────────────

async def train_model(force_tune: bool = False) -> dict[str, Path]:
    """
    Обучить отдельный ансамбль для каждого тикера из DATA_TICKERS.

    Для каждого тикера свой Optuna HPO (с кешем), свои веса.
    Файлы: ensemble_{ticker}_{version}.pkl, features_{ticker}_{version}.json.

    Аргументы:
        force_tune: если True — принудительно запустить Optuna для всех моделей.

    Возвращает:
        Словарь {ticker: Path} для успешно обученных тикеров.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if force_tune:
        logger.info("Режим force_tune: Optuna будет запущена для всех моделей")

    logger.info("Загрузка данных свечей...", tickers=data_settings.tickers)
    raw = await load_all_tickers_dataset(
        data_settings.tickers,
        interval=data_settings.candle_interval,
    )

    if raw.empty:
        raise RuntimeError("Нет данных. Запустите scripts/collect_candles.py.")

    results: dict[str, Path] = {}
    tickers_list = list(raw["ticker"].unique())
    total = len(tickers_list)

    for i, (ticker, group) in enumerate(raw.groupby("ticker", sort=False), 1):
        ticker = str(ticker)
        group = group.reset_index(drop=True)
        _print_ticker_header(ticker, i, total, len(group))
        logger.info("Обучение модели", ticker=ticker)
        try:
            path = _train_single_ticker(ticker, group, force_tune)
            results[ticker] = path
        except Exception as e:
            logger.error("Ошибка обучения тикера", ticker=ticker, error=str(e))
            print(f"  x Error: {e}", flush=True)

    failed = [t for t in data_settings.tickers if t not in results]
    _print_summary(results, failed)

    logger.info(
        "Обучение завершено",
        trained=list(results.keys()),
        failed=failed,
    )

    return results
