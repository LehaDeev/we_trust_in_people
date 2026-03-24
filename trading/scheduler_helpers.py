"""
Вспомогательные функции торгового планировщика.

Расчёт динамических SL/TP, фильтр режима рынка, обновление свечей,
получение цен и размеров лотов.
"""
import json
import math
from decimal import Decimal
from pathlib import Path

from utils.logger import logger

_WEIGHTS_DIR = Path(__file__).parent.parent / "ml" / "weights"


def _ticker_threshold(ticker: str) -> float:
    """Загрузить per-ticker порог уверенности из best_threshold_{ticker}_{version}.json.

    При отсутствии файла возвращает глобальный TRADING_CONFIDENCE_THRESHOLD.
    """
    from config.settings import ml_settings, trading_settings
    path = _WEIGHTS_DIR / f"best_threshold_{ticker}_{ml_settings.model_version}.json"
    try:
        with open(path) as f:
            return json.load(f)["threshold"]
    except Exception:
        return trading_settings.confidence_threshold


def _apply_regime_filter(lots: int, regime: int) -> int:
    """
    Скорректировать количество лотов с учётом рыночного режима.

    Режимы рынка (из predict_signal / compute_features):
        +1 = аптренд  → лоты не изменяются.
        -1 = даунтренд → BUY всегда блокируется (lots = 0).
         0 = флет     → поведение зависит от TRADING_REGIME_FILTER_MODE:
             "soft": лоты × TRADING_REGIME_FLAT_MULTIPLIER (по умолчанию 0.5).
             "hard": BUY блокируется полностью (lots = 0).

    При TRADING_REGIME_FILTER_ENABLED=false возвращает lots без изменений
    (backward compatibility — фильтр отключён).

    Аргументы:
        lots:   расчётное количество лотов (>= 0).
        regime: режим рынка (-1, 0 или +1).

    Возвращает:
        Скорректированное количество лотов (0 = BUY пропускается).
    """
    from config.settings import trading_settings as ts
    if not ts.regime_filter_enabled:
        return lots
    if regime == -1:
        return 0
    if regime == 0:
        if ts.regime_filter_mode == "hard":
            return 0
        if ts.regime_flat_lots_multiplier <= 0:
            return 0
        return max(1, int(lots * ts.regime_flat_lots_multiplier))
    return lots


def _compute_dynamic_sltp(atr_ratio: float) -> tuple[float, float]:
    """
    Вычислить динамические SL/TP на основе ATR-волатильности последнего бара.

    Алгоритм:
        sl_pct = clamp(atr_ratio × ATR_SL_MULTIPLIER, min_sl, max_sl)
        tp_pct = sl_pct × ATR_RISK_REWARD_RATIO

    При TRADING_DYNAMIC_SLTP_ENABLED=false или некорректном atr_ratio (0, NaN, inf)
    возвращает фиксированные значения TRADING_STOP_LOSS_PCT / TRADING_TAKE_PROFIT_PCT
    из настроек — полная обратная совместимость.

    Аргументы:
        atr_ratio: ATR(14) / close последнего бара из predict_signal().
                   Типичный диапазон MOEX 1h: 0.005–0.015.
                   0.0 означает «не вычислено» → возврат фиксированных значений.

    Возвращает:
        (sl_pct, tp_pct): доли от цены входа (например 0.025, 0.042).
    """
    from config.settings import trading_settings as ts
    if not ts.dynamic_sltp_enabled or not math.isfinite(atr_ratio) or atr_ratio <= 0.0:
        return ts.stop_loss_pct, ts.take_profit_pct
    sl = float(max(ts.atr_min_sl_pct, min(atr_ratio * ts.atr_sl_multiplier, ts.atr_max_sl_pct)))
    tp = sl * ts.atr_risk_reward_ratio
    tp = max(tp, sl)
    return sl, tp


async def _fetch_price(figi: str) -> Decimal:
    """
    Получить текущую цену одного инструмента.

    Аргументы:
        figi: FIGI инструмента

    Возвращает:
        Цена или Decimal("0") при ошибке.
    """
    from tinkoff.market_data import get_last_prices
    try:
        prices = await get_last_prices([figi])
        return prices.get(figi, Decimal("0"))
    except Exception as e:
        logger.error("Ошибка получения цены", figi=figi, error=str(e))
        return Decimal("0")


async def _update_candles() -> None:
    """
    Инкрементально обновить свечи из Tinkoff API и сбросить Redis-кеш.

    Вызывается перед каждым ML-инференсом, чтобы модель работала
    на актуальных данных, а не только на ночном снимке.
    При ошибке — логирует и продолжает тик без обновления.
    """
    from config.settings import data_settings, ml_settings, trading_settings
    from scripts.collect_candles import run_collection
    from utils.redis_cache import get_redis
    try:
        await run_collection(pause_seconds=trading_settings.candle_update_pause_seconds)
    except Exception as e:
        logger.warning("Не удалось обновить свечи перед инференсом", error=str(e))
        return

    try:
        redis = await get_redis()
        if redis is not None:
            interval = data_settings.candle_interval
            version = ml_settings.model_version
            for ticker in data_settings.tickers + ["USDRUB"]:
                await redis.delete(f"candles:{ticker}:{interval}")
            for ticker in data_settings.tickers:
                await redis.delete(f"signal:{ticker}:{version}")
            logger.debug("Redis-кеш свечей и сигналов инвалидирован")
    except Exception as e:
        logger.warning("Ошибка инвалидации Redis-кеша после обновления свечей", error=str(e))


async def _get_lot_size(ticker: str) -> int:
    """
    Получить количество бумаг в 1 лоте для тикера.

    Использует API Tinkoff; при ошибке возвращает 1 (безопасный дефолт).

    Аргументы:
        ticker: тикер инструмента

    Возвращает:
        Размер лота (>= 1).
    """
    from tinkoff.instruments import get_instrument_by_ticker
    try:
        info = await get_instrument_by_ticker(ticker)
        return info.lot if info and info.lot > 0 else 1
    except Exception as e:
        logger.warning("Не удалось получить lot_size", ticker=ticker, error=str(e))
        return 1
