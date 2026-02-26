"""
Вычисление технических индикаторов для ML-моделей через TA-Lib.

Принимает DataFrame с колонками OHLCV и возвращает обогащённый DataFrame
с 20+ признаками технического анализа, готовыми для обучения LightGBM.
"""
import numpy as np
import pandas as pd
import talib

# Имена колонок признаков (используются для согласованности между обучением и инференсом)
FEATURE_COLUMNS: list[str] = [
    # Тренд
    "SMA_20", "SMA_50", "EMA_12", "EMA_26", "ADX_14",
    # Импульс
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "ROC_10",
    # Волатильность
    "BB_upper", "BB_mid", "BB_lower", "ATR_14", "BB_width",
    # Объём
    "OBV", "VOLUME_SMA_20",
    # Ценовые отношения (производные)
    "close_sma20_ratio", "close_sma50_ratio", "high_low_ratio",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычислить признаки технических индикаторов для одного тикера.

    Аргументы:
        df: DataFrame с колонками [time, open, high, low, close, volume].
            Должен быть отсортирован по времени ASC. Значения — float64.

    Возвращает:
        DataFrame с исходными колонками + FEATURE_COLUMNS.
        Первые ~50 строк удаляются (период прогрева индикаторов).
        Индекс сбрасывается к целым числам начиная с 0.
    """
    df = df.copy()

    # TA-Lib требует numpy-массивы типа float64
    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ── Индикаторы тренда ────────────────────────────────────────────────────
    df["SMA_20"] = talib.SMA(close, timeperiod=20)
    df["SMA_50"] = talib.SMA(close, timeperiod=50)
    df["EMA_12"] = talib.EMA(close, timeperiod=12)
    df["EMA_26"] = talib.EMA(close, timeperiod=26)
    df["ADX_14"] = talib.ADX(high, low, close, timeperiod=14)

    # ── Индикаторы импульса ──────────────────────────────────────────────────
    df["RSI_14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    df["ROC_10"] = talib.ROC(close, timeperiod=10)

    # ── Индикаторы волатильности ─────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df["BB_upper"] = bb_upper
    df["BB_mid"] = bb_mid
    df["BB_lower"] = bb_lower
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    # Ширина полос Боллинджера: нормализованная ширина (защита от деления на ноль)
    df["BB_width"] = np.where(
        bb_mid != 0, (bb_upper - bb_lower) / bb_mid, np.nan
    )

    # ── Индикаторы объёма ────────────────────────────────────────────────────
    df["OBV"] = talib.OBV(close, volume)
    df["VOLUME_SMA_20"] = talib.SMA(volume, timeperiod=20)

    # ── Производные ценовые отношения ────────────────────────────────────────
    df["close_sma20_ratio"] = np.where(
        df["SMA_20"] != 0, close / df["SMA_20"], np.nan
    )
    df["close_sma50_ratio"] = np.where(
        df["SMA_50"] != 0, close / df["SMA_50"], np.nan
    )
    df["high_low_ratio"] = np.where(
        close != 0, (high - low) / close, np.nan
    )

    # Удаляем строки где хотя бы один признак NaN (период прогрева, ~50 строк на тикер)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df
