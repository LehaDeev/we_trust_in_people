"""
Вычисление технических индикаторов для ML-моделей через TA-Lib.

Принимает DataFrame с колонками OHLCV и возвращает обогащённый DataFrame
с 20 признаками технического анализа, готовыми для обучения LightGBM.

Все ценовые признаки нормализованы (отношения) — не зависят от абсолютного
уровня цены тикера. Это обеспечивает стационарность признаков и корректное
дообучение при росте или падении цены со временем.
"""
import numpy as np
import pandas as pd
import talib

# Имена колонок признаков (используются для согласованности между обучением и инференсом)
FEATURE_COLUMNS: list[str] = [
    # Тренд (нормализованные)
    "SMA_20", "SMA_50", "ema12_ratio", "ema26_ratio", "ADX_14",
    # Импульс
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "ROC_10",
    # Волатильность (нормализованные)
    "bb_pct_b", "atr_ratio", "BB_width",
    # Объём
    "OBV", "VOLUME_SMA_20", "volume_ratio",
    # Ценовые отношения (производные)
    "close_sma20_ratio", "close_sma50_ratio", "high_low_ratio", "sma20_sma50_ratio",
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
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ── Индикаторы тренда ────────────────────────────────────────────────────
    sma20 = talib.SMA(close, timeperiod=20)
    sma50 = talib.SMA(close, timeperiod=50)
    ema12 = talib.EMA(close, timeperiod=12)
    ema26 = talib.EMA(close, timeperiod=26)

    df["SMA_20"] = sma20
    df["SMA_50"] = sma50
    # Нормализованные EMA: отношение цены закрытия к скользящей средней
    # close/EMA > 1 → цена выше EMA (бычий сигнал), < 1 → ниже (медвежий)
    df["ema12_ratio"] = np.where(ema12 != 0, close / ema12, np.nan)
    df["ema26_ratio"] = np.where(ema26 != 0, close / ema26, np.nan)
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
    atr = talib.ATR(high, low, close, timeperiod=14)

    # Bollinger %B: позиция цены внутри полос [0 = нижняя полоса, 1 = верхняя]
    # < 0 → цена ниже нижней полосы (перепродан), > 1 → выше верхней (перекуплен)
    bb_band_width = bb_upper - bb_lower
    df["bb_pct_b"] = np.where(bb_band_width != 0, (close - bb_lower) / bb_band_width, np.nan)

    # ATR как доля от цены закрытия: мера волатильности независимо от уровня цены
    df["atr_ratio"] = np.where(close != 0, atr / close, np.nan)

    # Ширина полос Боллинджера: нормализованная ширина (защита от деления на ноль)
    df["BB_width"] = np.where(bb_mid != 0, (bb_upper - bb_lower) / bb_mid, np.nan)

    # ── Индикаторы объёма ────────────────────────────────────────────────────
    volume_sma20 = talib.SMA(volume, timeperiod=20)
    df["OBV"] = talib.OBV(close, volume)
    df["VOLUME_SMA_20"] = volume_sma20
    # Относительный объём: текущий бар vs средний (>1 = повышенный объём)
    df["volume_ratio"] = np.where(volume_sma20 != 0, volume / volume_sma20, np.nan)

    # ── Производные ценовые отношения ────────────────────────────────────────
    df["close_sma20_ratio"] = np.where(sma20 != 0, close / sma20, np.nan)
    df["close_sma50_ratio"] = np.where(sma50 != 0, close / sma50, np.nan)
    df["high_low_ratio"] = np.where(close != 0, (high - low) / close, np.nan)
    # Отношение SMA_20 к SMA_50: > 1 = золотой крест (бычий тренд), < 1 = мёртвый крест
    df["sma20_sma50_ratio"] = np.where(sma50 != 0, sma20 / sma50, np.nan)

    # Удаляем строки где хотя бы один признак NaN (период прогрева, ~50 строк на тикер)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df
