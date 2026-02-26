"""
Feature engineering for ML models using TA-Lib technical indicators.

Takes a DataFrame with OHLCV columns and returns enriched DataFrame
with 20+ technical indicator features ready for LightGBM training.
"""
import numpy as np
import pandas as pd
import talib

# Feature column names (used for consistent train/predict alignment)
FEATURE_COLUMNS: list[str] = [
    # Trend
    "SMA_20", "SMA_50", "EMA_12", "EMA_26", "ADX_14",
    # Momentum
    "RSI_14", "MACD", "MACD_signal", "MACD_hist", "ROC_10",
    # Volatility
    "BB_upper", "BB_mid", "BB_lower", "ATR_14", "BB_width",
    # Volume
    "OBV", "VOLUME_SMA_20",
    # Price ratios (derived)
    "close_sma20_ratio", "close_sma50_ratio", "high_low_ratio",
]


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicator features for a single ticker DataFrame.

    Args:
        df: DataFrame with columns [time, open, high, low, close, volume].
            Must be sorted by time ASC. Values must be float64.

    Returns:
        DataFrame with original columns + FEATURE_COLUMNS.
        First ~50 rows are dropped (indicator warmup period).
        Index is reset to 0-based integers.
    """
    df = df.copy()

    # TA-Lib requires numpy float64 arrays
    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)

    # ── Trend indicators ────────────────────────────────────────────────────
    df["SMA_20"] = talib.SMA(close, timeperiod=20)
    df["SMA_50"] = talib.SMA(close, timeperiod=50)
    df["EMA_12"] = talib.EMA(close, timeperiod=12)
    df["EMA_26"] = talib.EMA(close, timeperiod=26)
    df["ADX_14"] = talib.ADX(high, low, close, timeperiod=14)

    # ── Momentum indicators ─────────────────────────────────────────────────
    df["RSI_14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    df["ROC_10"] = talib.ROC(close, timeperiod=10)

    # ── Volatility indicators ───────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df["BB_upper"] = bb_upper
    df["BB_mid"] = bb_mid
    df["BB_lower"] = bb_lower
    df["ATR_14"] = talib.ATR(high, low, close, timeperiod=14)
    # BB width: normalised band width (avoids division by zero)
    df["BB_width"] = np.where(
        bb_mid != 0, (bb_upper - bb_lower) / bb_mid, np.nan
    )

    # ── Volume indicators ───────────────────────────────────────────────────
    df["OBV"] = talib.OBV(close, volume)
    df["VOLUME_SMA_20"] = talib.SMA(volume, timeperiod=20)

    # ── Derived price ratios ────────────────────────────────────────────────
    df["close_sma20_ratio"] = np.where(
        df["SMA_20"] != 0, close / df["SMA_20"], np.nan
    )
    df["close_sma50_ratio"] = np.where(
        df["SMA_50"] != 0, close / df["SMA_50"], np.nan
    )
    df["high_low_ratio"] = np.where(
        close != 0, (high - low) / close, np.nan
    )

    # Drop rows where any feature is NaN (warmup period, ~50 rows per ticker)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df
