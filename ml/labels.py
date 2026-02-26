"""
Label generation for supervised ML training.

Converts OHLCV price series into BUY / HOLD / SELL classification targets
based on future price returns over a configurable look-ahead window.
"""
import numpy as np
import pandas as pd

# Numeric label encoding used by LightGBM (must be 0-based integers)
LABEL_MAP: dict[str, int] = {
    "SELL": 0,
    "HOLD": 1,
    "BUY":  2,
}
LABEL_NAMES: list[str] = ["SELL", "HOLD", "BUY"]  # index → name


def create_labels(
    df: pd.DataFrame,
    lookahead: int = 4,
    threshold: float = 0.01,
) -> pd.Series:
    """
    Generate BUY/HOLD/SELL integer labels based on future price return.

    Label logic:
        future_return = (close[t+lookahead] - close[t]) / close[t]
        > +threshold  →  BUY  (2)
        < -threshold  →  SELL (0)
        otherwise     →  HOLD (1)

    The last `lookahead` rows are dropped because future price is unknown.

    Args:
        df: DataFrame with a "close" column, sorted by time ASC.
        lookahead: number of candles ahead to measure return (default 4 = 4h).
        threshold: minimum absolute return to trigger BUY/SELL (default 1%).

    Returns:
        pd.Series of int labels (SELL=0, HOLD=1, BUY=2), same index as df
        but with last `lookahead` rows removed.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    labels = np.ones(n, dtype=np.int8)  # default: HOLD

    # future_return is only defined for indices 0 .. n-lookahead-1
    valid = n - lookahead
    future_close = close[lookahead:lookahead + valid]
    current_close = close[:valid]

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        future_return = np.where(
            current_close != 0,
            (future_close - current_close) / current_close,
            0.0,
        )

    labels[:valid][future_return > threshold] = LABEL_MAP["BUY"]
    labels[:valid][future_return < -threshold] = LABEL_MAP["SELL"]

    # Drop last `lookahead` rows — no future data available there
    result = pd.Series(labels[:valid], index=df.index[:valid], name="label")
    return result
