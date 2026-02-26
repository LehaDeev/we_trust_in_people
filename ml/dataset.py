"""
Data loading utilities for the ML pipeline.

Loads OHLCV candle data from PostgreSQL via async SQLAlchemy
and returns pandas DataFrames ready for feature engineering.
"""
import pandas as pd
from sqlalchemy import select

from db.database import get_session
from db.models import Asset, Candle
from utils.logger import logger


async def load_ticker_data(
    ticker: str,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Load all candles for a ticker from the database.

    Args:
        ticker: instrument ticker (e.g. "SBER", "GAZP").
        interval: candle interval string stored in DB (e.g. "1h", "1d").

    Returns:
        DataFrame with columns [time, open, high, low, close, volume],
        sorted by time ASC, with float64 price columns.
        Returns empty DataFrame if ticker not found.
    """
    async with get_session() as session:
        stmt = (
            select(
                Candle.time,
                Candle.open,
                Candle.high,
                Candle.low,
                Candle.close,
                Candle.volume,
            )
            .join(Asset, Candle.asset_id == Asset.id)
            .where(Asset.ticker == ticker)
            .where(Candle.interval == interval)
            .order_by(Candle.time.asc())
        )
        result = await session.execute(stmt)
        rows = result.fetchall()

    if not rows:
        logger.warning("No candles found", ticker=ticker, interval=interval)
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])

    # Convert Decimal to float64 for TA-Lib compatibility
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)

    logger.info(
        "Ticker data loaded",
        ticker=ticker,
        interval=interval,
        rows=len(df),
    )
    return df


async def load_all_tickers_dataset(
    tickers: list[str],
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Load candle data for multiple tickers and combine into one DataFrame.

    Each ticker's rows get a "ticker" column for grouping during feature
    engineering (indicators must be computed per-ticker, not across all data).

    Args:
        tickers: list of ticker strings.
        interval: candle interval string (e.g. "1h").

    Returns:
        Combined DataFrame with columns [ticker, time, open, high, low, close, volume].
        Tickers with no data are skipped.
    """
    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        df = await load_ticker_data(ticker, interval)
        if df.empty:
            logger.warning("Skipping ticker — no data", ticker=ticker)
            continue
        df.insert(0, "ticker", ticker)
        frames.append(df)

    if not frames:
        logger.error("No data loaded for any ticker")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "All tickers loaded",
        tickers=len(frames),
        total_rows=len(combined),
    )
    return combined
