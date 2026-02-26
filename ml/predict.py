"""
Inference: generate BUY/SELL/HOLD signal for a single ticker.

Loads the trained LightGBM model from disk, fetches the latest candles
from the database, computes features, and returns a signal dict.

Usage:
    import asyncio
    from ml.predict import predict_signal

    result = asyncio.run(predict_signal("SBER"))
    # {"ticker": "SBER", "signal": "BUY", "confidence": 0.72}
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ml.dataset import load_ticker_data
from ml.features import compute_features
from ml.labels import LABEL_NAMES
from utils.logger import logger

# Default paths (can be overridden via arguments)
DEFAULT_WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_MODEL_VERSION = "v1"

# Minimum candles needed: 50 warmup + 200 buffer for robust indicator values
MIN_CANDLES = 250


async def predict_signal(
    ticker: str,
    model_version: str = DEFAULT_MODEL_VERSION,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    interval: str = "1h",
) -> dict:
    """
    Generate a BUY/SELL/HOLD signal for the given ticker.

    Args:
        ticker: instrument ticker (e.g. "SBER").
        model_version: model version suffix (default "v1").
        weights_dir: directory containing pkl and json weight files.
        interval: candle interval to load from DB (default "1h").

    Returns:
        dict with keys:
            ticker (str): the ticker
            signal (str): "BUY", "HOLD", or "SELL"
            confidence (float): model probability for the predicted class
            probabilities (dict): {"SELL": p, "HOLD": p, "BUY": p}

    Raises:
        FileNotFoundError: if model weights are not found.
        ValueError: if not enough candle data is available.
    """
    model_path = weights_dir / f"lgbm_{model_version}.pkl"
    features_path = weights_dir / f"features_{model_version}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            "Run: python -m scripts.train_model"
        )

    # Load model and expected feature list
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path) as f:
        feature_columns: list[str] = json.load(f)

    # Load candle data
    df = await load_ticker_data(ticker, interval)

    if len(df) < MIN_CANDLES:
        raise ValueError(
            f"Not enough candles for {ticker}: {len(df)} < {MIN_CANDLES} required."
        )

    # Use only the most recent candles to keep inference fast
    df = df.tail(MIN_CANDLES).reset_index(drop=True)

    # Compute features (drops warmup rows internally)
    feat_df = compute_features(df)

    if feat_df.empty:
        raise ValueError(f"No valid feature rows for {ticker} after indicator warmup.")

    # Take the most recent row
    last_row = feat_df[feature_columns].iloc[[-1]]

    # Predict
    proba = model.predict_proba(last_row)[0]  # shape: (3,) — SELL, HOLD, BUY
    predicted_class = int(np.argmax(proba))
    confidence = float(proba[predicted_class])
    signal = LABEL_NAMES[predicted_class]

    result = {
        "ticker": ticker,
        "signal": signal,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(LABEL_NAMES, proba)
        },
    }

    logger.info(
        "Signal generated",
        ticker=ticker,
        signal=signal,
        confidence=round(confidence, 4),
    )

    return result


async def predict_all(
    tickers: list[str],
    model_version: str = DEFAULT_MODEL_VERSION,
    weights_dir: Path = DEFAULT_WEIGHTS_DIR,
    interval: str = "1h",
) -> list[dict]:
    """
    Generate signals for a list of tickers. Skips on error.

    Returns:
        List of signal dicts (only for successfully processed tickers).
    """
    results: list[dict] = []
    for ticker in tickers:
        try:
            signal = await predict_signal(ticker, model_version, weights_dir, interval)
            results.append(signal)
        except Exception as e:
            logger.error("Failed to predict signal", ticker=ticker, error=str(e))
    return results
