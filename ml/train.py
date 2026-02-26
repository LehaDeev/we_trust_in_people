"""
LightGBM model training for BUY/SELL/HOLD signal prediction.

Loads candle data from PostgreSQL, computes technical indicator features,
generates labels, trains a multi-class LightGBM classifier, and saves
model weights to ml/weights/.

Usage (via scripts/train_model.py):
    python -m scripts.train_model
"""
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml.dataset import load_all_tickers_dataset
from ml.features import FEATURE_COLUMNS, compute_features
from ml.labels import LABEL_NAMES, create_labels
from utils.logger import logger

# ── Config ──────────────────────────────────────────────────────────────────

MODEL_VERSION = "v1"
WEIGHTS_DIR = Path(__file__).parent / "weights"

TICKERS = [
    "SBER", "GAZP", "LKOH", "YDEX", "NVTK",
    "GMKN", "MGNT", "TATN", "ROSN", "MTSS",
]

# LightGBM hyperparameters
LGB_PARAMS: dict = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbose": -1,
}

LOOKAHEAD = 4       # hours ahead to compute return
THRESHOLD = 0.01    # ±1% threshold for BUY/SELL


# ── Training logic ───────────────────────────────────────────────────────────

def _build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply feature engineering and label generation per ticker.

    Args:
        raw: combined DataFrame with [ticker, time, open, high, low, close, volume].

    Returns:
        (X, y): feature DataFrame and label Series, both with matching indices.
    """
    feature_frames: list[pd.DataFrame] = []
    label_series: list[pd.Series] = []

    for ticker, group in raw.groupby("ticker", sort=False):
        group = group.reset_index(drop=True)

        # Features (also drops warmup NaN rows)
        feat_df = compute_features(group)

        # Labels (drops last LOOKAHEAD rows)
        labels = create_labels(feat_df, lookahead=LOOKAHEAD, threshold=THRESHOLD)

        # Align: features and labels must cover the same rows
        feat_df = feat_df.loc[labels.index]

        feat_df = feat_df.copy()
        feat_df["_ticker"] = ticker  # keep for debugging, not used as feature

        feature_frames.append(feat_df)
        label_series.append(labels)

        logger.info(
            "Ticker dataset built",
            ticker=ticker,
            samples=len(labels),
            buy=int((labels == 2).sum()),
            hold=int((labels == 1).sum()),
            sell=int((labels == 0).sum()),
        )

    if not feature_frames:
        raise RuntimeError("No training data could be built from the database.")

    combined_features = pd.concat(feature_frames, ignore_index=True)
    combined_labels = pd.concat(label_series, ignore_index=True)

    X = combined_features[FEATURE_COLUMNS]
    y = combined_labels

    return X, y


async def train_model() -> Path:
    """
    Full training pipeline: load data → features → labels → train → save.

    Returns:
        Path to saved model pkl file.
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading candle data from database...", tickers=TICKERS)
    raw = await load_all_tickers_dataset(TICKERS, interval="1h")

    if raw.empty:
        raise RuntimeError("No candle data found. Run scripts/collect_candles.py first.")

    logger.info("Building feature/label dataset...")
    X, y = _build_dataset(raw)

    logger.info(
        "Dataset ready",
        total_samples=len(X),
        features=len(FEATURE_COLUMNS),
        class_distribution=y.value_counts().to_dict(),
    )

    # Train/validation split (chronological — do NOT shuffle for time series)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    logger.info(
        "Training LightGBM...",
        train_size=len(X_train),
        val_size=len(X_val),
    )

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=LABEL_NAMES)
    logger.info("Validation classification report:\n" + report)

    val_accuracy = float(np.mean(y_pred == y_val.values))
    logger.info("Validation accuracy", accuracy=round(val_accuracy, 4))

    # Save model
    model_path = WEIGHTS_DIR / f"lgbm_{MODEL_VERSION}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save feature list (required for consistent inference)
    features_path = WEIGHTS_DIR / f"features_{MODEL_VERSION}.json"
    with open(features_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    logger.info(
        "Model saved",
        model_path=str(model_path),
        features_path=str(features_path),
        val_accuracy=round(val_accuracy, 4),
    )

    return model_path
