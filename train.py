"""Train a next-day closing-price forecaster.

Usage:
    python src/train.py [--config config.yaml]

Uses a chronological (not random) train/test split, since shuffling time
series data would leak future information into training. Trains a Gradient
Boosting regressor on lag + rolling-statistic features and reports error
metrics on the held-out final `test_days` days.
"""


import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_price_series
from features import build_feature_frame, feature_columns


def main():
    parser = argparse.ArgumentParser(description="Train a stock price forecaster.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    n_lags = config.get("n_lags", 10)
    rolling_windows = config.get("rolling_windows", [5, 10, 20])
    test_days = config.get("test_days", 60)
    random_state = config.get("random_state", 42)

    print(f"Loading price data from {config['data_path']} ...")
    df = load_price_series(config["data_path"])
    print(f"Loaded {len(df)} rows spanning {df['date'].min().date()} to {df['date'].max().date()}")

    X, y, dates = build_feature_frame(df, n_lags, rolling_windows)
    cols = feature_columns(n_lags, rolling_windows)
    X = X[cols]

    if len(X) <= test_days:
        raise ValueError(
            f"Not enough data ({len(X)} usable rows) for a test_days={test_days} split. "
            "Generate more data or lower test_days in config.yaml."
        )

    X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
    y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]
    dates_test = dates.iloc[-test_days:]

    model = GradientBoostingRegressor(random_state=random_state)
    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds) * 100

    print(f"\nEvaluation on last {test_days} days ({dates_test.min().date()} to {dates_test.max().date()}):")
    print(f"  MAE:  {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAPE: {mape:.2f}%")

    # Naive baseline for comparison: "tomorrow = today's price"
    naive_preds = y_test.shift(1).bfill()
    naive_mae = mean_absolute_error(y_test, naive_preds)
    print(f"  (naive baseline MAE for comparison: {naive_mae:.3f})")

    model_path = config["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "n_lags": n_lags,
        "rolling_windows": rolling_windows,
        "feature_cols": cols,
    }, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
