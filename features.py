"""Feature engineering for the price forecaster.

All features are built from *past* prices only (never the current day's
close), so the model can't cheat by peeking at the value it's predicting.
The same feature logic is reused for training (build_feature_frame) and for
iterative multi-step forecasting (make_features_from_history).
"""

import numpy as np
import pandas as pd


def feature_columns(n_lags, rolling_windows):
    cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    for w in rolling_windows:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    cols += ["day_of_week"]
    return cols


def build_feature_frame(df, n_lags, rolling_windows):
    """Build a supervised-learning table from a `date,close` price DataFrame.

    Returns (X, y, dates) where X are the features, y is the next close
    price (the target), and dates align each row to its target's date.
    """
    close = df["close"]
    # shifted = the most recent *known* price before the day we're predicting
    shifted = close.shift(1)

    features = {}
    for i in range(1, n_lags + 1):
        features[f"lag_{i}"] = close.shift(i)

    for w in rolling_windows:
        features[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        features[f"roll_std_{w}"] = shifted.rolling(w).std()

    features["day_of_week"] = df["date"].dt.dayofweek

    X = pd.DataFrame(features)
    y = close

    valid = X.notna().all(axis=1)
    X, y, dates = X[valid], y[valid], df["date"][valid]
    return X.reset_index(drop=True), y.reset_index(drop=True), dates.reset_index(drop=True)


def make_features_from_history(history_closes, next_date, n_lags, rolling_windows, cols):
    """Build a single feature row to predict the close on `next_date`,
    given a list/array of the most recent known close prices (most recent last).
    """
    history = np.asarray(history_closes, dtype=float)
    row = {}
    for i in range(1, n_lags + 1):
        row[f"lag_{i}"] = history[-i]

    for w in rolling_windows:
        window = history[-w:]
        row[f"roll_mean_{w}"] = window.mean()
        row[f"roll_std_{w}"] = window.std()

    row["day_of_week"] = pd.Timestamp(next_date).dayofweek

    return pd.DataFrame([row], columns=cols)
