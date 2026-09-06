"""Forecast future closing prices with a trained model.

Usage:
    python src/predict.py [--horizon 14] [--plot]

Performs an iterative (walk-forward) multi-step forecast: predicts the next
day, appends that prediction to the price history, then predicts the day
after that, and so on, for `horizon` business days beyond the last known
date in the dataset.
"""

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_price_series
from features import make_features_from_history


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at '{model_path}'. Run `python src/train.py` first."
        )
    return joblib.load(model_path)


def forecast(df, bundle, horizon):
    model = bundle["model"]
    n_lags = bundle["n_lags"]
    rolling_windows = bundle["rolling_windows"]
    cols = bundle["feature_cols"]

    history = df["close"].tolist()
    last_date = df["date"].max()
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    forecasts = []
    for next_date in future_dates:
        X_next = make_features_from_history(history, next_date, n_lags, rolling_windows, cols)
        pred = model.predict(X_next)[0]
        forecasts.append((next_date, pred))
        history.append(pred)  # feed the prediction back in for the next step

    return pd.DataFrame(forecasts, columns=["date", "predicted_close"])


def main():
    parser = argparse.ArgumentParser(description="Forecast future closing prices.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--horizon", type=int, help="Number of business days to forecast (overrides config)")
    parser.add_argument("--plot", action="store_true", help="Save a PNG chart of history + forecast")
    args = parser.parse_args()

    config = load_config(args.config)
    horizon = args.horizon or config.get("forecast_horizon", 14)

    df = load_price_series(config["data_path"])
    bundle = load_model(config["model_path"])

    forecast_df = forecast(df, bundle, horizon)

    print(f"Last known close: {df['close'].iloc[-1]:.2f} on {df['date'].iloc[-1].date()}\n")
    print(f"Forecast for the next {horizon} business days:")
    for _, row in forecast_df.iterrows():
        print(f"  {row['date'].date()}  ->  {row['predicted_close']:.2f}")

    print(
        "\nNote: this is a walk-forward forecast built on synthetic/sample data for "
        "demonstration purposes. It is not financial advice and should not be used "
        "to make real trading or investment decisions."
    )

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        history_tail = df.tail(120)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history_tail["date"], history_tail["close"], label="History", color="#2A4D69")
        ax.plot(forecast_df["date"], forecast_df["predicted_close"], label="Forecast",
                color="#C1440E", linestyle="--", marker="o", markersize=3)
        ax.axvline(df["date"].iloc[-1], color="gray", linestyle=":", linewidth=1)
        ax.set_title("Closing price: history and forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.legend()
        fig.autofmt_xdate()

        out_path = "forecast_plot.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved chart to {out_path}")


if __name__ == "__main__":
    main()
