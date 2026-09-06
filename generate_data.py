"""Generate a synthetic daily stock price series for a fictional ticker.

This project ships with pre-generated data already in place
(data/stock_prices.csv), so you don't need to run this script to try the
project out. Run it again if you want a fresh random series, a different
date range, or different volatility.

Usage:
    python data/generate_data.py [--days 1000] [--seed 42] [--out data/stock_prices.csv]
"""

import argparse
import numpy as np
import pandas as pd


def generate_prices(n_days=1000, start_price=100.0, seed=42):
    """Simulate a daily close price series with:
    - a mild upward drift (long-term trend)
    - a slow sinusoidal cycle (simulating multi-month market cycles)
    - day-to-day random walk noise (geometric Brownian motion style)
    """
    rng = np.random.default_rng(seed)

    days = np.arange(n_days)
    drift = 0.0004                       # small daily upward drift
    cycle = 0.08 * np.sin(2 * np.pi * days / 250)   # slow yearly-ish cycle
    daily_vol = 0.014

    noise = rng.normal(loc=0.0, scale=daily_vol, size=n_days)
    log_returns = drift + np.diff(cycle, prepend=cycle[0]) + noise

    log_prices = np.log(start_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    return prices


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic stock price data.")
    parser.add_argument("--days", type=int, default=1000, help="Number of trading days to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-price", type=float, default=100.0, help="Starting price")
    parser.add_argument("--out", default="data/stock_prices.csv", help="Output CSV path")
    args = parser.parse_args()

    prices = generate_prices(n_days=args.days, start_price=args.start_price, seed=args.seed)

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=args.days)
    # Defensive check: bdate_range should return exactly `periods` rows, but
    # guard against any off-by-one edge case rather than letting the
    # DataFrame constructor fail with a confusing length-mismatch error.
    n = min(len(dates), len(prices))
    dates, prices = dates[-n:], prices[-n:]
    df = pd.DataFrame({"date": dates, "close": np.round(prices, 2)})
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows of synthetic daily prices to {args.out}")


if __name__ == "__main__":
    main()
