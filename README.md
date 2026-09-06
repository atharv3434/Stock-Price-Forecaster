# Stock Price Forecaster (Time Series)

A self-contained project for forecasting daily closing prices using lag and
rolling-statistic features with a Gradient Boosting regressor. Ships with a
synthetic sample price series so you can train and forecast immediately —
swap in real historical data whenever you're ready.

> **Educational project, not financial advice.** The bundled data is
> synthetic (randomly generated), and even with real data, this is a simple
> baseline model. Don't use its output to make real trading or investment
> decisions.

## Project structure

```
stock-forecaster/
├── config.yaml               # paths, feature settings, split, horizon
├── requirements.txt
├── data/
│   ├── generate_data.py      # (re)generates the synthetic sample series
│   └── stock_prices.csv      # pre-generated sample data: date,close
├── models/                    # trained model gets saved here
├── src/
│   ├── utils.py               # config + data loading helpers
│   ├── features.py            # lag / rolling-window feature engineering
│   ├── train.py                # trains and evaluates the model
│   └── predict.py              # multi-step forecast + optional chart
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Train

```bash
python src/train.py
```

Loads `data/stock_prices.csv`, builds lag and rolling-mean/std features,
does a **chronological** train/test split (the last 60 days are held out —
never shuffled, since shuffling time series leaks the future into training),
trains a Gradient Boosting regressor, and prints MAE / RMSE / MAPE alongside
a naive "tomorrow = today" baseline for comparison. The model is saved to
`models/forecaster.joblib`.

## 2. Forecast

```bash
python src/predict.py --horizon 14 --plot
```

Runs a walk-forward forecast: predicts the next day, feeds that prediction
back in as if it were real, predicts the day after, and so on for `horizon`
business days. Prints a table of forecasted dates and prices. Add `--plot`
to save `forecast_plot.png` showing recent history alongside the forecast.

## Using your own data

1. Replace `data/stock_prices.csv` with your own CSV with two columns:
   `date` and `close` (any ticker, any frequency — just keep it consistent).
2. Re-run `python src/train.py`.

To regenerate a fresh synthetic series instead (different length, seed, or
starting price):

```bash
python data/generate_data.py --days 1500 --seed 7 --start-price 50
```

## How the features work

For each day, the model sees:
- **Lag features** (`lag_1` … `lag_n`): the closing prices from the last
  `n_lags` days (default 10).
- **Rolling statistics**: the mean and standard deviation of recent closes
  over several window sizes (default 5, 10, and 20 days), capturing
  short/medium-term trend and volatility.
- **Day of week**: captures any weekly seasonality.

All features are built strictly from *past* prices, so there's no leakage
of the value being predicted.

## Extending this project

- **Predict returns instead of price**: modify `features.py` to target
  `pct_change()` instead of raw `close` — often more stable for financial
  series.
- **Try other models**: swap `GradientBoostingRegressor` in `src/train.py`
  for `RandomForestRegressor`, `XGBRegressor`, or a linear model.
- **Add more series**: extend `data/stock_prices.csv` with a `ticker` column
  and adapt `utils.py`/`features.py` to train per-ticker or multi-ticker
  models.
- **Real data**: plug in a real historical price CSV (many brokerages and
  data providers offer CSV export) in place of the synthetic sample.
