"""Shared helpers for loading config and data used across the project."""


import os
import yaml
import pandas as pd


def load_config(config_path="config.yaml"):
    """Load the YAML config file into a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_price_series(data_path):
    """Load a CSV with `date,close` columns, sorted chronologically."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find data at '{data_path}'. "
            "Run `python data/generate_data.py` to create it, "
            "or point config.yaml's data_path at your own CSV with 'date' and 'close' columns."
        )

    df = pd.read_csv(data_path, parse_dates=["date"])

    required_cols = {"date", "close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset must have columns {required_cols}, found {set(df.columns)}"
        )

    df = df.sort_values("date").reset_index(drop=True)
    return df
