"""
Given all pipeline functions have side-effects or heavily leverage external libraries, unit testing isn't feasible. Instead the following script does some basic sanity checks to ensure data is being processed as expected.
"""
from pathlib import Path

import pandas as pd


data_dir = Path(__file__).resolve().parents[1] / "data"

orig = pd.read_csv(data_dir / "stocks" / "AAPL.csv")
raw = pd.read_parquet(data_dir / "market_data" / "AAPL.parquet")
feats = pd.read_parquet(data_dir / "market_data_with_features" / "AAPL.parquet")

# check parquet files have the same number of rows as the original csv
assert len(orig) == len(raw) == len(feats)

# check ibis features give the same result as pandas processing
assert feats.iloc[30]["vol_moving_avg"] - feats.iloc[0:31]["Volume"].mean() < 1e-6
assert feats.iloc[31]["vol_moving_avg"] - feats.iloc[1:32]["Volume"].mean() < 1e-6

assert feats.iloc[30]["adj_close_rolling_med"] - feats.iloc[0:31]["Adj Close"].median() < 1e-6
assert feats.iloc[31]["adj_close_rolling_med"] - feats.iloc[1:32]["Adj Close"].median() < 1e-6

assert feats.iloc[0]["next_day_volume"] == feats.iloc[1]["Volume"]
assert feats.iloc[1]["next_day_volume"] == feats.iloc[2]["Volume"]

# check correct number of rows are NaN
assert feats.iloc[0:30]["vol_moving_avg"].isna().sum() == 30
assert feats.iloc[0:30]["adj_close_rolling_med"].isna().sum() == 30
assert feats.iloc[31:35]["vol_moving_avg"].isna().sum() == 4
assert feats.iloc[31:35]["adj_close_rolling_med"].isna().sum() == 4
