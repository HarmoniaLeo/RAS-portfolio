"""Market data and samples of the portfolio MDP (paper Sec. 2.2).

Prices are daily OHLC; ``bar_days`` consecutive days are aggregated into one
bar, so with ``bar_days=5`` an observation P_{t,m} is a weekly OHLC bar and a
decision step spans one week. Price arrays are stored channel-first in the order
``PRICE_FIELDS`` = (close, high, low, open); the released checkpoints were
trained with this channel order.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PRICE_FIELDS = ("close", "high", "low", "open")


def read_market_csv(data_dir, market, start, end):
    df = pd.read_csv(os.path.join(data_dir, f"{market}.csv"), dtype={"asset": str})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[(df["datetime"] >= start) & (df["datetime"] < end)]


def date_index(dates, date):
    return int((dates < pd.Timestamp(date)).sum())


class MarketData:
    """All M assets of a market on a common calendar, shape [4, T, M]."""

    def __init__(self, data_dir, market, start, val_start, test_start, end):
        df = read_market_csv(data_dir, market, start, end)
        close = df.pivot(index="datetime", columns="asset", values="close").ffill().bfill()
        fields = []
        for field in PRICE_FIELDS:
            values = df.pivot(index="datetime", columns="asset", values=field).reindex_like(close)
            fields.append(values.where(values.notna(), close).values)
        self.prices = np.stack(fields, axis=0).astype(np.float64)
        self.dates = close.index
        self.assets = list(close.columns)
        self.market = market
        self.val_start = date_index(self.dates, val_start)
        self.test_start = date_index(self.dates, test_start)

    @property
    def num_assets(self):
        return len(self.assets)


class AssetSeries:
    """Per-asset price series without calendar alignment (used to train f^pred)."""

    def __init__(self, data_dir, market, start, val_start, test_start, end):
        df = read_market_csv(data_dir, market, start, end)
        self.market = market
        self.assets, self.prices, self.dates, self.val_start, self.test_start = [], [], [], [], []
        for asset, df_m in df.groupby("asset", sort=False):
            dates = pd.DatetimeIndex(df_m["datetime"].values)
            self.assets.append(asset)
            self.prices.append(np.stack([df_m[f].values for f in PRICE_FIELDS], axis=0)[:, :, None].astype(np.float64))
            self.dates.append(dates)
            self.val_start.append(date_index(dates, val_start))
            self.test_start.append(date_index(dates, test_start))


def to_bars(window, bar_days):
    """Aggregate daily [4, l * bar_days, M] prices into bars [4, l, M]."""
    window = window.reshape(window.shape[0], -1, bar_days, window.shape[-1])
    return np.stack([window[0, :, -1], window[1].max(axis=1), window[2].min(axis=1), window[3, :, 0]], axis=0)


def sample_starts(length, val_start, test_start, split, window, bar_days, min_history, step, train):
    """First day of the price windows of a split (one window per decision step).

    The decision step of a window starting on day ``j`` happens on day
    ``j + window * bar_days`` and earns the price movement of the next bar.
    In training (``train=True``) every trading day is a possible episode start.
    """
    span = window * bar_days
    first = (min_history - window) * bar_days
    if split == "train":
        stop = val_start - span
        return range(first, stop) if train else range(first, stop - step + 1, step)
    if split == "val":
        return range(max(first, val_start - span), max(first, test_start - span) - step + 1, step)
    return range(max(first, test_start - span), length - span - bar_days + 1, step)


class PortfolioDataset(Dataset):
    """Decision steps of the portfolio MDP.

    Each item holds the price patterns of all M assets, the price movement
    r_{t,m} of the next bar and the asset mask. During training ARM masks every
    non-cash asset with probability ``gamma`` (paper Sec. 4.2): a masked asset
    gets ``asset_mask=False`` and its score is replaced by -Y in the policy.
    """

    def __init__(self, data, split, cfg, train=False):
        self.data = data
        self.window = cfg["window"]
        self.bar_days = cfg["bar_days"]
        self.gamma = float(cfg["gamma"]) if (train and split == "train") else 0.0
        step = self.bar_days  # one decision step per bar (trading period = 1 bar)
        self.starts = list(sample_starts(data.prices.shape[1], data.val_start, data.test_start, split,
                                         self.window, self.bar_days, cfg["min_history"], step, train))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        j = self.starts[index]
        t = j + self.window * self.bar_days
        patterns = to_bars(self.data.prices[:, j:t], self.bar_days)
        close = self.data.prices[0]
        movement = close[t + self.bar_days - 1] / close[t - 1]
        asset_mask = np.ones(self.data.num_assets, dtype=bool)
        if self.gamma > 0:
            for m in range(self.data.num_assets):
                if random.random() < self.gamma:
                    patterns[:, :, m] = 0
                    movement[m] = 0
                    asset_mask[m] = False
        return {
            "patterns": torch.tensor(patterns, dtype=torch.float32),   # [4, l, M]
            "movement": torch.tensor(movement, dtype=torch.float32),   # r_{t,1..M}
            "asset_mask": torch.tensor(asset_mask),
            "t": t,
            "date": int(self.data.dates[t].timestamp()),
        }


class PatternDataset(Dataset):
    """Single-asset price patterns with their next price movement (MAL pre-training)."""

    def __init__(self, series, split, cfg, train=False):
        self.series = series
        self.window = cfg["window"]
        self.bar_days = cfg["bar_days"]
        self.items = []
        for m in range(len(series.assets)):
            starts = sample_starts(series.prices[m].shape[1], series.val_start[m], series.test_start[m], split,
                                   self.window, self.bar_days, cfg["min_history"], 1, train)
            self.items += [(m, j) for j in starts]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        m, j = self.items[index]
        prices = self.series.prices[m]
        t = j + self.window * self.bar_days
        return {
            "patterns": torch.tensor(to_bars(prices[:, j:t], self.bar_days), dtype=torch.float32),  # [4, l, 1]
            "movement": torch.tensor(prices[0, t + self.bar_days - 1] / prices[0, t - 1], dtype=torch.float32),  # [1]
            "asset_mask": torch.ones(1, dtype=torch.bool),
            "asset": m,
            "date": int(self.series.dates[m][t].timestamp()),
        }
