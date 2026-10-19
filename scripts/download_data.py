"""Download the daily OHLC data of the S&P500 and CSI300 constituents from Yahoo Finance.

    python scripts/download_data.py --markets SP CSI --out data

Writes ``data/<market>.csv`` with the columns datetime, open, high, low, close,
asset. Prices are adjusted for splits and dividends with the ratio
Adj Close / Close. The constituent lists in ``configs/tickers`` are exactly the
475 (S&P500) and 222 (CSI300) stocks used in the paper, i.e. the constituents
with enough history in 2004-2023. Yahoo Finance revises adjusted prices over
time, so a fresh download can differ slightly from the snapshot used in the
paper.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def yahoo_symbol(market, asset):
    if market == "CSI":  # sh600000 -> 600000.SS, sz000001 -> 000001.SZ
        return asset[2:] + (".SS" if asset.startswith("sh") else ".SZ")
    return asset


def download(market, asset, start, end, retries=3):
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.download(yahoo_symbol(market, asset), start=start, end=end, auto_adjust=False, progress=False)
            break
        except Exception as error:  # network errors
            if attempt == retries - 1:
                raise
            print(f"retry {asset}: {error}")
            time.sleep(5)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    factor = df["Adj Close"] / df["Close"]
    out = pd.DataFrame({"datetime": df["Date"], **{c.lower(): df[c] * factor for c in ("Open", "High", "Low", "Close")}})
    out["high"] = np.max(out[["open", "close", "high"]], axis=1)
    out["low"] = np.min(out[["open", "close", "low"]], axis=1)
    out["asset"] = asset
    return out.dropna()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["SP", "CSI"])
    parser.add_argument("--out", default="data")
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default="2024-01-01")
    parser.add_argument("--limit", type=int, default=0, help="only download the first N tickers (for testing)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for market in args.markets:
        assets = open(os.path.join(ROOT, "configs", "tickers", f"{market}.txt")).read().split()
        if args.limit:
            assets = assets[:args.limit]
        frames, missing = [], []
        for i, asset in enumerate(assets):
            df = download(market, asset, args.start, args.end)
            if df is None:
                missing.append(asset)
            else:
                frames.append(df)
            print(f"\r{market}: {i + 1}/{len(assets)}", end="", flush=True)
        data = pd.concat(frames)
        path = os.path.join(args.out, f"{market}.csv")
        data.to_csv(path, index=False)
        print(f"\n{market}: {data['asset'].nunique()} assets, {len(data)} rows -> {path}")
        if missing:
            print(f"{market}: no data for {len(missing)} tickers: {missing}")


if __name__ == "__main__":
    main()
