"""Test trained checkpoints: trade over the train/val/test splits and write the results.

Examples:
    # re-evaluate one released run and compare with the released returns
    python scripts/evaluate.py --runs results/paper/SP/transformer_eiie_mal_gamma0.5/partC_seed0 --out outputs/reeval

    # all released runs of a market
    python scripts/evaluate.py --runs "results/paper/CSI/*/part*_seed*" --out outputs/reeval
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ras.config import run_dir, split_dates  # noqa: E402
from ras.data import AssetSeries, MarketData  # noqa: E402
from ras.trainer import (SPLITS, evaluate_policy, evaluate_predictor, load_policy, load_predictor,  # noqa: E402
                         save_policy_results, save_predictor_results)


def evaluate_predictor_run(directory, args):
    ckpt = torch.load(os.path.join(directory, "checkpoint.pt"), map_location="cpu")
    cfg = dict(ckpt["config"], num_workers=4)
    predictor = load_predictor(os.path.join(directory, "checkpoint.pt"), args.device).to(args.device)
    series = AssetSeries(args.data_dir, cfg["market"], *split_dates(cfg))
    results = {split: evaluate_predictor(predictor, series, split, cfg, args.device) for split in SPLITS}
    out = run_dir(dict(cfg, output_dir=args.out))
    os.makedirs(out, exist_ok=True)
    save_predictor_results(out, series, results)
    new = json.load(open(os.path.join(out, "metrics.json")))["test"]
    line = f"{directory}: test MSE {new['mse']:.6f} Pearson {new['pearson']:.4f}"
    reference = os.path.join(directory, "predictions_test.npz")
    if os.path.exists(reference):
        old, cur = np.load(reference), np.load(os.path.join(out, "predictions_test.npz"))
        if old["z_hat"].shape == cur["z_hat"].shape:
            line += f" | max |z_new - z_released| = {np.nanmax(np.abs(old['z_hat'] - cur['z_hat'])):.2e}"
    print(line, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run directories (glob patterns allowed)")
    parser.add_argument("--out", default="outputs/reeval", help="where to write the new results")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    runs = sorted(d for pattern in args.runs for d in glob.glob(pattern) if os.path.exists(os.path.join(d, "checkpoint.pt")))
    cache = {}
    for directory in runs:
        if directory.rstrip("/").split("/")[-2] == "predictor":
            evaluate_predictor_run(directory, args)
            continue
        policy, cfg = load_policy(os.path.join(directory, "checkpoint.pt"), args.device)
        policy.to(args.device)
        key = (cfg["market"], cfg["part"])
        if key not in cache:
            cache[key] = MarketData(args.data_dir, cfg["market"], *split_dates(cfg))
        data = cache[key]
        results = {split: evaluate_policy(policy, data, split, cfg, args.device) for split in SPLITS}
        out = run_dir(dict(cfg, output_dir=args.out))
        os.makedirs(out, exist_ok=True)
        summary = save_policy_results(out, data, cfg, results)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        line = f"{directory}: test AR {summary['test']['AR']:.4f} ASR {summary['test']['ASR']:.4f} ACR {summary['test']['ACR']:.4f}"
        reference = os.path.join(directory, "returns_test.csv")
        if os.path.exists(reference):
            old = pd.read_csv(reference)["return"].values
            new = results["test"]["return"]
            line += f" | max |R_new - R_released| = {np.max(np.abs(new - old)):.2e}" if len(old) == len(new) else " | length mismatch"
        print(line, flush=True)


if __name__ == "__main__":
    main()
