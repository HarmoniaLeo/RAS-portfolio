"""Visualise trained runs: accumulated log return on the test split and the portfolio allocation.

    python scripts/plot_run.py --runs "outputs/CSI/*/partA_seed0" --out outputs/plots/csi_partA.pdf
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ras import metrics  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="run directories (glob patterns allowed)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top", type=int, default=10, help="assets shown in the allocation plot")
    parser.add_argument("--out", default="outputs/plots/runs.pdf")
    args = parser.parse_args()
    runs = sorted(d for p in args.runs for d in glob.glob(p) if os.path.exists(os.path.join(d, f"returns_{args.split}.csv")))
    if not runs:
        sys.exit("no runs found")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, (ax_ret, ax_alloc) = plt.subplots(2, 1, figsize=(9, 9))
    for directory in runs:
        R = pd.read_csv(os.path.join(directory, f"returns_{args.split}.csv"), parse_dates=["date"]).set_index("date")["return"]
        label = "/".join(directory.rstrip("/").split("/")[-2:])
        ax_ret.plot(R.index, np.log(R).cumsum(), label=f"{label} (ASR {metrics.annual_sharpe(R.values):.2f})")
        print(f"{label}: " + ", ".join(f"{k} {v:.4f}" for k, v in metrics.performance(R.values).items()))
    ax_ret.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax_ret.set_ylabel("Accumulated Log Return")
    ax_ret.set_title(f"{args.split} split")
    ax_ret.legend(frameon=False, fontsize=7)

    scores = os.path.join(runs[0], "scores_test.npz")
    if args.split == "test" and os.path.exists(scores) and "allocations" in np.load(scores).files:
        npz = np.load(scores, allow_pickle=True)
        alloc = pd.DataFrame(npz["allocations"], index=pd.to_datetime(npz["dates"]), columns=["cash"] + list(npz["assets"]))
        top = alloc.mean().sort_values(ascending=False).index[:args.top]
        shown = alloc[top].assign(others=alloc.drop(columns=top).sum(axis=1))
        ax_alloc.stackplot(shown.index, shown.T.values, labels=shown.columns)
        ax_alloc.set_ylabel("Portfolio allocation a_t")
        ax_alloc.set_title("/".join(runs[0].rstrip("/").split("/")[-2:]))
        ax_alloc.legend(loc="upper left", fontsize=6, ncol=4, frameon=False)
    else:
        ax_alloc.axis("off")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"Figure written to {args.out}")


if __name__ == "__main__":
    main()
