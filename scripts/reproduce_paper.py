"""Reproduce the tables and figures of the experiments section from a results directory.

    python scripts/reproduce_paper.py --results results/paper --out outputs/paper

Writes ``tables.md`` and the figures ``compare.pdf`` (accumulated log returns of
all methods) and ``ablation.pdf`` (Transformer EIIE under different settings).
``--results`` can also point to your own runs (``outputs`` by default after
``scripts/run_all.sh``), as long as the same configurations exist.
"""
import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ras import analysis  # noqa: E402
from ras.config import MARKETS  # noqa: E402

warnings.filterwarnings("ignore")


def plot_curves(curves_by_market, keys, path):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(len(curves_by_market), 1, figsize=(8, 5 * len(curves_by_market)))
    axes = list(axes) if len(curves_by_market) > 1 else [axes]
    for ax, (market, curves) in zip(axes, curves_by_market.items()):
        palette = sns.hls_palette(len(keys), l=0.6, s=0.5)
        for color, key in zip(palette, keys):
            if key in curves:
                curves[key].plot(ax=ax, label=key, color=color)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(MARKETS[market], loc="left")
        ax.set_ylabel("Accumulated Log Return")
        ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/paper")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="outputs/paper")
    parser.add_argument("--markets", nargs="+", default=["SP", "CSI"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sections, compare_curves, ablation_curves = [], {}, {}
    for market in args.markets:
        res = analysis.Results(args.results, market)
        close = analysis.load_close(args.data_dir, market)
        name = MARKETS[market]
        print(f"[{name}] selected gamma for + RAS per part:", res.select_ras(), flush=True)

        rows, compare_curves[market] = analysis.comparison_table(res, close)
        sections.append(f"## Performance comparison on {name}\n\n{analysis.to_markdown(rows)}")
        rows = analysis.mal_table(res, close)
        sections.append(f"## Standalone evaluation of MAL on {name}\n\n{analysis.to_markdown(rows)}")
        rows = analysis.regime_table(res, close)
        sections.append(f"## Performance across market regimes on {name}\n\n{analysis.to_markdown(rows)}")
        rows, ablation_curves[market] = analysis.ablation_table(res, close)
        sections.append(f"## Ablation of Transformer EIIE on {name}\n\n{analysis.to_markdown(rows)}")
        print("\n\n".join(sections[-4:]), flush=True)

    with open(os.path.join(args.out, "tables.md"), "w") as f:
        f.write("# Reproduced tables\n\n" + "\n\n".join(sections) + "\n")
    plot_curves(compare_curves, list(analysis.BENCHMARKS) + list(analysis.BASELINES.values())
                + ["Transformer EIIE", "Transformer EIIE + RAS"], os.path.join(args.out, "compare.pdf"))
    plot_curves(ablation_curves, [f"{m} MAL, γ = {g}" for m in ("w/o", "with") for g in analysis.GAMMAS],
                os.path.join(args.out, "ablation.pdf"))
    print(f"Tables and figures written to {args.out}")


if __name__ == "__main__":
    main()
