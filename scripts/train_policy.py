"""Stage 2 of Algorithm 1: train an EIIE policy (optionally with ARM and MAL) and test it.

Examples:
    # Transformer EIIE + RAS (MAL on, gamma = 0.5) on S&P500, Part C, seed 0
    python scripts/train_policy.py --config configs/transformer_eiie.yaml \
        --set market=SP part=C seed=0 gamma=0.5 mal=true \
              predictor=outputs/SP/predictor/partC_seed0/checkpoint.pt

    # two GPUs, as in the paper
    torchrun --nproc_per_node 2 scripts/train_policy.py --config configs/rat.yaml --set market=CSI part=A

Results are written to <output_dir>/<market>/<run name>/part<P>_seed<s>/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ras.config import load_config  # noqa: E402
from ras.trainer import train_policy  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/transformer_eiie.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    args = parser.parse_args()
    print("results:", train_policy(load_config(args.config, args.set)))
