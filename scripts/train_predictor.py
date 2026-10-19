"""Stage 1 of Algorithm 1 (MAL pre-training): train the predictive network f^pred, g^pred by L_pred.

Example:
    python scripts/train_predictor.py --set market=SP part=C seed=0

The checkpoint <output_dir>/<market>/predictor/part<P>_seed<s>/checkpoint.pt is
then passed to scripts/train_policy.py with ``mal=true predictor=<path>``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ras.config import PREDICTOR_DEFAULTS, load_config  # noqa: E402
from ras.trainer import train_predictor  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/predictor.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    args = parser.parse_args()
    print("results:", train_predictor(load_config(args.config, args.set, PREDICTOR_DEFAULTS)))
