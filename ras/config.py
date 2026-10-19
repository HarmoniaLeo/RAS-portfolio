"""Experiment configuration.

Notation follows the paper (Sec. 2.2 and Sec. 5.2):

    l                   observation window size (``window``)
    c                   transaction cost rate (``transaction_cost``)
    d, d_pred           model dimensions (``d``)
    N                   number of Transformer layers (``num_layers``)
    gamma               ARM masking ratio (``gamma``), 0 disables ARM
    Y                   large constant used to mask a score (``mask_constant``)
    alpha               learning rate (``lr``)

A configuration is a flat dict. YAML files in ``configs/`` hold the defaults of
each method; any key can be overridden from the command line with
``--set key=value``.
"""
import ast
import copy
import json
import os

import yaml

# Data partitions of Table 1: (validation start, test start, end).
# The train set always starts on ``start`` (2004-01-01).
PARTS = {
    "A": ("2008-01-01", "2012-01-01", "2016-01-01"),
    "B": ("2012-01-01", "2016-01-01", "2020-01-01"),
    "C": ("2016-01-01", "2020-01-01", "2024-01-01"),
}

MARKETS = {"SP": "S&P500", "CSI": "CSI300"}

POLICY_MODELS = ["transformer_eiie", "cnn_eiie", "brnn_eiie", "rat", "hgam", "lsre_caan"]

DEFAULTS = {
    # data
    "market": "SP",
    "part": "C",
    "data_dir": "data",
    "start": "2004-01-01",
    "window": 50,               # l: number of bars in a price pattern
    "bar_days": 5,              # trading days aggregated into one bar (weekly OHLC)
    "min_history": 50,
    # trading
    "transaction_cost": 0.0006,  # c
    # network
    "model": "transformer_eiie",
    "d": 32,
    "num_layers": 2,
    "num_heads": 2,
    "input_kernel": 5,
    "causal": False,
    # RAS
    "gamma": 0.0,               # ARM masking ratio
    "mask_constant": 1e8,       # Y
    "mal": False,
    "predictor": "",            # path of a trained MAL predictor checkpoint
    "holding_in_score": False,  # concatenate w_{t,m} to h~_{t,m} in g, see ras/policy.py
    # optimisation
    "seed": 0,
    "epochs": 1000,
    "eval_every": 10,
    "batch_size": 64,
    "lr": 1e-4,                 # alpha
    "weight_decay": 5e-8,
    "restart_prob": 0.05,       # probability of resetting w_t to cash (episode restart)
    "num_workers": 4,
    "device": "cuda",
    "output_dir": "outputs",
}

PREDICTOR_DEFAULTS = dict(DEFAULTS, model="predictor", epochs=500, batch_size=4096)


def _parse_value(text):
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError):
            pass
    return text


def load_config(path=None, overrides=(), defaults=DEFAULTS):
    cfg = copy.deepcopy(defaults)
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    for item in overrides:
        key, value = item.split("=", 1)
        if key not in cfg:
            raise KeyError(f"Unknown config key: {key}")
        cfg[key] = _parse_value(value)
    if cfg["part"] not in PARTS:
        raise ValueError(f"part must be one of {list(PARTS)}")
    return cfg


def split_dates(cfg):
    val_start, test_start, end = PARTS[cfg["part"]]
    return cfg["start"], val_start, test_start, end


def run_name(cfg):
    """Directory name of a run, e.g. ``transformer_eiie_mal_gamma0.5``."""
    if cfg["model"] == "predictor":
        return "predictor"
    if cfg["model"] != "transformer_eiie":
        return cfg["model"]
    name = "transformer_eiie_mal" if cfg["mal"] else "transformer_eiie"
    return f"{name}_gamma{float(cfg['gamma'])}"


def run_dir(cfg):
    return os.path.join(cfg["output_dir"], cfg["market"], run_name(cfg), f"part{cfg['part']}_seed{cfg['seed']}")
