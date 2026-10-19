"""Export the paper's runs from the original experiment code base (maintainers only).

The original code identified a run by the SHA-1 of its configuration. This
script recomputes those identifiers from the original configuration files,
selects the runs reported in the paper (S&P500 and CSI300, weekly bars) and
writes them in the layout produced by ``scripts/train_policy.py`` and
``scripts/train_predictor.py``:

    results/paper/<market>/<run name>/part<P>_seed<s>/
        config.json  metrics.json  returns_{train,val,test}.csv
        scores_test.npz (Transformer EIIE)  predictions_test.npz (MAL predictor)
        checkpoint.pt (converted to the module names of this repository)

Usage:
    python tools/export_legacy.py --legacy_dir ../PortModelBased --out results/paper
"""
import argparse
import copy
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ras import metrics  # noqa: E402
from ras.config import DEFAULTS, PREDICTOR_DEFAULTS, run_dir  # noqa: E402

PART_OF_END = {"2016/01/01": "A", "2020/01/01": "B", "2024/01/01": "C"}
BASELINES = {"EIIEConv": "cnn_eiie", "EIIELSTM": "brnn_eiie", "RAT": "rat", "HGAM": "hgam", "LSRECAAN": "lsre_caan"}


# ----------------------------------------------------------------------------- legacy configs

def legacy_configs(config_dir, name):
    """Expand a configuration file of the original code base (copied from its utils/analysis.py)."""
    raw = json.load(open(f"{config_dir}/{name}.json"))

    def roll_out(config):
        buf = {}
        if "father" in config:
            buf = roll_out(json.load(open(f"{config_dir}/{config['father']}.json")))
            del config["father"]
        for key in config:
            for key_buf in key.split("&"):
                for key_father in list(buf):
                    if key_buf in key_father.split("&"):
                        del buf[key_father]
            buf[key] = config[key]
        return buf

    raw = roll_out(raw)
    configs = [{}]
    for key, values in raw.items():
        if isinstance(values, list):
            n = len(configs)
            for j, value in enumerate(values):
                for i in range(n):
                    target = configs[i].copy() if j > 0 else configs[i]
                    if "&" in key:
                        for k, key_ in enumerate(key.split("&")):
                            target[key_] = value[k]
                    else:
                        target[key] = value
                    if j > 0:
                        configs.append(target)
        else:
            for config in configs:
                config[key] = values
    for config in configs:
        for key in [k for k, v in config.items() if v == ""]:
            del config[key]
    for config in configs:
        config["model id"] = hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return configs


# ----------------------------------------------------------------------------- checkpoint conversion

TRANSFORMER_RULES = [
    (r"input_projection\.", "input_conv."),
    (r"rnn\.(\d+)\.attn\.self_attention\.fc_q\.", r"layers.\1.self_attn.q_proj."),
    (r"rnn\.(\d+)\.attn\.self_attention\.fc_k\.", r"layers.\1.self_attn.k_proj."),
    (r"rnn\.(\d+)\.attn\.self_attention\.fc_v\.", r"layers.\1.self_attn.v_proj."),
    (r"rnn\.(\d+)\.attn\.self_attention\.fc_o\.", r"layers.\1.self_attn.out_proj."),
    (r"rnn\.(\d+)\.attn\.self_attention_layer_norm\.", r"layers.\1.norm1."),
    (r"rnn\.(\d+)\.attn\.feed_forward\.", r"layers.\1.ffn."),
    (r"rnn\.(\d+)\.attn\.ffn_layer_norm\.", r"layers.\1.norm2."),
]


def convert_transformer(key):
    if re.match(r"rnn\.\d+\.pos\.pe$", key):
        return None  # deterministic positional encoding, recomputed at construction
    for pattern, repl in TRANSFORMER_RULES:
        if re.match(pattern, key):
            return re.sub(pattern, repl, key, count=1)
    raise KeyError(key)


def convert_policy_state(state, model):
    new = {}
    for key, value in state.items():
        if key.startswith("pretrained_backbone."):
            sub = convert_transformer(key[len("pretrained_backbone."):])
            new_key = None if sub is None else "predictor.f_pred." + sub
        elif key.startswith("backbone."):
            sub = key[len("backbone."):]
            if model == "transformer_eiie":
                sub = convert_transformer(sub)
            new_key = None if sub is None else "f." + sub
        elif key.startswith("mergeConv."):
            new_key = "fusion." + key[len("mergeConv."):]
        elif key.startswith("outputConv."):
            new_key = "g." + key[len("outputConv."):]
        elif key == "bias":
            new_key = "cash_bias"
        elif model == "lsre_caan":
            new_key = "f." + key
        else:
            raise KeyError(key)
        if new_key is not None:
            new[new_key] = value
    return new


def convert_predictor_state(state):
    new = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            sub = convert_transformer(key[len("backbone."):])
            if sub is not None:
                new["f_pred." + sub] = value
        elif key.startswith("outputConv."):
            new["g_pred." + key[len("outputConv."):]] = value
        else:
            raise KeyError(key)
    return new


# ----------------------------------------------------------------------------- export

def read_returns(log_dir, mid, key, split):
    df = pd.read_csv(f"{log_dir}/{mid}_{key}_{split}_details.csv", parse_dates=["date"])
    return pd.DataFrame({"date": df["date"].dt.strftime("%Y-%m-%d"), "return": df["return"].values})


def export_policy(args, legacy, cfg):
    mid, key = legacy["model id"], f"{cfg['market']}_5"
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        returns = read_returns(args.log_dir, mid, key, split)
        returns.to_csv(f"{out}/returns_{split}.csv", index=False)
        summary[split] = metrics.performance(returns["return"].values)
        old = json.load(open(f"{args.log_dir}/{mid}_{key}_{split}_metrics.json"))
        for name in ("AR", "ASR", "ACR"):
            assert np.isclose(summary[split][name], old[f"{key}_{name}"], rtol=1e-4, atol=1e-5), (mid, split, name)
    json.dump(summary, open(f"{out}/metrics.json", "w"), indent=2)
    if cfg["model"] == "transformer_eiie":
        scores = pd.read_csv(f"{args.log_dir}/{mid}_{key}_test_{key}_raw_scores.csv", index_col=0)
        np.savez_compressed(f"{out}/scores_test.npz", dates=pd.to_datetime(scores.index).strftime("%Y-%m-%d").values,
                            assets=np.array([c.split("_", 1)[1] for c in scores.columns[1:]]),
                            scores=scores.values.astype(np.float32))
    state = torch.load(f"{args.ckpt_dir}/{mid}_{key}_model.pth", map_location="cpu")
    # whether w_{t,m} enters the score head, read from the weights
    cfg["holding_in_score"] = state["outputConv.weight"].shape[1] == cfg["d"] + 1
    if args.checkpoints:
        new_state = convert_policy_state(state, cfg["model"])
        if cfg["mal"]:  # the original checkpoints only stored f^pred of the frozen predictor
            predictor = torch.load(cfg["predictor"], map_location="cpu")["state_dict"]
            new_state.update({f"predictor.{k}": v for k, v in predictor.items() if k.startswith("g_pred.")})
        torch.save({"config": cfg, "state_dict": new_state}, f"{out}/checkpoint.pt")
    json.dump(dict(cfg, legacy_id=mid), open(f"{out}/config.json", "w"), indent=2)


def export_predictor(args, legacy, cfg):
    mid, key = legacy["model id"], f"{cfg['market']}_5"
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    summary = {}
    for split in ("train", "val", "test"):
        old = json.load(open(f"{args.log_dir}/{mid}_{key}_{split}_metrics.json"))
        summary[split] = {"mse": old[f"{key}_raw-mse"], "pearson": old[f"{key}_raw-corr"]}
    json.dump(summary, open(f"{out}/metrics.json", "w"), indent=2)
    df = pd.read_csv(f"{args.log_dir}/{mid}_{key}_test_details.csv", parse_dates=["date"])
    table = df.pivot(index="date", columns="codes", values="pred_raw")
    np.savez_compressed(f"{out}/predictions_test.npz", dates=table.index.strftime("%Y-%m-%d").values,
                        assets=np.array([c.split("_", 1)[1] for c in table.columns]), z_hat=table.values.astype(np.float32))
    state = torch.load(f"{args.ckpt_dir}/{mid}_{key}_model.pth", map_location="cpu")
    torch.save({"config": cfg, "state_dict": convert_predictor_state(state)}, f"{out}/checkpoint.pt")
    json.dump(dict(cfg, legacy_id=mid), open(f"{out}/config.json", "w"), indent=2)


def base_config(legacy, defaults, out_dir):
    cfg = copy.deepcopy(defaults)
    cfg.update(market=legacy["dataset"], part=PART_OF_END[legacy["end"]], seed=legacy["rand seed"],
               d=legacy["model dim"], num_layers=legacy["layer num"], input_kernel=legacy.get("input kernel", 1),
               lr=legacy["learning rate"], batch_size=legacy["batch size"], epochs=legacy["total epoch"],
               restart_prob=legacy.get("episode random start rate", 0.0), output_dir=out_dir)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy_dir", default="../PortModelBased")
    parser.add_argument("--out", default="results/paper")
    parser.add_argument("--markets", nargs="+", default=["SP", "CSI"])
    parser.add_argument("--checkpoints", type=int, default=1)
    args = parser.parse_args()
    args.config_dir = f"{args.legacy_dir}/config"
    args.log_dir = f"{args.legacy_dir}/log_modelbased"
    args.ckpt_dir = f"{args.legacy_dir}/checkpoint_modelbased"

    def selected(c):
        return c["dataset"] in args.markets and c["freq"] == "5"

    predictors = {}
    for legacy in legacy_configs(args.config_dir, "forecast_for_manage"):
        if selected(legacy) and legacy["loss normed"] == 0:
            cfg = base_config(legacy, PREDICTOR_DEFAULTS, args.out)
            export_predictor(args, legacy, cfg)
            predictors[legacy["model id"]] = os.path.join(run_dir(cfg), "checkpoint.pt")
    print("predictors:", len(predictors))

    count = 0
    for legacy in legacy_configs(args.config_dir, "manage_compares"):
        if selected(legacy):
            cfg = base_config(legacy, DEFAULTS, args.out)
            cfg.update(model=BASELINES[legacy["baseline"]], gamma=0.0)
            export_policy(args, legacy, cfg)
            count += 1
    for legacy in legacy_configs(args.config_dir, "manage_baselines_random_masking"):
        if selected(legacy):
            cfg = base_config(legacy, DEFAULTS, args.out)
            cfg.update(model="transformer_eiie", gamma=float(legacy.get("random masking", 0.0)))
            export_policy(args, legacy, cfg)
            count += 1
    for legacy in legacy_configs(args.config_dir, "manage_pretrain_market_masking"):
        pretrain = legacy["pretrain model hash"]
        if selected(legacy) and pretrain.split("_")[0] in predictors and pretrain.endswith(f"_{legacy['dataset']}_5"):
            cfg = base_config(legacy, DEFAULTS, args.out)
            cfg.update(model="transformer_eiie", gamma=float(legacy.get("random masking", 0.0)), mal=True,
                       predictor=predictors[pretrain.split("_")[0]])
            cfg["predictor_config"] = torch.load(cfg["predictor"], map_location="cpu")["config"]
            export_policy(args, legacy, cfg)
            count += 1
    print("policies:", count)


if __name__ == "__main__":
    main()
