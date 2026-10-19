"""Training and evaluation (paper Algorithm 1 and Sec. 5.2).

Stage 1 (MAL pre-training): ``train_predictor`` fits f^pred, g^pred by L_pred.
Stage 2 (policy learning):  ``train_policy`` maximises the reward log R_t by
deterministic policy gradient, with ARM masking and the frozen MAL predictor.

The policy is validated every ``eval_every`` epochs and the checkpoint with the
best validation ASR is kept. Testing runs the same forward pass sequentially
over a split, without masking and without parameter updates.

Multi-GPU training is supported through ``torchrun`` (DistributedDataParallel);
the paper's runs used two GPUs.
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from ras import metrics
from ras.config import run_dir, split_dates
from ras.data import AssetSeries, MarketData, PatternDataset, PortfolioDataset
from ras.policy import build_policy
from ras.portfolio import PortfolioVectorMemory, cash_holding, step
from ras.predictor import PricePredictor, prediction_loss

SPLITS = ("train", "val", "test")


# ----------------------------------------------------------------------------- utilities

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_device(cfg):
    """Return (rank, world_size, device); initialises torch.distributed under torchrun."""
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)
    device = cfg["device"] if (cfg["device"] != "cuda" or torch.cuda.is_available()) else "cpu"
    return 0, 1, torch.device(device)


def train_loader(dataset, cfg, world_size):
    sampler = DistributedSampler(dataset) if world_size > 1 else RandomSampler(dataset)
    return DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler, drop_last=True,
                      num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0)


def save_checkpoint(path, cfg, model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save({"config": cfg, "state_dict": model.state_dict()}, path)


def load_predictor(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    predictor = PricePredictor(ckpt["config"])
    predictor.load_state_dict(ckpt["state_dict"])
    return predictor


def load_policy(path, device="cpu"):
    """Load a policy checkpoint (MAL policies contain their frozen predictor)."""
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    predictor = PricePredictor(cfg.get("predictor_config") or cfg) if cfg["mal"] else None
    policy = build_policy(cfg, predictor)
    policy.load_state_dict(ckpt["state_dict"])
    return policy, cfg


def _to_dates(timestamps):
    return pd.to_datetime(np.asarray(timestamps), unit="s")


# ----------------------------------------------------------------------------- policy

@torch.no_grad()
def evaluate_policy(policy, data, split, cfg, device):
    """Trade sequentially over a split, starting from all cash.

    Returns a dict with the dates, the returns R_t, the scores (cash first) and
    the allocations a_t of every decision step.
    """
    policy.eval()
    dataset = PortfolioDataset(data, split, cfg, train=False)
    holding = cash_holding(1, data.num_assets, device)
    out = {"date": [], "return": [], "scores": [], "allocation": []}
    for i in range(len(dataset)):
        item = dataset[i]
        outputs = policy(item["patterns"][None].to(device), holding, item["asset_mask"][None].to(device))
        R, holding = step(outputs["allocation"], item["movement"][None].to(device), holding, cfg["transaction_cost"])
        out["date"].append(item["date"])
        out["return"].append(R.item())
        out["scores"].append(outputs["scores"][0].cpu().numpy())
        out["allocation"].append(outputs["allocation"][0].cpu().numpy())
    out["date"] = _to_dates(out["date"])
    out["return"] = np.array(out["return"])
    out["scores"] = np.stack(out["scores"]).astype(np.float32)
    out["allocation"] = np.stack(out["allocation"]).astype(np.float32)
    return out


def save_policy_results(directory, data, cfg, results):
    """Write returns_{split}.csv, metrics.json and scores_test.npz of a run."""
    periods = metrics.periods_per_year(cfg["bar_days"])
    summary = {}
    for split, res in results.items():
        pd.DataFrame({"date": res["date"], "return": res["return"]}).to_csv(
            os.path.join(directory, f"returns_{split}.csv"), index=False)
        summary[split] = metrics.performance(res["return"], periods)
    with open(os.path.join(directory, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    test = results["test"]
    np.savez_compressed(os.path.join(directory, "scores_test.npz"), dates=test["date"].strftime("%Y-%m-%d").values,
                        assets=np.array(data.assets), scores=test["scores"], allocations=test["allocation"])
    return summary


def train_policy(cfg):
    rank, world_size, device = setup_device(cfg)
    set_seed(cfg["seed"])
    directory = run_dir(cfg)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    data = MarketData(cfg["data_dir"], cfg["market"], *split_dates(cfg))
    predictor = None
    if cfg["mal"]:
        predictor = load_predictor(cfg["predictor"])
        cfg["predictor_config"] = torch.load(cfg["predictor"], map_location="cpu")["config"]
    policy = build_policy(cfg, predictor).to(device)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg["weight_decay"])
    model = DistributedDataParallel(policy, device_ids=[device.index]) if world_size > 1 else policy

    loader = train_loader(PortfolioDataset(data, "train", cfg, train=True), cfg, world_size)
    memory = PortfolioVectorMemory(len(data.dates), data.num_assets)
    periods = metrics.periods_per_year(cfg["bar_days"])
    best_asr, logs = -np.inf, []
    ckpt_path = os.path.join(directory, "checkpoint.pt")

    for epoch in range(cfg["epochs"]):
        if world_size > 1:
            loader.sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for batch in loader:
            holding = memory.read(batch["t"], cfg["restart_prob"]).to(device)
            outputs = model(batch["patterns"].to(device), holding, batch["asset_mask"].to(device))
            R, next_holding = step(outputs["allocation"], batch["movement"].to(device), holding, cfg["transaction_cost"])
            loss = -torch.log(R).mean()  # gradient ascent on the reward log R_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            memory.write(batch["t"] + cfg["bar_days"], next_holding)
            total += loss.item()

        if rank == 0 and (epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1):
            val = metrics.performance(evaluate_policy(policy, data, "val", cfg, device)["return"], periods)
            logs.append({"epoch": epoch, "loss": total / max(len(loader), 1), **{f"val_{k}": v for k, v in val.items()}})
            print(f"epoch {epoch}: loss {logs[-1]['loss']:.6f} val ASR {val['ASR']:.4f} AR {val['AR']:.4f}", flush=True)
            if np.isnan(val["ASR"]):
                break
            if val["ASR"] > best_asr:
                best_asr = val["ASR"]
                save_checkpoint(ckpt_path, cfg, policy)

    if rank == 0:
        pd.DataFrame(logs).to_csv(os.path.join(directory, "train_log.csv"), index=False)
        policy, _ = load_policy(ckpt_path, device)
        policy.to(device)
        results = {split: evaluate_policy(policy, data, split, cfg, device) for split in SPLITS}
        summary = save_policy_results(directory, data, cfg, results)
        print(json.dumps(summary, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()
    return directory


# ----------------------------------------------------------------------------- MAL predictor

@torch.no_grad()
def evaluate_predictor(predictor, series, split, cfg, device):
    """Predicted scores z^ and price movements r of all (asset, date) samples of a split."""
    predictor.eval()
    dataset = PatternDataset(series, split, cfg, train=False)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    z_hat, movement, asset, date = [], [], [], []
    for batch in loader:
        z_hat.append(predictor(batch["patterns"].to(device)).cpu().numpy()[:, 0])
        movement.append(batch["movement"].numpy()[:, 0])
        asset.append(batch["asset"].numpy())
        date.append(batch["date"].numpy())
    return {"z_hat": np.concatenate(z_hat), "movement": np.concatenate(movement),
            "asset": np.concatenate(asset), "date": _to_dates(np.concatenate(date))}


def predictor_metrics(res):
    return {"mse": float(np.mean((res["z_hat"] - res["movement"]) ** 2)),
            "pearson": float(np.corrcoef(res["z_hat"], res["movement"])[0, 1])}


def save_predictor_results(directory, series, results):
    with open(os.path.join(directory, "metrics.json"), "w") as f:
        json.dump({split: predictor_metrics(res) for split, res in results.items()}, f, indent=2)
    test = results["test"]
    table = pd.DataFrame({"date": test["date"], "asset": np.array(series.assets)[test["asset"]], "z_hat": test["z_hat"]})
    table = table.pivot(index="date", columns="asset", values="z_hat")
    np.savez_compressed(os.path.join(directory, "predictions_test.npz"), dates=table.index.strftime("%Y-%m-%d").values,
                        assets=table.columns.values.astype(str), z_hat=table.values.astype(np.float32))


def train_predictor(cfg):
    rank, world_size, device = setup_device(cfg)
    set_seed(cfg["seed"])
    directory = run_dir(cfg)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    series = AssetSeries(cfg["data_dir"], cfg["market"], *split_dates(cfg))
    predictor = PricePredictor(cfg).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg["weight_decay"])
    model = DistributedDataParallel(predictor, device_ids=[device.index]) if world_size > 1 else predictor
    loader = train_loader(PatternDataset(series, "train", cfg, train=True), cfg, world_size)
    best_corr, logs = -np.inf, []
    ckpt_path = os.path.join(directory, "checkpoint.pt")

    for epoch in range(cfg["epochs"]):
        if world_size > 1:
            loader.sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for batch in loader:
            mask = batch["asset_mask"].to(device)
            loss = prediction_loss(model(batch["patterns"].to(device), mask), batch["movement"].to(device), mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        if rank == 0 and (epoch % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1):
            val = predictor_metrics(evaluate_predictor(predictor, series, "val", cfg, device))
            logs.append({"epoch": epoch, "loss": total / max(len(loader), 1), **{f"val_{k}": v for k, v in val.items()}})
            print(f"epoch {epoch}: loss {logs[-1]['loss']:.6f} val MSE {val['mse']:.6f} Pearson {val['pearson']:.4f}", flush=True)
            if np.isnan(val["pearson"]):
                break
            if val["pearson"] > best_corr:  # model selection by validation Pearson correlation
                best_corr = val["pearson"]
                save_checkpoint(ckpt_path, cfg, predictor)

    if rank == 0:
        pd.DataFrame(logs).to_csv(os.path.join(directory, "train_log.csv"), index=False)
        predictor = load_predictor(ckpt_path, device).to(device)
        results = {split: evaluate_predictor(predictor, series, split, cfg, device) for split in SPLITS}
        save_predictor_results(directory, series, results)
    if dist.is_initialized():
        dist.destroy_process_group()
    return directory
