"""Aggregation of runs into the tables and figures of the paper (Sec. 5.3-5.6).

A results directory has the layout ``<root>/<market>/<run name>/part<P>_seed<s>/``
(see ``ras.config.run_dir``). Test metrics of a seed are computed on the test
returns of Parts A, B and C concatenated (2012-2023); mean and standard
deviation are taken over the seeds.

The conventions follow the notebooks that produced the paper: the ASR of a
seed uses the sample standard deviation (ddof=1) of its returns, except in the
COVID-19 row of the regime table (ddof=0); the spread over seeds is the
population standard deviation (ddof=0) for AR/ASR/ACR/Train Reward and the
sample standard deviation (ddof=1) for Rank IC, Pearson, Mean Bias and Cohen's d.
"""
import json
import os

import numpy as np
import pandas as pd

from ras import metrics
from ras.config import PARTS
from ras.data import read_market_csv

SEEDS = (0, 64, 128)
GAMMAS = (0.0, 0.2, 0.5, 0.8)
BASELINES = {"cnn_eiie": "CNN EIIE", "brnn_eiie": "bRNN EIIE", "hgam": "HGAM", "lsre_caan": "LSRE-CAAN", "rat": "RAT"}
BENCHMARKS = ("CRP", "BAH", "EG", "OLMAR", "RMR")
PERIODS = metrics.periods_per_year(5)
COVID = ("2020-01-01", "2020-07-01")


def transformer_run(mal, gamma):
    return f"transformer_eiie{'_mal' if mal else ''}_gamma{float(gamma)}"


class Results:
    def __init__(self, root, market):
        self.root = os.path.join(root, market)
        self.market = market

    def path(self, run, part, seed, name):
        return os.path.join(self.root, run, f"part{part}_seed{seed}", name)

    def returns(self, run, part, seed, split):
        df = pd.read_csv(self.path(run, part, seed, f"returns_{split}.csv"), parse_dates=["date"])
        return df.set_index("date")["return"]

    def test_returns(self, run, seed, parts=tuple(PARTS)):
        return pd.concat([self.returns(run, p, seed, "test") for p in parts])

    def val_asr(self, run, part, seed):
        return json.load(open(self.path(run, part, seed, "metrics.json")))["val"]["ASR"]

    def scores(self, run, part, seed):
        """Test scores z_{t,m} (cash column dropped) as a DataFrame [dates x assets]."""
        npz = np.load(self.path(run, part, seed, "scores_test.npz"), allow_pickle=True)
        return pd.DataFrame(npz["scores"][:, 1:], index=pd.to_datetime(npz["dates"]), columns=npz["assets"])

    def predictions(self, part, seed):
        """Test predicted scores z^_{t,m} of the MAL predictor [dates x assets] (daily)."""
        npz = np.load(self.path("predictor", part, seed, "predictions_test.npz"), allow_pickle=True)
        return pd.DataFrame(npz["z_hat"], index=pd.to_datetime(npz["dates"]), columns=npz["assets"])

    def select_ras(self):
        """RAS (MAL on): per part, the gamma with the best validation ASR averaged over seeds."""
        best = {}
        for part in PARTS:
            asr = {g: np.mean([self.val_asr(transformer_run(True, g), part, s) for s in SEEDS]) for g in GAMMAS}
            best[part] = max(asr, key=asr.get)
        return best

    def ras_returns(self, seed, split="test"):
        best = self.select_ras()
        return [self.returns(transformer_run(True, best[p]), p, seed, split) for p in PARTS]


def summarize(values, ddof=0):
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(ddof=ddof)


def performance_over_seeds(returns_by_seed, ddof=1):
    rows = {k: [] for k in ("AR", "ASR", "ACR")}
    for R in returns_by_seed:
        rows["AR"].append(metrics.annual_return(R, PERIODS))
        rows["ASR"].append(metrics.annual_sharpe(R, PERIODS, ddof=ddof))
        rows["ACR"].append(metrics.annual_calmar(R, PERIODS))
    return {k: summarize(v) for k, v in rows.items()}


def train_reward_runs(res, run):
    """Train Reward of every (part, seed) run of a configuration."""
    return [metrics.train_reward(res.returns(run, p, s, "train")) for s in SEEDS for p in PARTS]


# ----------------------------------------------------------------------------- market data helpers

def load_close(data_dir, market, start="2004-01-01", end="2024-01-02"):
    df = read_market_csv(data_dir, market, start, end)
    return df.pivot(index="datetime", columns="asset", values="close").ffill().bfill()


def price_movements(close, start, end, window=50, bar_days=5, step=None):
    """r_{t,m} = p^C at the end of the next bar / p^C before the decision, for decisions in [start, end]."""
    step = bar_days if step is None else step
    r = close.iloc[window * bar_days - 1:].pct_change(bar_days).shift(-bar_days + 1).dropna() + 1
    return r.loc[start:end].iloc[::step]


def benchmark_returns(close, method, commission=0.0006, bar_days=5):
    """Test returns of a classical strategy from ``universal-portfolios`` on weekly closes, 2012-2023."""
    from universal import algos

    algo = {"CRP": algos.CRP, "BAH": algos.BAH, "EG": algos.EG, "OLMAR": algos.OLMAR, "RMR": algos.RMR}[method]()
    prices = close.loc[PARTS["A"][1]:PARTS["C"][2]].iloc[::bar_days]
    result = algo.run(prices)
    result.fee = commission
    return result.r


# ----------------------------------------------------------------------------- tables

def comparison_table(res, close):
    """Tables 2 and 3: benchmarks, classical and DRL methods, Transformer EIIE and + RAS."""
    rows, curves = [], {}
    for method in BENCHMARKS:
        R = benchmark_returns(close, method)
        rows.append({"Type": "Benchmark" if method in ("CRP", "BAH") else "Classical", "Method": method,
                     **{k: (v, None) for k, v in performance_over_seeds([R.values]).items()}, "Train Reward": None})
        curves[method] = np.log(R).cumsum()
    for run, name in list(BASELINES.items()) + [(transformer_run(False, 0.0), "Transformer EIIE")]:
        test = [res.test_returns(run, s) for s in SEEDS]
        rows.append({"Type": "DRL" if run in BASELINES else "Ours", "Method": name,
                     **performance_over_seeds([R.values for R in test]),
                     "Train Reward": summarize(train_reward_runs(res, run))})
        curves[name] = pd.concat([np.log(R).cumsum().rename(s) for s, R in zip(SEEDS, test)], axis=1).mean(axis=1)
    test = [pd.concat(res.ras_returns(s)) for s in SEEDS]
    reward = [np.mean([metrics.train_reward(R) for R in res.ras_returns(s, "train")]) for s in SEEDS]
    rows.append({"Type": "Ours", "Method": "+ RAS", **performance_over_seeds([R.values for R in test]),
                 "Train Reward": summarize(reward)})
    curves["Transformer EIIE + RAS"] = pd.concat([np.log(R).cumsum().rename(s) for s, R in zip(SEEDS, test)], axis=1).mean(axis=1)
    return rows, curves


def ablation_table(res, close):
    """Ablation tables: MAL on/off x gamma, with Train Reward and Rank IC."""
    rows, curves = [], {}
    for mal in (False, True):
        for gamma in GAMMAS:
            run = transformer_run(mal, gamma)
            test = [res.test_returns(run, s) for s in SEEDS]
            rank_ic = []
            for s in SEEDS:
                per_part = []
                for p in PARTS:
                    z = res.scores(run, p, s)
                    r = price_movements(close, PARTS[p][1], PARTS[p][2])[z.columns].iloc[:len(z)]
                    per_part.append(metrics.rank_ic_ir(z.values, r.values))
                rank_ic.append(np.mean(per_part))
            rows.append({"MAL": mal, "gamma": gamma, **performance_over_seeds([R.values for R in test]),
                         "Train Reward": summarize(train_reward_runs(res, run)), "Rank IC": summarize(rank_ic, ddof=1)})
            label = f"{'with' if mal else 'w/o'} MAL, γ = {gamma}"
            curves[label] = pd.concat([np.log(R).cumsum().rename(s) for s, R in zip(SEEDS, test)], axis=1).mean(axis=1)
    return rows, curves


def mal_table(res, close):
    """Standalone evaluation of MAL: learned score z (Transformer EIIE) vs predicted score z^ (MAL)."""
    rows = []
    run = transformer_run(False, 0.0)
    stats = {"z": {k: [] for k in ("Rank IC", "Pearson", "Mean Bias", "Cohen's d")},
             "z_hat": {k: [] for k in ("Rank IC", "Pearson", "Mean Bias", "Cohen's d")}}
    for s in SEEDS:
        # learned score z: Rank IC is averaged over the parts, the other statistics pool the parts
        zs, rs, ics = [], [], []
        for p in PARTS:
            z = res.scores(run, p, s)
            r = price_movements(close, PARTS[p][1], PARTS[p][2])[z.columns].iloc[:len(z)]
            ics.append(metrics.rank_ic_ir(z.values, r.values))
            zs.append(z.values.ravel())
            rs.append(r.values.ravel())
        z, r = np.concatenate(zs), np.concatenate(rs)
        for key, value in (("Rank IC", np.mean(ics)), ("Pearson", metrics.pearson(z, r)),
                           ("Mean Bias", metrics.mean_bias(z, r)), ("Cohen's d", metrics.cohens_d(z, r))):
            stats["z"][key].append(value)
        # predicted score z^: daily predictions, all parts pooled
        zs, rs, ics = [], [], []
        for p in PARTS:
            z_hat = res.predictions(p, s).fillna(1.0)
            r = price_movements(close, PARTS[p][1], PARTS[p][2], step=1)[z_hat.columns].iloc[:len(z_hat)]
            ics.append(metrics.rank_ic_series(z_hat.values, r.values))
            zs.append(z_hat.values.ravel())
            rs.append(r.values.ravel())
        z, r, ic = np.concatenate(zs), np.concatenate(rs), np.concatenate(ics)
        for key, value in (("Rank IC", np.nanmean(ic) / np.nanstd(ic)), ("Pearson", metrics.pearson(z, r)),
                           ("Mean Bias", metrics.mean_bias(z, r)), ("Cohen's d", metrics.cohens_d(z, r))):
            stats["z_hat"][key].append(value)
    for name, values in stats.items():
        rows.append({"Score": name, **{k: summarize(v, ddof=1) for k, v in values.items()}})
    return rows


def mean_pairwise_correlation(close, start, end):
    corr = close.pct_change().dropna().loc[start:end].corr().values
    n = corr.shape[0]
    return (corr.sum() - n) / (n * (n - 1))


def regime_table(res, close):
    """Performance with and without RAS per test segment and in the COVID-19 crash."""
    rows = []
    best = res.select_ras()
    base = transformer_run(False, 0.0)
    for part, (_, test_start, end) in PARTS.items():
        row = {"Period": f"{test_start[:4]}-{int(end[:4]) - 1}", "Corr.": mean_pairwise_correlation(close, test_start, end)}
        for label, run in (("no", base), ("yes", transformer_run(True, best[part]))):
            rows.append({**row, "RAS": label, **performance_over_seeds([res.returns(run, part, s, "test").values for s in SEEDS])})
    row = {"Period": "2020 H1 (COVID)", "Corr.": mean_pairwise_correlation(close, *COVID)}
    for label, returns in (("no", [res.test_returns(base, s) for s in SEEDS]),
                           ("yes", [pd.concat(res.ras_returns(s)) for s in SEEDS])):
        rows.append({**row, "RAS": label, **performance_over_seeds([R.loc[COVID[0]:COVID[1]].values for R in returns], ddof=0)})
    return rows


# ----------------------------------------------------------------------------- formatting

def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, tuple):
        mean, std = value
        if std is None:
            return fmt(mean)
        return f"{fmt(mean)} ± {fmt(std)}"
    if isinstance(value, (bool, np.bool_)):
        return "✓" if value else "×"
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}" if abs(value) > 10 else f"{value:.3f}"
    return str(value)


def to_markdown(rows):
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in columns) + " |" for row in rows]
    return "\n".join(lines)
