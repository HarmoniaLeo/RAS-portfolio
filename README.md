# Robust Asset Scoring (RAS)

Code, trained models and experiment logs for the paper

> **Deep Reinforcement Learning for Portfolio Management with Robust Asset Scoring**
> Bingde Liu, Ziwei Xu, Ryutaro Ichise — Institute of Science Tokyo

DRL portfolio policies of the EIIE family score all assets jointly, through a softmax over the scores. The score an asset receives therefore depends on which other assets it is compared with (the *non-robust scoring problem*). RAS fixes this with two components:

- **ARM (Asset Random Masking)**: during training, each asset is masked with probability γ in every episode. Each price pattern is then compared against many random subsets of the other assets.
- **MAL (Model-Augmented Learning)**: a separately trained, frozen predictive network estimates each asset's price movement from its own price pattern. Its embedding is fused into the policy's asset embedding.

The repository covers the whole pipeline:

**data collection → MAL pre-training → policy training → testing → tables and figures**

It also contains the 252 trained runs reported in the paper, so every table and figure can be reproduced without any training.

## Contents

```
ras/                    library
  config.py             configuration, data partitions (Table 1), paper notation
  data.py               market data, MDP samples, ARM masking
  networks/             feature extractors f: Transformer (ours), CNN/bRNN EIIE, RAT, HGAM, LSRE-CAAN
  predictor.py          MAL predictive network f^pred, g^pred and loss L_pred
  policy.py             EIIE policy: f -> MAL fusion -> score head g -> cash bias -> softmax
  portfolio.py          immediate return R_t, holding transition, portfolio vector memory
  trainer.py            Algorithm 1 (stage 1 and 2), validation-based model selection, testing
  metrics.py            AR, ASR, ACR, Train Reward, Rank IC, Pearson, Mean Bias, Cohen's d
  analysis.py           aggregation of runs into the paper's tables and figures
scripts/
  download_data.py      download S&P500 / CSI300 constituents from Yahoo Finance
  train_predictor.py    stage 1: MAL pre-training
  train_policy.py       stage 2: policy learning (+ testing)
  evaluate.py           test trained checkpoints
  run_all.sh            the full experiment grid of the paper
  reproduce_paper.py    tables and figures of Sec. 5 from a results directory
  plot_run.py           plots for individual runs
  toy_example.py        the toy example of Sec. 3.2 and 4.2
configs/                default hyperparameters of every method, ticker lists
results/paper/          the paper's runs: returns, metrics, scores, checkpoints
tools/export_legacy.py  how results/paper was exported from the original experiment code
```

## Notation

Code names follow the paper.

| Paper | Code |
|---|---|
| price pattern 𝒫_{t,m}, window size l | `patterns` `[B, 4, l, M]`, `window` (50 weekly bars) |
| price movement r_{t,m} | `movement` |
| portfolio holding w_t, allocation a_t | `holding`, `allocation` (cash first) |
| immediate return R_t, reward log R_t | `ras.portfolio.step`, `-log(R).mean()` is the loss |
| transaction cost rate c | `transaction_cost` (0.0006) |
| feature extractor f, embedding h_{t,m} | `EIIEPolicy.f`, `EIIEPolicy.embed` |
| score head g, score z_{t,m}, cash bias z_{0,0} | `EIIEPolicy.g`, `scores`, `cash_bias` |
| masking ratio γ, constant Y | `gamma`, `mask_constant` (1e8) |
| predictive network f^pred, g^pred, predicted score ẑ_{t,m} | `PricePredictor.f_pred`, `.g_pred`, `z_hat` |
| fusing MLP, augmented embedding h̃_{t,m} | `EIIEPolicy.fusion` |
| prediction loss L_pred | `ras.predictor.prediction_loss` |
| model dimension d = d_pred, decoder layers N | `d` (32), `num_layers` (2) |
| learning rate α | `lr` |
| Parts A/B/C (Table 1) | `part`, see `ras.config.PARTS` |

## Installation

```bash
conda create -n ras python=3.11 -y && conda activate ras
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121   # or any torch >= 2.0 for your platform
pip install -r requirements.txt
```

## 1. Data

```bash
python scripts/download_data.py --markets SP CSI --out data     # ~700 tickers, takes a while
```

This writes `data/SP.csv` (475 stocks) and `data/CSI.csv` (222 stocks): daily, split/dividend-adjusted OHLC prices from 2004-01-01 to 2023-12-31. The ticker lists in `configs/tickers/` are exactly the constituents used in the paper. Yahoo Finance revises its adjusted prices over time, so a new download can differ slightly from the paper's snapshot. Daily returns are unchanged, but the overall adjustment level shifts. The snapshot used in the paper is available from the authors on request.

Every observation is a weekly bar: 5 trading days are aggregated into one OHLC bar (`bar_days: 5`), and one decision step is one week.

## 2. Reproduce the tables and figures from the released runs (no training)

```bash
python scripts/reproduce_paper.py --results results/paper --out outputs/paper
python scripts/toy_example.py --out outputs/toy
```

`outputs/paper/tables.md` contains:

- the performance comparison on S&P500 and CSI300 (benchmarks, classical methods, DRL baselines, Transformer EIIE, + RAS)
- the standalone evaluation of MAL
- the performance across market regimes
- the ablation of MAL × γ

`compare.pdf` and `ablation.pdf` are the accumulated-log-return figures. Headline numbers (test set 2012–2023, mean ± std over 3 seeds):

| | S&P500 AR | S&P500 ASR | S&P500 ACR | CSI300 AR | CSI300 ASR | CSI300 ACR |
|---|---|---|---|---|---|---|
| Transformer EIIE | 0.343 ± 0.165 | 0.920 ± 0.325 | 0.914 ± 0.521 | 0.288 ± 0.228 | 0.709 ± 0.317 | 0.575 ± 0.554 |
| + RAS | 0.537 ± 0.144 | 1.065 ± 0.072 | 1.037 ± 0.124 | 0.475 ± 0.078 | 0.971 ± 0.095 | 0.940 ± 0.225 |

`+ RAS` always uses MAL. γ is chosen per part on the validation set (S&P500: A 0.8, B 0.0, C 0.5; CSI300: A 0.2, B 0.8, C 0.5).

`results/paper/<market>/<run>/part<P>_seed<s>/` holds, for every run:

- `returns_{train,val,test}.csv` and `metrics.json`
- `scores_test.npz` (Transformer EIIE: the scores z)
- `predictions_test.npz` (MAL predictor: ẑ)
- `checkpoint.pt`
- `config.json`, whose `legacy_id` is the run's identifier in the original experiment code

## 3. Test the released checkpoints

```bash
python scripts/evaluate.py --runs "results/paper/*/*/part*_seed*" --out outputs/reeval
```

This re-runs every checkpoint on the train/val/test splits and reports the largest deviation from the released returns:

- Transformer EIIE (all ARM/MAL variants), CNN EIIE and RAT: up to about 1e-6 per step.
- The MAL predictors: up to about 1e-7 per prediction.
- The LSTM-based baselines (bRNN EIIE, HGAM) and LSRE-CAAN: small numerical differences across GPUs and library versions. Metrics change by at most 0.02.

## 4. Train

Stage 1 trains the MAL predictor. Stage 2 trains a policy, then tests the checkpoint with the best validation ASR. Results go to `outputs/<market>/<run>/part<P>_seed<s>/`, in the same layout as `results/paper`.

```bash
# Stage 1 (Algorithm 1): MAL pre-training
python scripts/train_predictor.py --set market=SP part=C seed=0

# Stage 2: Transformer EIIE + RAS (ARM gamma = 0.5, MAL with the frozen predictor of stage 1)
python scripts/train_policy.py --config configs/transformer_eiie.yaml \
    --set market=SP part=C seed=0 gamma=0.5 mal=true predictor=outputs/SP/predictor/partC_seed0/checkpoint.pt

# Baselines
python scripts/train_policy.py --config configs/rat.yaml --set market=SP part=C seed=0

# Multi-GPU (the paper used two GPUs)
torchrun --nproc_per_node 2 scripts/train_policy.py --config configs/transformer_eiie.yaml --set market=CSI part=A
```

Any configuration key can be overridden with `--set key=value` (see `ras/config.py`).

The whole grid of the paper covers 2 markets × 3 parts × 3 seeds: 18 predictors, 90 baseline runs and 144 Transformer EIIE runs. With the default 1000 epochs, a policy run takes hours on one GPU. `run_all.sh` runs everything and finishes with the tables:

```bash
bash scripts/run_all.sh                     # everything, then tables/figures in outputs/paper
GPUS=2 bash scripts/run_all.sh              # with torchrun
# quick check of the whole pipeline (a few minutes):
MARKETS=CSI PARTS=A SEEDS=0 EPOCHS=2 PRED_EPOCHS=2 bash scripts/run_all.sh
```

Retraining reproduces the results statistically rather than bit for bit, because multi-GPU training and cuDNN kernels are not deterministic.

## 5. Visualise

```bash
python scripts/plot_run.py --runs "outputs/CSI/*/partA_seed0" --out outputs/plots/csi_partA.pdf
python scripts/reproduce_paper.py --results outputs --out outputs/paper     # after the full grid
```

## Implementation notes

These notes describe exactly what the code (and the released runs) do, where the paper's description is abbreviated:

- **Transaction cost.** R_t = (a_tᵀ r_t)(1 − c‖a_t − w_t‖₁), i.e. the cost is charged proportionally to the gross return.
- **Masking.** ARM subtracts Y = 1e8 from a masked score (`z − Y`), and also zeroes the masked asset's pattern and price movement.
- **Attention.** The Transformer layers of f and f^pred apply no causal mask (`causal: false`); `causal: true` adds one.
- **Training details.**
  - Every trading day of the train set can be sampled as a decision step. Holdings are read from and written to a portfolio vector memory.
  - With probability `restart_prob = 0.05`, w_t is reset to cash (episode restart).
  - Adam with lr 1e-4, betas (0.9, 0.98), eps 1e-9, weight decay 5e-8. Batch size 64 per GPU (two GPUs in the paper), 1000 epochs, validation every 10 epochs.
- **MAL predictor.** Trained on single-asset windows with batch 4096 for 500 epochs. The per-sample squared error is clipped at 0.1, and the checkpoint is selected by the validation Pearson correlation.
- **Reported statistics.**
  - The "Rank IC" columns are the information ratio of the per-step Rank IC, mean(Rank IC_t) / std(Rank IC_t). `ras.metrics.rank_ic` gives the plain mean.
  - Cohen's d is the mean bias divided by the standard deviation of r.
  - The ASR of a seed uses the sample standard deviation of its returns (the COVID-19 row uses the population one); `ras.analysis` documents all conventions.
- **Classical benchmarks.** These come from `universal-portfolios` on weekly closes with fee 0.0006. With the current library version, OLMAR and RMR differ somewhat from the values printed in the paper.

## Citation

```bibtex
@article{liu2025ras,
  title   = {Deep Reinforcement Learning for Portfolio Management with Robust Asset Scoring},
  author  = {Liu, Bingde and Xu, Ziwei and Ichise, Ryutaro},
  year    = {2025}
}
```
