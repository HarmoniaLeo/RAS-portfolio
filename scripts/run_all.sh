#!/usr/bin/env bash
# Full experiment grid of the paper, followed by the tables and figures.
#
#   bash scripts/run_all.sh                                  # everything (as in the paper)
#   GPUS=2 bash scripts/run_all.sh                           # two GPUs per run via torchrun
#   MARKETS=CSI PARTS=A SEEDS=0 EPOCHS=5 PRED_EPOCHS=3 bash scripts/run_all.sh   # quick pipeline check
#
# Runs whose metrics.json already exists are skipped, so the script can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs}
GPUS=${GPUS:-1}
MARKETS=${MARKETS:-"SP CSI"}
PARTS=${PARTS:-"A B C"}
SEEDS=${SEEDS:-"0 64 128"}
GAMMAS=${GAMMAS:-"0.0 0.2 0.5 0.8"}
BASELINES=${BASELINES:-"cnn_eiie brnn_eiie rat hgam lsre_caan"}
EPOCHS=${EPOCHS:-1000}
PRED_EPOCHS=${PRED_EPOCHS:-500}

if [ "$GPUS" -gt 1 ]; then LAUNCH="torchrun --nproc_per_node $GPUS"; else LAUNCH="python"; fi

run() {  # run <result dir> <command...>
  local dir=$1; shift
  if [ -f "$dir/metrics.json" ]; then echo "skip $dir"; return; fi
  echo ">>> $*"
  "$@"
}

for market in $MARKETS; do
  for part in $PARTS; do
    for seed in $SEEDS; do
      common="market=$market part=$part seed=$seed output_dir=$OUT"
      run_id="part${part}_seed${seed}"

      # Stage 1: MAL predictor
      run "$OUT/$market/predictor/$run_id" $LAUNCH scripts/train_predictor.py --set $common epochs=$PRED_EPOCHS
      predictor="$OUT/$market/predictor/$run_id/checkpoint.pt"

      # DRL baselines
      for model in $BASELINES; do
        run "$OUT/$market/$model/$run_id" $LAUNCH scripts/train_policy.py --config configs/$model.yaml --set $common epochs=$EPOCHS
      done

      # Transformer EIIE with ARM (gamma) and MAL on/off
      for gamma in $GAMMAS; do
        run "$OUT/$market/transformer_eiie_gamma$gamma/$run_id" $LAUNCH scripts/train_policy.py \
          --config configs/transformer_eiie.yaml --set $common epochs=$EPOCHS gamma=$gamma
        run "$OUT/$market/transformer_eiie_mal_gamma$gamma/$run_id" $LAUNCH scripts/train_policy.py \
          --config configs/transformer_eiie.yaml --set $common epochs=$EPOCHS gamma=$gamma mal=true predictor=$predictor
      done
    done
  done
done

if [ "$MARKETS" = "SP CSI" ] && [ "$PARTS" = "A B C" ] && [ "$SEEDS" = "0 64 128" ]; then
  python scripts/reproduce_paper.py --results "$OUT" --out "$OUT/paper"
else
  echo "Partial grid finished; visualise it with: python scripts/plot_run.py --runs \"$OUT/*/*/part*_seed*\" --out $OUT/plots/runs.pdf"
fi
