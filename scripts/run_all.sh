#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Full Benchmark Pipeline for codec-finetuning
#
# Runs the complete pipeline for a single codec: train -> reconstruct ->
# evaluate -> log results.
#
# Usage:
#   bash scripts/run_all.sh mimi turkish_sample
#   bash scripts/run_all.sh dualcodec hindi
#   bash scripts/run_all.sh kanade turkish_sample --sweep  # Run sweep first
#
# Prerequisites:
#   - Environment set up via scripts/setup.sh
#   - Data prepared via scripts/download_data.sh
#   - WandB login for logging (wandb login)
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

CODEC="${1:?Usage: bash scripts/run_all.sh <codec> <dataset> [--sweep]}"
DATASET="${2:?Usage: bash scripts/run_all.sh <codec> <dataset> [--sweep]}"
SWEEP="${3:-}"

EXPERIMENT="${CODEC}_${DATASET}"
CONFIG="configs/experiments/${EXPERIMENT}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "Full Pipeline: $EXPERIMENT"
echo "Config: $CONFIG"
echo "═══════════════════════════════════════════════════════════"

# Step 1: Optional sweep
if [[ "$SWEEP" == "--sweep" ]]; then
    echo "[1/5] Running hyperparameter sweep..."
    bash scripts/run_sweep.sh "$CODEC"
    echo "Sweep complete. Using best config from sweep."
    # TODO: Auto-detect best config from sweep
else
    echo "[1/5] Skipping sweep (use --sweep to enable)"
fi

# Step 2: Train
echo "[2/5] Training..."
if [[ "$CODEC" == "mimi" ]]; then
    uv run python train/train_mimi.py --config "$CONFIG"
elif [[ "$CODEC" == "dualcodec" ]]; then
    bash train/train_dualcodec.sh --config "$CONFIG"
elif [[ "$CODEC" == "kanade" ]]; then
    bash train/train_kanade.sh --config "$CONFIG"
fi

# Step 3: Reconstruct
echo "[3/5] Reconstructing test audio..."
uv run python eval/reconstruct.py --config "$CONFIG" --split test --use-ema

# Step 4: Evaluate
echo "[4/5] Evaluating..."
uv run python eval/bootstrap_eval.py --experiment "$EXPERIMENT"
uv run python eval/measure_ttfat.py --config "$CONFIG"
uv run python eval/measure_ssnr.py --experiment "$EXPERIMENT"

# Step 5: Log to WandB
echo "[5/5] Logging results to WandB..."
uv run python eval/log_to_wandb.py --experiment "$EXPERIMENT"

echo "═══════════════════════════════════════════════════════════"
echo "Pipeline complete: $EXPERIMENT"
echo "═══════════════════════════════════════════════════════════"
