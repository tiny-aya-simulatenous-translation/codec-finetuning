#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Kanade Fine-tuning Launcher
#
# Wraps Kanade's native training pipeline (Lightning CLI) with our config
# system. Reads our YAML config, translates to Kanade's expected format,
# and launches training.
#
# Usage:
#   bash train/train_kanade.sh --config configs/experiments/kanade_turkish_sample.yaml
#
# Prerequisites:
#   - Kanade installed: pip install kanade-tokenizer (from git)
#   - Data prepared: uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: bash train/train_kanade.sh --config <path>"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "Kanade Fine-tuning"
echo "Config: $CONFIG"
echo "═══════════════════════════════════════════════════════════"

read_config() {
    uv run python -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
keys = '$1'.split('.')
val = c
for k in keys:
    val = val[k]
print(val)
"
}

OUTPUT_DIR=$(read_config "output_dir")
PRETRAINED=$(read_config "codec.pretrained")
LR=$(read_config "optimizer.lr")
MAX_STEPS=$(read_config "training.max_steps")
BATCH_SIZE=$(read_config "codec.training.micro_batch_size")
SEED=$(read_config "training.seed")
DATA_DIR=$(read_config "dataset.local_dir")
VOCODER=$(read_config "codec.training.vocoder")

mkdir -p "$OUTPUT_DIR"

echo "Pretrained: $PRETRAINED"
echo "LR: $LR"
echo "Steps: $MAX_STEPS"
echo "Batch: $BATCH_SIZE"
echo "Data: $DATA_DIR"
echo "Vocoder: $VOCODER"
echo "Output: $OUTPUT_DIR"

# Kanade uses Lightning CLI for training.
# Post-training stage (with GAN) is the default for fine-tuning.
uv run python -m kanade.train \
    --trainer.default_root_dir "$OUTPUT_DIR" \
    --trainer.max_steps "$MAX_STEPS" \
    --trainer.precision "bf16-mixed" \
    --trainer.gradient_clip_val 1.0 \
    --seed_everything "$SEED" \
    --data.batch_size "$BATCH_SIZE" \
    --data.train_dir "$DATA_DIR/train" \
    --data.val_dir "$DATA_DIR/val" \
    --model.pretrained "$PRETRAINED" \
    --model.vocoder "$VOCODER" \
    --optimizer.lr "$LR"

echo "═══════════════════════════════════════════════════════════"
echo "Kanade training complete. Output: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════"
