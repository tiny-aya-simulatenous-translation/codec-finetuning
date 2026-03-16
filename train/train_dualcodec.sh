#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# DualCodec Fine-tuning Launcher
#
# Wraps DualCodec's native training pipeline (accelerate + Hydra) with our
# config system. Reads our YAML config, translates to DualCodec's expected
# format, and launches training.
#
# Usage:
#   bash train/train_dualcodec.sh --config configs/experiments/dualcodec_turkish_sample.yaml
#
# Prerequisites:
#   - DualCodec installed: pip install dualcodec
#   - Data prepared: uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# Parse arguments
CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Usage: bash train/train_dualcodec.sh --config <path>"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "DualCodec Fine-tuning"
echo "Config: $CONFIG"
echo "═══════════════════════════════════════════════════════════"

# Extract key fields from config using Python
read_config() {
    uv run python -c "
import yaml, sys
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
# Navigate nested keys
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

mkdir -p "$OUTPUT_DIR"

echo "Pretrained: $PRETRAINED"
echo "LR: $LR"
echo "Steps: $MAX_STEPS"
echo "Batch: $BATCH_SIZE"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"

# Launch DualCodec training via accelerate
# DualCodec expects its own Hydra config format. We pass overrides via CLI.
uv run accelerate launch \
    --mixed_precision bf16 \
    -m dualcodec.train \
    --config-name dualcodec_ft_12hzv1 \
    training.output_dir="$OUTPUT_DIR" \
    training.learning_rate="$LR" \
    training.max_steps="$MAX_STEPS" \
    training.per_device_train_batch_size="$BATCH_SIZE" \
    training.seed="$SEED" \
    data.train_data_dir="$DATA_DIR/train" \
    data.val_data_dir="$DATA_DIR/val" \
    model.pretrained_model_name_or_path="$PRETRAINED"

echo "═══════════════════════════════════════════════════════════"
echo "DualCodec training complete. Output: $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════"
