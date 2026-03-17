#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Dataset Download for codec-finetuning
#
# Downloads and prepares datasets for the codec-finetuning benchmark.
#
# Usage:
#   bash scripts/download_data.sh                    # Download Turkish sample (default)
#   bash scripts/download_data.sh turkish_sample      # Same as above
#   bash scripts/download_data.sh hindi               # Requires HF login
#   bash scripts/download_data.sh all                 # Download all available
#
# Prerequisites:
#   - Environment set up via scripts/setup.sh
#   - HuggingFace login for gated datasets (huggingface-cli login)
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

DATASET="${1:-turkish_sample}"

echo "═══════════════════════════════════════════════════════════"
echo "Dataset Download: $DATASET"
echo "═══════════════════════════════════════════════════════════"

download_turkish_sample() {
    echo "Turkish Sample (10h) -- MediaSpeech (OpenSLR 108, CC BY 4.0)"
    echo ""

    RAW_DIR="data/turkish_sample_raw"
    mkdir -p "$RAW_DIR"

    if [ -d "$RAW_DIR/TR" ] && [ "$(ls -A "$RAW_DIR/TR" 2>/dev/null)" ]; then
        echo "Raw data already present in $RAW_DIR/TR. Skipping download."
    else
        echo "Downloading TR.tgz from OpenSLR (~618 MB)..."
        rm -f "$RAW_DIR/TR.tgz"
        wget -O "$RAW_DIR/TR.tgz" "https://openslr.trmal.net/resources/108/TR.tgz"
        echo "Extracting..."
        tar xzf "$RAW_DIR/TR.tgz" -C "$RAW_DIR"
        rm -f "$RAW_DIR/TR.tgz"
        echo "Download complete."
    fi

    echo "Running data preparation..."
    uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml
}

download_hindi() {
    echo "Hindi (~90h) -- HuggingFace private repo"
    echo ""

    # Check HF auth
    if ! uv run python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
        echo "ERROR: Not logged in to HuggingFace."
        echo "Run: huggingface-cli login"
        exit 1
    fi

    echo "Authenticated. Downloading and preparing Hindi dataset..."
    uv run python prepare_data.py --config configs/datasets/hindi.yaml
}

case "$DATASET" in
    turkish_sample)
        download_turkish_sample
        ;;
    hindi)
        download_hindi
        ;;
    all)
        download_turkish_sample
        download_hindi
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available: turkish_sample, hindi, all"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════════════════════════"
echo "Done."
echo "═══════════════════════════════════════════════════════════"
