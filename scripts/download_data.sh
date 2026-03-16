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
    echo "Turkish Sample (3.1h) -- MagicHub ASR-STurkDuSC"
    echo ""
    echo "This dataset requires manual download from MagicHub."
    echo "Please follow these steps:"
    echo "  1. Visit: https://magichub.com/datasets/turkish-conversational-speech-corpus/"
    echo "  2. Register for a free account"
    echo "  3. Download the dataset ZIP"
    echo "  4. Extract to: data/turkish_sample_raw/"
    echo "  5. Then run: uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml"
    echo ""
    
    mkdir -p data/turkish_sample_raw
    
    if [ -d "data/turkish_sample_raw" ] && [ "$(ls -A data/turkish_sample_raw 2>/dev/null)" ]; then
        echo "Raw data found. Running preparation..."
        uv run python prepare_data.py --config configs/datasets/turkish_sample.yaml
    else
        echo "No raw data found in data/turkish_sample_raw/. Please download first."
    fi
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
