#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Download and prepare the Turkish Full dataset for codec-finetuning
#
# Downloads the tiny-aya-translate/tr-subset-v0.1 HuggingFace dataset
# (~251k utterances, ~62.4 GB) and runs prepare_data.py to resample
# audio to 24 kHz, create train/val/test splits, and write manifests
# into data/turkish_full/.
#
# Usage:
#   bash scripts/download_turkish_full.sh
#
# Prerequisites:
#   - Environment set up via scripts/setup.sh (uv sync --extra all)
#   - ~130 GB free disk space (62.4 GB download + processed output)
#   - Internet access for HuggingFace download
#
# Environment variables:
#   HF_TOKEN          Optional HuggingFace token (if repo requires auth).
#   HF_DATASETS_CACHE Override the HuggingFace datasets cache directory.
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "Turkish Full Dataset Download & Preparation"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  HF Repo   : tiny-aya-translate/tr-subset-v0.1"
echo "  Split     : train (251,118 utterances)"
echo "  Size      : ~62.4 GB"
echo "  Output    : data/turkish_full/"
echo ""

# Check if data is already prepared (manifests exist).
if [ -f "data/turkish_full/train/manifest.json" ] && \
   [ -f "data/turkish_full/val/manifest.json" ] && \
   [ -f "data/turkish_full/test/manifest.json" ]; then
    echo "Data already prepared in data/turkish_full/. Skipping."
    echo "To re-prepare, remove data/turkish_full/ and re-run."
    exit 0
fi

# Verify that the datasets library is available.
echo "[1/2] Checking dependencies..."
if ! uv run python -c "from datasets import load_dataset; print('datasets OK')" 2>/dev/null; then
    echo "ERROR: 'datasets' library not found."
    echo "Install with: uv sync --extra train"
    exit 1
fi

# Run the data preparation pipeline.
echo "[2/2] Downloading and preparing dataset..."
echo "  This will download ~62.4 GB from HuggingFace and resample"
echo "  all audio to 24 kHz. This may take a while."
echo ""

uv run python prepare_data.py --config configs/datasets/turkish_full.yaml

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Done. Data prepared in data/turkish_full/"
echo "═══════════════════════════════════════════════════════════"
