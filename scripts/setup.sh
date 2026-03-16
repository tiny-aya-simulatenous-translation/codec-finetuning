#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Environment Setup for codec-finetuning
#
# Installs uv, creates venv, installs all dependencies (train + eval extras),
# installs FlashAttention, downloads pretrained models, and validates the setup.
#
# Usage:
#   bash scripts/setup.sh
#
# Prerequisites:
#   - NVIDIA GPU with CUDA 12.6+ drivers
#   - Python 3.12+ available on PATH
#   - Internet access for downloads
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════════════════"
echo "codec-finetuning: Environment Setup"
echo "═══════════════════════════════════════════════════════════"

# 1. Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "[1/6] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/6] uv already installed: $(uv --version)"
fi

# 2. Create venv and install dependencies
echo "[2/6] Installing dependencies (train + eval)..."
uv sync --extra train --extra eval

# 3. Install FlashAttention (prebuilt wheel for torch 2.9.1 + cu128)
echo "[3/6] Installing FlashAttention 2.8.3..."
uv pip install flash-attn==2.8.3 --no-build-isolation

# 4. Download pretrained models
echo "[4/6] Downloading pretrained models..."
uv run python -c "
from transformers import MimiModel
print('Downloading Mimi...')
MimiModel.from_pretrained('kyutai/mimi')
print('Mimi downloaded.')
"

# DualCodec model download
uv run python -c "
print('Downloading DualCodec...')
try:
    import dualcodec
    dualcodec.load_model('12hz_v1')
    print('DualCodec downloaded.')
except Exception as e:
    print(f'DualCodec download skipped: {e}')
"

# Kanade model download
uv run python -c "
print('Downloading Kanade...')
try:
    from huggingface_hub import snapshot_download
    snapshot_download('frothywater/kanade-25hz-clean')
    print('Kanade downloaded.')
except Exception as e:
    print(f'Kanade download skipped: {e}')
"

# 5. Download Whisper model for WER evaluation
echo "[5/6] Downloading Whisper large-v3 for evaluation..."
uv run python -c "
import whisper
print('Downloading Whisper large-v3...')
whisper.load_model('large-v3')
print('Whisper downloaded.')
"

# 6. Validate setup
echo "[6/6] Validating setup..."
uv run python -c "
import torch
import torchaudio
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')

# Check FlashAttention
try:
    import flash_attn
    print(f'FlashAttention: {flash_attn.__version__}')
except ImportError:
    print('FlashAttention: NOT INSTALLED (will use PyTorch SDPA fallback)')

# Check bf16 support
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    print('BF16: supported')
else:
    print('BF16: NOT supported (will use fp32)')

# Check codecs
for name, imp in [('moshi', 'moshi'), ('dualcodec', 'dualcodec'), ('kanade_tokenizer', 'kanade_tokenizer')]:
    try:
        __import__(imp)
        print(f'{name}: installed')
    except ImportError:
        print(f'{name}: NOT installed')

print()
print('Setup complete!')
"

echo "═══════════════════════════════════════════════════════════"
echo "Setup complete. Next steps:"
echo "  1. Prepare data:    bash scripts/download_data.sh"
echo "  2. Train:           uv run python train/train_mimi.py --config configs/experiments/mimi_turkish_sample.yaml"
echo "═══════════════════════════════════════════════════════════"
