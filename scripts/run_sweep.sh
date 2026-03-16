#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# WandB Hyperparameter Sweep for codec-finetuning
#
# Launches a WandB Bayesian hyperparameter sweep for a given codec.
#
# Usage:
#   bash scripts/run_sweep.sh mimi
#   bash scripts/run_sweep.sh dualcodec
#   bash scripts/run_sweep.sh kanade
#   WANDB_AGENTS=2 bash scripts/run_sweep.sh mimi  # Run 2 agents in parallel
#
# Prerequisites:
#   - Environment set up via scripts/setup.sh
#   - WandB login (wandb login)
#   - Sweep config at configs/sweeps/<codec>_sweep.yaml
#
# License: MIT
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

CODEC="${1:?Usage: bash scripts/run_sweep.sh <codec>}"
NUM_AGENTS="${WANDB_AGENTS:-1}"

SWEEP_CONFIG="configs/sweeps/${CODEC}_sweep.yaml"

if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "ERROR: Sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo "Hyperparameter Sweep: $CODEC"
echo "Config: $SWEEP_CONFIG"
echo "Agents: $NUM_AGENTS"
echo "═══════════════════════════════════════════════════════════"

# Create sweep
echo "Creating WandB sweep..."
SWEEP_ID=$(uv run wandb sweep "$SWEEP_CONFIG" 2>&1 | grep -oP 'wandb agent \K\S+' || true)

if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Failed to create sweep. Check WandB authentication."
    echo "Run: wandb login"
    exit 1
fi

echo "Sweep ID: $SWEEP_ID"

# Launch agents
for i in $(seq 1 "$NUM_AGENTS"); do
    if [ "$i" -lt "$NUM_AGENTS" ]; then
        echo "Launching agent $i (background)..."
        uv run wandb agent "$SWEEP_ID" &
    else
        echo "Launching agent $i (foreground)..."
        uv run wandb agent "$SWEEP_ID"
    fi
done

echo "═══════════════════════════════════════════════════════════"
echo "Sweep complete: $CODEC"
echo "Analyze results: uv run python scripts/analyze_sweep.py --sweep-id $SWEEP_ID"
echo "═══════════════════════════════════════════════════════════"
