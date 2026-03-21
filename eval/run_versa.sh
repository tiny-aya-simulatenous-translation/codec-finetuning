#!/usr/bin/env bash
# =========================================================================
# Run VERSA evaluation toolkit (optional, heavy dependency).
#
# VERSA provides 90+ metrics.  We use it for comprehensive final
# evaluation of reconstructed audio against original utterances.
#
# Invocation modes
# ----------------
# 1. **Standalone** (full manifest):
#      bash eval/run_versa.sh <experiment_name>
#
#    Reads the reconstruction manifest, generates Kaldi-style .scp
#    files, and runs the VERSA scorer on all utterances.
#
# 2. **Sharded** (called by _run_versa_shard in run_all.py):
#    When the environment variables VERSA_SHARD_GT, VERSA_SHARD_PRED,
#    VERSA_SHARD_OUTPUT, and VERSA_SCORE_CONFIG are set, the script
#    skips .scp generation and uses the pre-built shard files directly.
#    This allows run_all.py to launch N parallel instances of this
#    script, each processing a disjoint subset of utterances.
#
#    Environment variables:
#      VERSA_SHARD_GT      - Path to ground-truth .scp for this shard.
#      VERSA_SHARD_PRED    - Path to prediction .scp for this shard.
#      VERSA_SHARD_OUTPUT  - Path for JSON-lines output of this shard.
#      VERSA_SCORE_CONFIG  - Path to VERSA score config YAML.
#
# Prerequisites:
#   pip install versa  # or clone from github.com/shinjiwlab/versa
#
# License: MIT
# =========================================================================
set -euo pipefail

# Use the Python interpreter passed by the caller (run_all.py), or fall back
# to whichever python3 is on PATH.
PYTHON="${PYTHON:-python3}"

EXPERIMENT="${1:?Usage: bash eval/run_versa.sh <experiment_name>}"
CONFIG="configs/experiments/${EXPERIMENT}.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}" >&2
    exit 1
fi

# Extract output_dir from the YAML config.
OUTPUT_DIR=$(${PYTHON} -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('output_dir', 'outputs/default'))
")

RECON_DIR="${OUTPUT_DIR}/reconstructed/test"
MANIFEST="${RECON_DIR}/manifest.json"

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Reconstruction manifest not found: ${MANIFEST}" >&2
    echo "Run eval/reconstruct.py first." >&2
    exit 1
fi

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"
OUTPUT_FILE="${RESULTS_DIR}/${EXPERIMENT}_versa.json"
SCORE_CONFIG="eval/versa_score_config.yaml"

echo "════════════════════════════════════════════════════════════"
echo "VERSA Evaluation: ${EXPERIMENT}"
echo "────────────────────────────────────────────────────────────"
echo "  Config       : ${CONFIG}"
echo "  Score config : ${SCORE_CONFIG}"
echo "  Recon dir    : ${RECON_DIR}"
echo "  Output       : ${OUTPUT_FILE}"
echo "════════════════════════════════════════════════════════════"

# Check that versa is installed.
if ! ${PYTHON} -c "import versa" 2>/dev/null; then
    echo "ERROR: versa is not installed." >&2
    echo "Install with: pip install git+https://github.com/shinjiwlab/versa.git" >&2
    exit 1
fi

# Support shard-level overrides from _run_versa_shard (env vars).
# When VERSA_SHARD_GT is set, use pre-built scp files and output path
# instead of generating them from the manifest.
if [ -n "${VERSA_SHARD_GT:-}" ]; then
    GT_SCP="${VERSA_SHARD_GT}"
    PRED_SCP="${VERSA_SHARD_PRED}"
    OUTPUT_FILE="${VERSA_SHARD_OUTPUT}"
    SCORE_CONFIG="${VERSA_SCORE_CONFIG}"
else
    # Build wav.scp files for VERSA from the reconstruction manifest.
    # Format: <utt_id> <absolute_path>
    GT_SCP=$(mktemp)
    PRED_SCP=$(mktemp)
    trap 'rm -f "${GT_SCP}" "${PRED_SCP}"' EXIT

    ${PYTHON} -c "
import json, os
with open('${MANIFEST}') as f:
    manifest = json.load(f)
gt = open('${GT_SCP}', 'w')
pred = open('${PRED_SCP}', 'w')
for entry in manifest:
    utt_id = entry.get('id', os.path.splitext(os.path.basename(entry['original_path']))[0])
    orig = os.path.abspath(entry['original_path'])
    recon = os.path.abspath(entry['reconstructed_path'])
    gt.write(f'{utt_id} {orig}\n')
    pred.write(f'{utt_id} {recon}\n')
gt.close()
pred.close()
"
fi

# Run VERSA scorer.
# Apply protobuf compat shim (visqol uses removed MessageFactory.GetPrototype).
${PYTHON} -c "
import eval._protobuf_compat  # noqa: F401 – patches MessageFactory
import sys, runpy
sys.argv = [
    'versa.bin.scorer',
    '--pred', '${PRED_SCP}',
    '--gt', '${GT_SCP}',
    '--score_config', '${SCORE_CONFIG}',
    '--output_file', '${OUTPUT_FILE}',
    '--io', 'kaldi',
    '--use_gpu', 'true',
    '--verbose', '1',
]
runpy.run_module('versa.bin.scorer', run_name='__main__')
" 2>&1

echo ""
echo "════════════════════════════════════════════════════════════"
echo "VERSA results saved to: ${OUTPUT_FILE}"
echo "════════════════════════════════════════════════════════════"
