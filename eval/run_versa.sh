#!/usr/bin/env bash
# Run VERSA evaluation toolkit (optional, heavy dependency).
# VERSA provides 90+ metrics. We use it for comprehensive final evaluation.
#
# Usage:
#   bash eval/run_versa.sh <experiment_name>
#
# Prerequisites:
#   pip install versa  # or clone from github.com/shinjiwlab/versa
#
# License: MIT
set -euo pipefail

EXPERIMENT="${1:?Usage: bash eval/run_versa.sh <experiment_name>}"
CONFIG="configs/experiments/${EXPERIMENT}.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}" >&2
    exit 1
fi

# Extract output_dir from the YAML config (simple grep for top-level key).
OUTPUT_DIR=$(python3 -c "
import yaml, sys
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
OUTPUT_JSON="${RESULTS_DIR}/${EXPERIMENT}_versa.json"

echo "════════════════════════════════════════════════════════════"
echo "VERSA Evaluation: ${EXPERIMENT}"
echo "────────────────────────────────────────────────────────────"
echo "  Config     : ${CONFIG}"
echo "  Recon dir  : ${RECON_DIR}"
echo "  Output     : ${OUTPUT_JSON}"
echo "════════════════════════════════════════════════════════════"

# Check that versa is installed.
if ! python3 -c "import versa" 2>/dev/null; then
    echo "ERROR: versa is not installed." >&2
    echo "Install with: pip install versa" >&2
    echo "  or clone: git clone https://github.com/shinjiwlab/versa" >&2
    exit 1
fi

# Build file lists for VERSA from the manifest.
REF_LIST=$(mktemp)
DEG_LIST=$(mktemp)
trap 'rm -f "${REF_LIST}" "${DEG_LIST}"' EXIT

python3 -c "
import json, sys
with open('${MANIFEST}') as f:
    manifest = json.load(f)
ref_f = open('${REF_LIST}', 'w')
deg_f = open('${DEG_LIST}', 'w')
for entry in manifest:
    ref_f.write(entry['original_path'] + '\n')
    deg_f.write(entry['reconstructed_path'] + '\n')
ref_f.close()
deg_f.close()
"

# Run VERSA evaluation.
python3 -m versa \
    --ref_list "${REF_LIST}" \
    --deg_list "${DEG_LIST}" \
    --output_json "${OUTPUT_JSON}" \
    2>&1

echo ""
echo "════════════════════════════════════════════════════════════"
echo "VERSA results saved to: ${OUTPUT_JSON}"
echo "════════════════════════════════════════════════════════════"
