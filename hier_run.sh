#!/bin/bash
set -e  # exit immediately if any command fails

# Run from the repo root (the dir this script lives in). Paths are root-relative.
cd "$(dirname "$0")"

ROOT="/home/hp/Desktop/spoof_attr/SER_PTM_Spoof_attr/MLAAD/Mlaad_v5/mlaad_v5"
PY=".venv/bin/python"

PROTO="mlaad4sourcetracing"
FEATS="feats_mlaad"                       # Wav2Vec2 layer-05 frame cache
ARTIFACTS="hier_spec/artifacts"
RESULTS="hier_spec/results"

# --- Step 1: cache Wav2Vec2-Base layer-5 features (skips existing) -----------
echo "====================================="
echo "Extracting Wav2Vec2 layer-5 features..."
echo "====================================="
"$PY" hier_spec/extract_feats.py \
    --mlaad-root "$ROOT" \
    --protocol-dir "$PROTO" \
    --out-dir "$FEATS"

# --- Step 2: train Hier-Spec + estimate Mahalanobis stats -------------------
echo "====================================="
echo "Training Hier-Spec..."
echo "====================================="
"$PY" hier_spec/train.py \
    --feat-root "$FEATS" \
    --protocol-dir "$PROTO" \
    --out-dir "$ARTIFACTS"

# --- Step 3: calibrate Stage-1/Stage-2 thresholds on dev (EER) --------------
echo "====================================="
echo "Calibrating OOD thresholds on dev..."
echo "====================================="
"$PY" hier_spec/calibrate.py \
    --feat-root "$FEATS" \
    --protocol-dir "$PROTO" \
    --artifacts "$ARTIFACTS" \
    --split dev

# --- Step 4: evaluate on the eval split (overall + fine subsplits) ----------
echo "====================================="
echo "Evaluating on eval split..."
echo "====================================="
"$PY" hier_spec/eval.py \
    --feat-root "$FEATS" \
    --protocol-dir "$PROTO" \
    --artifacts "$ARTIFACTS" \
    --split eval \
    --fine \
    --out-json "$RESULTS/eval_metrics.json" \
    --out-csv "$RESULTS/eval_preds.csv"

echo "====================================="
echo "Done."
echo "====================================="
