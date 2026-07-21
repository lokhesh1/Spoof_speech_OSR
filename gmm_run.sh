# #!/bin/bash
# set -e  # exit immediately if any command fails

# # Run from the repo root (the dir this script lives in). All paths below are
# # relative to that root, so there is no cd / no double "gmm_baseline/" nesting.
# cd "$(dirname "$0")"

# ROOT="/home/hp/Desktop/spoof_attr/SER_PTM_Spoof_attr/MLAAD/Mlaad_v5/mlaad_v5"
# PY=".venv/bin/python"

# ARTIFACTS="gmm_baseline/artifacts/model.joblib"
# CACHE="gmm_baseline/cache/lfcc"
# RESULTS="gmm_baseline/results"

# # --- Training (commented out: model already trained at $ARTIFACTS) ----------
# # Uncomment this block to retrain from scratch.
# # echo "====================================="
# # echo "Training GMM baseline..."
# # echo "====================================="
# #
# # "$PY" gmm_baseline/train.py \
# #     --mlaad-root "$ROOT" \
# #     --out "$ARTIFACTS" \
# #     --cache-dir "$CACHE"

# echo "====================================="
# echo "Evaluating on eval split..."
# echo "====================================="

# "$PY" gmm_baseline/eval.py \
#     --mlaad-root "$ROOT" \
#     --artifacts "$ARTIFACTS" \
#     --cache-dir "$CACHE" \
#     --split eval \
#     --fine \
#     --out-json "$RESULTS/eval_metrics.json" \
#     --out-csv "$RESULTS/eval_preds.csv"

# echo "====================================="
# echo "Done."
# echo "====================================="
#!/usr/bin/env bash
set -euo pipefail

# Paths
PY="../.venv/bin/python"
ROOT="/home/hp/Desktop/spoof_attr/SER_PTM_Spoof_attr/MLAAD/Mlaad_v5/mlaad_v5"

# Cap BLAS threads to 1: the per-clip score_vector() loop makes many small
# GMM.score() calls, and BLAS auto-threading each one causes thread
# spin-up/teardown overhead to dominate (measured ~40% slower than single-
# threaded on this workload).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /home/hp/Desktop/spoof_attr/Spoof_speech_OSR/gmm_baseline

echo "========================================"
echo "Running Baseline 1: OCSVM Gate"
echo "========================================"

"$PY" eval.py \
    --mlaad-root "$ROOT" \
    --artifacts artifacts/model.joblib \
    --split eval \
    --cache-dir cache/lfcc \
    --fine \
    --out-json results/eval_ocsvm.json \
    --out-csv results/eval_ocsvm.csv

echo
echo "========================================"
echo "Running Baseline 2: Top-2 Likelihood Ratio Gate"
echo "========================================"

"$PY" eval_ratio.py \
    --mlaad-root "$ROOT" \
    --artifacts artifacts/model.joblib \
    --split eval \
    --cache-dir cache/lfcc \
    --fine \
    --thresh 2.0 \
    --out-json results/eval_ratio.json \
    --out-csv results/eval_ratio.csv

echo
echo "========================================"
echo "All evaluations completed successfully."
echo "========================================"