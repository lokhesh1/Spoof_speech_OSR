#!/usr/bin/env bash
set -euo pipefail

# C-ICMM experiment runner for MLAADv5 source tracing.
#
# Prerequisites:
#   1. Extract XLS-R-300M features (adjust --mlaad-root to your audio path):
#        python extract_mlaad.py \
#          --hf-id facebook/wav2vec2-xls-r-300m \
#          --layers 5 \
#          --mlaad-root /path/to/Mlaad_v5/mlaad_v5 \
#          --protocol-dir mlaad4sourcetracing \
#          --out-dir feats_xlsr \
#          --pooling none
#
#   2. Run this script from the repo root:
#        bash cicmm_run.sh
#
# Each trial writes to its own artifacts/ subdirectory under cicmm/.
# Results land in cicmm/results/.

FEAT_ROOT="../PTM_Feat/feats_xlsr"
LAYER=5
cd "$(dirname "$0")/cicmm"

# echo "============================================"
# echo "Trial 1/6: BASELINE (e256, manual, full-cov)"
# echo "============================================"
# python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
#     --embed-dim 256 --icmm-weighting manual --gmm-covariance full \
#     --out-dir artifacts/cicmm_e256_manual_full

# echo "============================================"
# echo "Trial 2/6: e512, manual ICMM, full-cov GMM"
# echo "============================================"
# python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
#     --embed-dim 512 --icmm-weighting manual --gmm-covariance full \
#     --out-dir artifacts/cicmm_e512_manual_full

# echo "============================================"
# echo "Trial 3/6: e256, auto ICMM, full-cov GMM"
# echo "============================================"
# python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
#     --embed-dim 256 --icmm-weighting auto --gmm-covariance full \
#     --out-dir artifacts/cicmm_e256_auto_full

# echo "============================================"
# echo "Trial 4/6: e512, auto ICMM, full-cov GMM"
# echo "============================================"
# python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
#     --embed-dim 512 --icmm-weighting auto --gmm-covariance full \
#     --out-dir artifacts/cicmm_e512_auto_full

# echo "============================================"
# echo "Trial 5/6: e256, manual ICMM, adaptive-cov"
# echo "============================================"
# python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
#     --embed-dim 256 --icmm-weighting manual --gmm-covariance adaptive \
#     --out-dir artifacts/cicmm_e256_manual_adaptive

# echo "============================================"
# echo "Trial 6/6: e512, auto ICMM, adaptive-cov"
# echo "============================================"
# python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
#     --embed-dim 512 --icmm-weighting auto --gmm-covariance adaptive \
#     --out-dir artifacts/cicmm_e512_auto_adaptive

echo "============================================"
echo "Trial 7: e512, manual, full-cov, centroid=ema"
echo "============================================"
python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
    --embed-dim 512 --icmm-weighting manual --gmm-covariance full \
    --centroid-mode ema \
    --out-dir artifacts/cicmm_e512_manual_full-cov_ema

echo "============================================"
echo "Trial 8: e512, manual, full-cov, centroid=batch"
echo "============================================"
python main.py --feat-root "$FEAT_ROOT" --layer "$LAYER" \
    --embed-dim 512 --icmm-weighting manual --gmm-covariance full \
    --centroid-mode batch \
    --out-dir artifacts/cicmm_e512_manual_full-cov_batch

echo ""
echo "All trials complete. Results in cicmm/results/"
