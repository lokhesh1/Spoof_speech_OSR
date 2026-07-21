#!/usr/bin/env bash
# End-to-end OSR-gate pipeline.
#
#   1. Extract XLS-R-300M features (protocol clips only, fp32, full sequence)
#      for layers 1 and 5 in a single pass -> feats_xlsr/.
#   2. For each layer in {1,5} x head in {busemann, euclidean}:
#         train.py -> embed.py -> gate.py -> eval.py
#      producing osr_gate/artifacts/<head>_layer<NN>/ and
#      osr_gate/results/<head>_layer<NN>.json.
#
# Long job (extraction is hours). Run detached if you like:
#   nohup ./osr_run.sh > osr_run.log 2>&1 &
#   tail -f osr_run.log
#
# Resumable: each stage is skipped if its output already exists, so a re-run
# continues from where a crash stopped. Set FORCE=1 to redo everything.
#
# Env overrides:
#   RUN_EXTRACT=0   skip extraction (features already on disk)
#   FORCE=1         ignore existing outputs and redo every stage
#   EPOCHS=30 BATCH=32 DEVICE=auto
#   If eval/train OOMs on the 8 GB GPU with long clips, lower BATCH.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$REPO/.venv/bin/python"
MLAAD_ROOT="/home/hp/Desktop/spoof_attr/SER_PTM_Spoof_attr/MLAAD/Mlaad_v5/mlaad_v5"
FEAT_ROOT="$REPO/PTM_Feat/feats_xlsr"
PROTO_DIR="$REPO/mlaad4sourcetracing"
HF_ID="facebook/wav2vec2-xls-r-300m"

LAYERS=(1 5)
HEADS=(busemann euclidean)

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-32}"
DEVICE="${DEVICE:-auto}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
FORCE="${FORCE:-0}"

log() { echo -e "\n=== $* ($(date '+%H:%M:%S')) ==="; }

# done <marker-file> : true if the stage's output exists and FORCE is off.
done_already() { [[ "$FORCE" != "1" && -f "$1" ]]; }

# --------------------------------------------------------------------------- #
# 1. Feature extraction (protocol-only, all requested layers in one forward)
# --------------------------------------------------------------------------- #
if [[ "$RUN_EXTRACT" == "1" ]]; then
  log "Extract XLS-R layers ${LAYERS[*]} -> $FEAT_ROOT (protocol-only, fp32)"
  "$PY" "$REPO/extract_mlaad.py" \
      --hf-id "$HF_ID" \
      --mlaad-root "$MLAAD_ROOT" \
      --out-dir "$FEAT_ROOT" \
      --layers "${LAYERS[@]}" \
      --pooling none \
      --protocol-dir "$PROTO_DIR" \
      --device "$DEVICE" \
      --log-every 500
else
  log "Skip extraction (RUN_EXTRACT=0)"
fi

# --------------------------------------------------------------------------- #
# 2. train -> embed -> gate -> eval, per (layer, head)
# --------------------------------------------------------------------------- #
cd "$REPO/osr_gate"
for layer in "${LAYERS[@]}"; do
  LNN=$(printf "%02d" "$layer")
  for head in "${HEADS[@]}"; do
    ART="artifacts/${head}_layer${LNN}"
    RESULT="results/${head}_layer${LNN}.json"

    if done_already "$ART/meta.json"; then
      log "[$head | layer $layer] train  SKIP (meta.json exists)"
    else
      log "[$head | layer $layer] train"
      "$PY" train.py --feat-root "$FEAT_ROOT" --layer "$layer" --head "$head" \
          --epochs "$EPOCHS" --batch-size "$BATCH" --device "$DEVICE"
    fi

    if done_already "$ART/gauss_stats.npz"; then
      log "[$head | layer $layer] embed  SKIP (gauss_stats.npz exists)"
    else
      log "[$head | layer $layer] embed"
      "$PY" embed.py --feat-root "$FEAT_ROOT" --layer "$layer" --head "$head" \
          --device "$DEVICE"
    fi

    if done_already "$ART/gate.pkl"; then
      log "[$head | layer $layer] gate  SKIP (gate.pkl exists)"
    else
      log "[$head | layer $layer] gate"
      "$PY" gate.py --artifacts-dir "$ART"
    fi

    if done_already "$RESULT"; then
      log "[$head | layer $layer] eval  SKIP ($RESULT exists)"
    else
      log "[$head | layer $layer] eval"
      "$PY" eval.py --artifacts-dir "$ART" --feat-root "$FEAT_ROOT" --layer "$layer"
    fi
  done
done

log "Done. results -> osr_gate/results/  |  artifacts -> osr_gate/artifacts/"


# rm -rf osr_gate/results
# RUN_EXTRACT=0 ./osr_run.sh