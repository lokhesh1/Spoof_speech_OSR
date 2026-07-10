#!/bin/bash
set -e  # exit immediately if any command fails

# Run from the repo root (the dir this script lives in). Paths are root-relative.
cd "$(dirname "$0")"

ROOT="/home/hp/Desktop/spoof_attr/SER_PTM_Spoof_attr/MLAAD/Mlaad_v5/mlaad_v5"
PY=".venv/bin/python"

SUPRA="supra-to-sub"
PROTO_RAW="mlaad4sourcetracing"                       # original protocol CSVs (57000 rows, eval.csv has 109 dupes)
PROTO_DEDUP="$SUPRA/data/protocol_dedup"               # deduped copy (56891 rows), used for extraction
FEATS="$SUPRA/data/prepared_ds_seg_enc"                # fixed-length (1s x4-seg-avg) encoded features
MODEL_DIR="$SUPRA/exp/trained_models/hier_cas_arc_m03" # H-Arch, ArcFace m=0.3 (matches makefile's train_hier_arch)
SUP_LUT_KNOWN="$SUPRA/data/superclass_mapping_known.csv"
SUP_LUT_FULL="$SUPRA/data/superclass_mapping_test.csv"
LABEL_FILE="$SUPRA/data/label_assignment.txt"
SUP_LABEL_FILE="$SUPRA/data/label_assignment_superclass.txt"

# --- Step 0: dedup eval.csv (109 exact-duplicate rows) into its own protocol dir ------
# supra-to-sub's own scripts don't dedup; without this eval.csv would be encoded/scored
# with 109 redundant clips, wasting compute and making eval counts not match the other
# pipelines (gmm_baseline / hier_spec), which both dedup via protocols_mlaad.load_split.
echo "====================================="
echo "Deduping protocol CSVs..."
echo "====================================="
mkdir -p "$PROTO_DEDUP"
"$PY" - <<PYEOF
import pandas as pd
import shutil

for split in ["train", "dev"]:
    shutil.copy(f"$PROTO_RAW/{split}.csv", f"$PROTO_DEDUP/{split}.csv")

eval_df = pd.read_csv("$PROTO_RAW/eval.csv")
before = len(eval_df)
eval_df = eval_df.drop_duplicates()
eval_df.to_csv("$PROTO_DEDUP/eval.csv", index=False)
print(f"eval.csv: {before} -> {len(eval_df)} rows ({before - len(eval_df)} duplicates dropped)")
PYEOF

# --- Step 1: feature extraction (Wav2Vec2-base, layer 5, 4x1s segments averaged) ------
# Skips files that are already encoded (prepare_original_dataset.py checks target_dir
# contents vs dataset length), so re-running this script is safe / resumable.
echo "====================================="
echo "Extracting fixed-length Wav2Vec2 features..."
echo "====================================="
"$PY" "$SUPRA/scripts/prepare_original_dataset.py" \
    --mlaad_path "$ROOT" \
    --protocol_path "$PROTO_DEDUP" \
    --out_folder "$FEATS" \
    --n_segments 4 \
    --encode \
    --batch_size 8 \
    --num_workers 2

# --- Step 2: train H-Arch (architecture-gated hierarchical classifier, ArcFace m=0.3) --
echo "====================================="
echo "Training H-Arch..."
echo "====================================="
"$PY" "$SUPRA/scripts/training/train_hier.py" \
    --path_to_dataset "$FEATS" \
    --pre_encoded True \
    --out_folder "$MODEL_DIR" \
    --pre_augmented True --is_segmented True \
    --weighted_sampling True \
    --use_arc_margin \
    --easy_margin True --arc_m 0.3 \
    --hierarchy_type "H-Arch" --superclass_lut "$SUP_LUT_KNOWN"

# --- Step 3: in-domain classification report (accuracy, per-class F1, macro F1) -------
echo "====================================="
echo "In-domain evaluation..."
echo "====================================="
"$PY" "$SUPRA/scripts/get_classification_metrics.py" \
    --model_path "$MODEL_DIR/anti-spoofing_feat_model.pth" \
    --path_to_dataset "$FEATS" \
    --superclass_lut "$SUP_LUT_KNOWN" \
    --batch_size 32

# --- Step 4: OOD / unknown-model detection (F1 for unknown, EER, AUC) -----------------
echo "====================================="
echo "OOD evaluation (mahalanobis)..."
echo "====================================="
"$PY" "$SUPRA/scripts/ood_detector.py" \
    --model_path "$MODEL_DIR/anti-spoofing_feat_model.pth" \
    --path_to_dataset "$FEATS" \
    --label_assignment_file "$LABEL_FILE" \
    --sup_label_assignment_file "$SUP_LABEL_FILE" \
    --superclass_lut_known "$SUP_LUT_KNOWN" \
    --superclass_lut_full "$SUP_LUT_FULL" \
    --confidence_scaling 'sup' \
    --ood_method "mahalanobis" \
    --batch_size 32

echo "====================================="
echo "Done."
echo "====================================="
echo "In-domain report (accuracy, F1, macro F1): $MODEL_DIR/eval_in_domain_results.txt"
echo "OOD report (F1 for unknown, accuracy):     $MODEL_DIR/ood/OOD_eval_results_mahalanobis.txt"
echo "OOD summary (EER, AUC):                    $MODEL_DIR/ood/OOD_summary_mahalanobis.json"
