# Spoof_speech_OSR

Open-set recognition / source tracing experiments on MLAADv5 spoofed speech: given a
clip, attribute it to one of the known TTS models seen in training, or reject it as
coming from an unseen ("unknown") model. Several independent pipelines are compared,
each launched by its own top-level `*_run.sh` script.

All scripts assume the MLAADv5 audio dataset lives at a fixed path (edit the
`MLAAD_ROOT` / `ROOT` variable at the top of each script if yours differs) and the
protocol split CSVs live in `mlaad4sourcetracing/` (`train.csv` / `dev.csv` /
`eval.csv`). Run every script from the **repo root** unless noted otherwise.

---

## `osr_run.sh` — OSR gate (XLS-R + hyperbolic Busemann / Euclidean heads)

End-to-end pipeline for `osr_gate/`: extracts XLS-R-300M features, then for each of
`{layer 1, layer 5} x {busemann, euclidean}` runs `train.py -> embed.py -> gate.py ->
eval.py`.

```bash
./osr_run.sh
# or, detached (extraction can take hours):
nohup ./osr_run.sh > osr_run.log 2>&1 &
tail -f osr_run.log
```

- **Resumable**: each stage is skipped if its output marker already exists
  (`meta.json`, `gauss_stats.npz`, `gate.pkl`, `results/<head>_layer<NN>.json`). Set
  `FORCE=1` to redo every stage regardless.
- **Env overrides**: `RUN_EXTRACT=0` (skip feature extraction if `PTM_Feat/feats_xlsr/`
  already has the features), `EPOCHS=30 BATCH=32 DEVICE=auto`, e.g.:
  ```bash
  RUN_EXTRACT=0 EPOCHS=20 ./osr_run.sh
  ```
- **Outputs**: `osr_gate/artifacts/<head>_layer<NN>/` (checkpoints, embeddings, fitted
  gate) and `osr_gate/results/<head>_layer<NN>.json` (detection AUROC/AUPR/EER,
  pointwise F1, OSCR, closed-set accuracy, per-unknown-model F1, ablation).

## `gmm_run.sh` — GMM baseline (LFCC features, OCSVM / likelihood-ratio gates)

Runs from `gmm_baseline/` (the script `cd`s there itself). Evaluates two known/unknown
gating strategies against an **already-trained** GMM classifier
(`gmm_baseline/artifacts/model.joblib`):

```bash
./gmm_run.sh
```

- **Baseline 1 — OCSVM gate**: `eval.py --fine` (per-`fine`-subsplit metrics too).
- **Baseline 2 — Top-2 likelihood-ratio gate**: `eval_ratio.py --thresh 2.0`.
- Sets `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1` /
  `NUMEXPR_NUM_THREADS=1` — BLAS auto-threading on the per-clip `GMM.score()` loop is
  ~40% slower than single-threaded on this workload.
- **To retrain from scratch**: the training call is commented out at the top of the
  script (`gmm_baseline/train.py --mlaad-root ... --out artifacts/model.joblib
  --cache-dir cache/lfcc`) — uncomment it if `model.joblib` doesn't exist yet.
- **Outputs**: `gmm_baseline/results/eval_ocsvm.{json,csv}` and
  `gmm_baseline/results/eval_ratio.{json,csv}`.

## `cicmm_run.sh` — C-ICMM trials

Runs from `cicmm/` (the script `cd`s there itself).

**Prerequisite** — XLS-R features must already exist at `PTM_Feat/feats_xlsr/` (or
wherever `FEAT_ROOT` in the script points):
```bash
python extract_mlaad.py \
    --hf-id facebook/wav2vec2-xls-r-300m --layers 5 \
    --mlaad-root /path/to/Mlaad_v5/mlaad_v5 \
    --protocol-dir mlaad4sourcetracing \
    --out-dir feats_xlsr --pooling none
```

Then, from the repo root:
```bash
bash cicmm_run.sh
```

- Each trial is a separate `python cicmm/main.py` call varying `--embed-dim`
  (256/512), `--icmm-weighting` (manual/auto), `--gmm-covariance` (full/adaptive), and
  `--centroid-mode` (ema/batch). Most trials are commented out in the script by
  default — uncomment the ones you want to (re)run.
- **Outputs**: `cicmm/artifacts/cicmm_<config>/` per trial, results under
  `cicmm/results/`.

## `hier_run.sh` — Hier-Spec hierarchical OOD pipeline

Four sequential stages, all from the repo root:

```bash
./hier_run.sh
```

1. **Extract** Wav2Vec2-Base layer-5 features (`hier_spec/extract_feats.py`, skips
   files already cached) → `feats_mlaad/`.
2. **Train** Hier-Spec + estimate Mahalanobis stats (`hier_spec/train.py`) →
   `hier_spec/artifacts/`.
3. **Calibrate** Stage-1/Stage-2 OOD thresholds on `dev` via EER
   (`hier_spec/calibrate.py`).
4. **Evaluate** on `eval` (overall + `--fine` subsplits) →
   `hier_spec/results/eval_metrics.json` + `eval_preds.csv`.

## `supra_run.sh` — supra-to-sub (H-Arch hierarchical classifier, ArcFace)

Five sequential stages, all from the repo root:

```bash
./supra_run.sh
```

1. **Dedup** protocol CSVs (`eval.csv` has 109 exact-duplicate rows) into
   `supra-to-sub/data/protocol_dedup/` — needed so eval counts match the other
   pipelines.
2. **Extract** fixed-length Wav2Vec2-base features (4 x 1 s segments, averaged) via
   `supra-to-sub/scripts/prepare_original_dataset.py` → resumable, skips already-encoded
   files.
3. **Train** H-Arch (architecture-gated hierarchical classifier, ArcFace margin
   `m=0.3`) → `supra-to-sub/exp/trained_models/hier_cas_arc_m03/`.
4. **In-domain evaluation**: accuracy, per-class F1, macro-F1
   (`get_classification_metrics.py`).
5. **OOD evaluation**: Mahalanobis-based unknown-model detection — F1(unknown), EER,
   AUC (`ood_detector.py`).

- **Outputs**: in-domain report at
  `$MODEL_DIR/eval_in_domain_results.txt`; OOD report at
  `$MODEL_DIR/ood/OOD_eval_results_mahalanobis.txt`; OOD summary (EER/AUC) at
  `$MODEL_DIR/ood/OOD_summary_mahalanobis.json`.

---

## Notes

- Every script hardcodes `MLAAD_ROOT`/`ROOT` (the raw MLAADv5 audio path) and `PY`
  (the Python interpreter, usually `.venv/bin/python`) near the top — edit these if
  your paths differ.
- The pipelines are independent of each other; run whichever ones you need, in any
  order. They share the same protocol CSVs (`mlaad4sourcetracing/{train,dev,eval}.csv`)
  but cache features/artifacts under their own subdirectories.
