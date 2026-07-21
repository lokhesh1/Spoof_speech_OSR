# Spoof_speech_OSR

Open-set recognition / source tracing on MLAADv5 spoofed speech — attribute a clip to a known TTS model or reject it as unknown.

Run every script from the repo root. Edit `MLAAD_ROOT`/`ROOT`/`PY` at the top of each script if your paths differ. Protocol CSVs: `mlaad4sourcetracing/{train,dev,eval}.csv`.

---

## `osr_run.sh` — OSR gate (XLS-R, Busemann / Euclidean heads)

- Runs `train.py -> embed.py -> gate.py -> eval.py` for `{layer 1, layer 5} x {busemann, euclidean}`.
- Usage: `./osr_run.sh`
- Resumable: skips a stage if its output marker already exists; `FORCE=1` to redo everything.
- Env overrides: `RUN_EXTRACT=0`, `EPOCHS=30 BATCH=32 DEVICE=auto`.
- Report: [`docs/osr_gate.md`](docs/osr_gate.md)

## `gmm_run.sh` — GMM baseline (LFCC + OCSVM / likelihood-ratio gates)

- Evaluates two known/unknown gating strategies against a pretrained GMM.
- Usage: `./gmm_run.sh`
- Training call is commented out at the top of the script — uncomment to retrain.
- Report: [`docs/gmm_baseline.md`](docs/gmm_baseline.md)

## `cicmm_run.sh` — C-ICMM trials

- Prerequisite: XLS-R features already extracted via `extract_mlaad.py`.
- Usage: `bash cicmm_run.sh`
- Each trial varies `--embed-dim`, `--icmm-weighting`, `--gmm-covariance`, `--centroid-mode`; most are commented out by default.
- Report: [`docs/cicmm.md`](docs/cicmm.md)

## `hier_run.sh` — Hier-Spec hierarchical OOD pipeline

- Stages: extract Wav2Vec2 layer-5 features -> train -> calibrate OOD thresholds on dev -> evaluate on eval.
- Usage: `./hier_run.sh`
- Report: none pushed yet.

## `supra_run.sh` — supra-to-sub (H-Arch, ArcFace)

- Stages: dedup protocol CSVs -> extract features -> train H-Arch -> in-domain eval -> Mahalanobis OOD eval.
- Usage: `./supra_run.sh`
- Report: not pushed yet.

---

## Reports

- [OSR Gate](docs/osr_gate.md)
- [GMM Baseline](docs/gmm_baseline.md)
- [C-ICMM](docs/cicmm.md)
- [Datasets & Protocol](docs/datasets_and_protocol.md)
