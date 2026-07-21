# Datasets & Data-Split Protocol

Shared reference for all four OSR experiments (`gmm_baseline`, `supra-to-sub`, `cicmm`, `osr_gate`), which all train/evaluate against the same `mlaad4sourcetracing/` protocol at the repo root.

## Dataset: MLAAD (via `mlaad4sourcetracing`)

- Source-tracing protocol built on **MLAADv5** (Multi-Language Audio Anti-Spoofing Dataset, Fraunhofer AISEC). Every sample in every split is **synthetic (fake) speech** — there is no bonafide/real class. The task is therefore *which TTS system generated this utterance* (multi-class attribution), not spoof/real detection.
- **24 known TTS generator classes** in train, spanning classical neural TTS (Tacotron2 variants, VITS, Glow-TTS, FastPitch, Speedy-Speech), large multilingual/zero-shot systems (XTTS v1.1/v2, Bark, Bark-small, MetaVoice-1B, Mars5, MeloTTS, vixTTS, per-language MMS-TTS), and a signal-processing baseline (Griffin-Lim).
- **8 languages** in train (en, fr, de, pl, it, ko, lt, ar), expanding to 21 languages in dev and 37 in eval as new, unseen languages are introduced.
- Each row across `train.csv`, `dev.csv`, `eval.csv`, and all `fine/` quadrant CSVs shares an identical 2-column schema: `path, model_name` (language is embedded in `path`, not a separate column).

## Split Protocol

Train defines the "known" set (24 models, 8 languages). Every dev/eval sample is independently tagged along two orthogonal axes — **language seen/unseen** and **TTS model seen/unseen** — producing a 2×2 quadrant partition: `lang_seen__model_seen`, `lang_seen__model_not_seen`, `lang_not_seen__model_seen`, `lang_not_seen__model_not_seen`. The `fine/` directory pre-splits dev/eval into these four quadrant CSVs so every experiment can report both a pooled open-set score and per-quadrant breakdowns (isolating generalization to a new architecture vs. a new language vs. both at once). Split sizes: **train** 11,100 samples / 8 languages / 24 models (all known); **dev** 12,000 / 21 / 25; **eval** 33,900 (33,791 after some experiments' de-duplication/filtering) / 37 / 64. All four experiments train only on the 24 known classes and use this same quadrant structure to jointly evaluate closed-set attribution accuracy and open-set (known-vs-unknown) detection.
