"""Pre-encode WAV dataset using SSL features (W2V2, HuBERT, Whisper, etc.).

Multi-channel WAV files (one channel per audio segment, created by
prepare_original_dataset.py with --n_segments) are handled correctly:
each channel is encoded separately and features are averaged in
**feature space** (not audio space).

Reads WAV files from data/prepared_ds_seg/{split}/ and writes
[T, feat_dim] float tensors to data/prepared_ds_seg_enc/{split}/.

File naming preserves the {prefix}_{label_id}_{...}.wav → .pt convention
required by MLAADFDDataset.
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import transformers
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Pre-encode dataset with HuggingFace features")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/prepared_ds_seg",
        help="Directory with raw WAV files containing train/dev/eval splits",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/prepared_ds_seg_enc",
        help="Directory to write .pt feature files",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="Wav2Vec2Model",
        help="HuggingFace model class name (e.g. Wav2Vec2Model, HubertModel)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-base",
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=5,
        help="Which hidden-state layer to save (0-indexed)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,eval",
        help="Comma-separated split names to encode",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of files per GPU batch",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16_000,
        help="Target sampling rate; files are resampled if needed",
    )
    return parser.parse_args()


def load_wav(path: str, target_sr: int) -> list[np.ndarray]:
    """Load WAV file, returning a list of segments (one per channel).

    Multi-channel WAV files store one segment per channel.
    Each segment is returned as a separate 1-D array so that
    features can be computed per-segment and averaged in feature
    space (matching prepare_original_dataset.py --encode).
    """
    audio, sr = librosa.load(path, sr=None, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # [1, T]
    if sr != target_sr:
        audio = np.stack([librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in audio])
    return [ch.astype(np.float32) for ch in audio]  # list of 1-D arrays


def _encode_segments(
    segments: list[np.ndarray],
    feature_extractor,
    model,
    layer: int,
    sampling_rate: int,
    device: str,
) -> torch.Tensor:
    """Encode a list of 1-D audio segments and return feature-space average."""
    # WhisperFeatureExtractor produces fixed-length mel spectrograms (3000 frames).
    # padding=True only pads to the longest sequence in the batch; for 1-s segments
    # that yields 100 frames, which Whisper rejects. Use padding="max_length" instead.
    is_whisper = isinstance(feature_extractor, transformers.WhisperFeatureExtractor)
    padding_arg = "max_length" if is_whisper else True

    inputs = feature_extractor(
        segments,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=padding_arg,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
        feats = outputs.encoder_hidden_states[layer].cpu().float()
    else:
        feats = outputs.hidden_states[layer].cpu().float()

    # feats: [n_segments, T, feat_dim] → average over segments
    return feats.mean(dim=0)  # [T, feat_dim]


def flush_batch(
    batch_segments: list,
    batch_paths: list,
    feature_extractor,
    model,
    layer: int,
    sampling_rate: int,
    out_split_dir: Path,
    device: str,
):
    """Encode and save a batch of files.

    Each entry in *batch_segments* is a list of 1-D arrays (segments/channels).
    Segments belonging to the same file are encoded together and averaged
    in feature space.
    """
    if not batch_segments:
        return

    for segments, wav_path in zip(batch_segments, batch_paths, strict=False):
        feat = _encode_segments(
            segments,
            feature_extractor,
            model,
            layer,
            sampling_rate,
            device,
        )
        out_path = out_split_dir / (wav_path.stem + ".pt")
        torch.save(feat, out_path)

    batch_segments.clear()
    batch_paths.clear()


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading {args.model_class} ({args.model_name}) …")
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(args.model_name)
    model_class = getattr(transformers, args.model_class)
    model = model_class.from_pretrained(args.model_name, output_hidden_states=True)
    model.eval()
    model.to(device)
    print(f"Model loaded. Using hidden layer {args.layer}.\n")

    for split in splits:
        in_dir = Path(args.input_dir) / split
        out_dir = Path(args.output_dir) / split
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_files = sorted(in_dir.glob("*.wav"))
        if not wav_files:
            print(f"[WARN] No .wav files found in {in_dir}, skipping.")
            continue

        # Skip already-encoded files
        already_done = {p.stem for p in out_dir.glob("*.pt")}
        wav_files = [p for p in wav_files if p.stem not in already_done]
        if not wav_files:
            print(f"{split}: all {len(already_done)} files already encoded, skipping.")
            continue

        print(f"{split}: encoding {len(wav_files)} files → {out_dir}")

        batch_segments: list = []
        batch_paths: list = []

        for wav_path in tqdm(wav_files, desc=split):
            segments = load_wav(str(wav_path), args.sampling_rate)
            batch_segments.append(segments)
            batch_paths.append(wav_path)

            if len(batch_segments) >= args.batch_size:
                flush_batch(
                    batch_segments,
                    batch_paths,
                    feature_extractor,
                    model,
                    args.layer,
                    args.sampling_rate,
                    out_dir,
                    device,
                )

        flush_batch(
            batch_segments,
            batch_paths,
            feature_extractor,
            model,
            args.layer,
            args.sampling_rate,
            out_dir,
            device,
        )

        print(f"{split}: done.")

    print("\nAll splits encoded.")


if __name__ == "__main__":
    main()
