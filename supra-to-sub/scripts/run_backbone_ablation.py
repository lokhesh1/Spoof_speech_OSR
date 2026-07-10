import argparse
import subprocess
import sys
from pathlib import Path

BACKBONES = {
    "w2v2_base": {
        "model_class": "Wav2Vec2Model",
        "hugging_face_path": "facebook/wav2vec2-base",
        "model_layer": "5",
    },
    "hubert_base": {
        "model_class": "HubertModel",
        "hugging_face_path": "facebook/hubert-base-ls960",
        "model_layer": "5",
    },
    "whisper_small": {
        "model_class": "WhisperModel",
        "hugging_face_path": "openai/whisper-small",
        "model_layer": "5",
    },
    "wavlm_base_plus": {
        "model_class": "WavLMModel",
        "hugging_face_path": "microsoft/wavlm-base-plus",
        "model_layer": "6",
    },
}


def parse_args():
    parser = argparse.ArgumentParser("Run Priority-4 SSL backbone ablation")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument(
        "--raw_dataset",
        type=str,
        default="data/MLAADv5_for_sourcetracing/",
        help="MLAAD root containing fake/<lang>/<model>/ and mlaad4sourcetracing/ (used for re-encoding per backbone)",
    )
    parser.add_argument(
        "--protocol_path",
        type=str,
        default=None,
        help="Path to the protocol CSVs. Defaults to <raw_dataset>/mlaad4sourcetracing/",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="data/prepared_ds_seg_enc/",
        help="Default pre-encoded dataset (W2V2). Per-backbone dirs auto-generated.",
    )
    parser.add_argument("--eval_dataset", type=str, default="data/prepared_ds_seg_enc/")
    parser.add_argument("--superclass_lut", type=str, default="data/superclass_mapping_known.csv")
    parser.add_argument(
        "--superclass_lut_full", type=str, default="data/superclass_mapping_test.csv"
    )
    parser.add_argument("--label_assignment_file", type=str, default="data/label_assignment.txt")
    parser.add_argument(
        "--sup_label_assignment_file",
        type=str,
        default="data/label_assignment_superclass.txt",
    )
    parser.add_argument(
        "--out_root", type=str, default="exp/trained_models/paper_backbone_ablation"
    )
    parser.add_argument("--seed", type=int, default=688)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument("--run", action="store_true", help="Actually execute commands")
    parser.add_argument(
        "--backbones",
        type=str,
        default=None,
        help="Comma-separated list of backbones to run (default: all). E.g. hubert_base",
    )
    return parser.parse_args()


def run_cmd(cmd, cwd, run):
    print("$", " ".join(cmd))
    if run:
        subprocess.run(cmd, cwd=cwd, check=True)


def ensure_encoded_dataset(backbone_name, conf, raw_dataset, protocol_path, repo_root, run):
    """Pre-encode dataset for this backbone using prepare_original_dataset.py."""
    enc_dir = repo_root / f"data/prepared_ds_seg_enc_{backbone_name}"

    # Skip if every split directory already has .pt files (treat any non-empty
    # subdir as previously encoded — coarse but avoids hardcoded row counts that
    # don't hold for subsets).
    all_done = all(
        (enc_dir / split).exists()
        and any(not f.name.startswith("d_") for f in (enc_dir / split).glob("*.pt"))
        for split in ("train", "dev", "eval")
    )
    if all_done:
        print(
            f"[INFO] Encoded dataset for {backbone_name} already exists at {enc_dir}, skipping encoding."
        )
        return str(enc_dir)

    print(f"[INFO] Encoding dataset for {backbone_name} using prepare_original_dataset.py...")

    encode_cmd = [
        sys.executable,
        "scripts/prepare_original_dataset.py",
        "--mlaad_path",
        raw_dataset,
        "--protocol_path",
        protocol_path,
        "--out_folder",
        str(enc_dir),
        "--encode",
        "--model_class",
        conf["model_class"],
        "--hugging_face_path",
        conf["hugging_face_path"],
        "--model_layer",
        conf["model_layer"],
        "--n_segments",
        "4",
        "--max_length",
        "4",
        "--batch_size",
        "16",
        "--sampling_rate",
        "16000",
    ]
    run_cmd(encode_cmd, cwd=str(repo_root), run=run)

    return str(enc_dir)


def main():
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    protocol_path = args.protocol_path or str(Path(args.raw_dataset) / "mlaad4sourcetracing")

    backbones_to_run = set(args.backbones.split(",")) if args.backbones else set(BACKBONES.keys())

    for name, conf in BACKBONES.items():
        if name not in backbones_to_run:
            print(f"[SKIP] {name} not in --backbones list")
            continue
        out_dir = repo_root / args.out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Determine the correct encoded dataset for this backbone
        if name == "w2v2_base":
            train_ds = args.train_dataset  # Use default W2V2 encoded dataset
            eval_ds = args.eval_dataset
        else:
            train_ds = ensure_encoded_dataset(
                name, conf, args.raw_dataset, protocol_path, repo_root, args.run
            )
            eval_ds = train_ds

        train_cmd = [
            sys.executable,
            "scripts/training/train_hier.py",
            "-d",
            train_ds,
            "--hierarchy_type",
            "H-Arch",
            "--superclass_lut",
            args.superclass_lut,
            "--use_arc_margin",
            "--arc_m",
            "0.3",
            "--easy_margin",
            "True",
            "--weighted_sampling",
            "True",
            "--pre_augmented",
            "True",
            "--pre_encoded",
            "True",
            "--seed",
            str(args.seed),
            "--num_epochs",
            str(args.num_epochs),
            "--batch_size",
            str(args.batch_size),
            "--num_classes",
            str(args.num_classes),
            "--model_class",
            conf["model_class"],
            "--hugging_face_path",
            conf["hugging_face_path"],
            "--model_layer",
            conf["model_layer"],
            "--out_folder",
            str(out_dir),
        ]
        metric_cmd = [
            sys.executable,
            "scripts/get_classification_metrics.py",
            "--model_path",
            str(out_dir / "anti-spoofing_feat_model.pth"),
            "-d",
            eval_ds,
            "--superclass_lut",
            args.superclass_lut,
        ]
        ood_cmd = [
            sys.executable,
            "scripts/ood_detector.py",
            "--model_path",
            str(out_dir / "anti-spoofing_feat_model.pth"),
            "-d",
            eval_ds,
            "--label_assignment_file",
            args.label_assignment_file,
            "--sup_label_assignment_file",
            args.sup_label_assignment_file,
            "--superclass_lut_known",
            args.superclass_lut,
            "--superclass_lut_full",
            args.superclass_lut_full,
            "--confidence_scaling",
            "sup",
            "--ood_method",
            "mahalanobis",
        ]

        run_cmd(train_cmd, cwd=str(repo_root), run=args.run)
        run_cmd(metric_cmd, cwd=str(repo_root), run=args.run)
        run_cmd(ood_cmd, cwd=str(repo_root), run=args.run)

    print("Backbone ablation commands prepared/executed.")


if __name__ == "__main__":
    main()
