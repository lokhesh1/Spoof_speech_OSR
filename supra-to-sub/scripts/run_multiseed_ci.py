import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

MACRO_PATTERN = re.compile(r"^\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$")


def parse_args():
    parser = argparse.ArgumentParser("Run multi-seed training/evaluation and aggregate mean±std")
    parser.add_argument("--workspace", type=str, default=".", help="Repository root")
    parser.add_argument("--train_dataset", type=str, default="data/prepared_ds_seg_enc/")
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
        "--seeds", type=str, default="688,689,690", help="Comma-separated seed list"
    )
    parser.add_argument(
        "--base_out", type=str, default="exp/trained_models/paper_multiseed_lca_arc"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_classes", type=int, default=24)
    parser.add_argument(
        "--dry_run", action="store_true", help="Only print commands without executing"
    )
    return parser.parse_args()


def run_cmd(cmd, cwd, dry_run=False):
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def extract_macro_f1(report_path):
    with open(report_path) as f:
        for line in f:
            m = MACRO_PATTERN.match(line)
            if m:
                return float(m.group(3))
    raise ValueError(f"Could not parse macro-F1 from {report_path}")


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main():
    args = parse_args()
    repo_root = Path(args.workspace).resolve()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    results = []

    for seed in seeds:
        out_dir = repo_root / args.base_out / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "scripts/training/train_hier.py",
            "-d",
            args.train_dataset,
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
            "--num_classes",
            str(args.num_classes),
            "--batch_size",
            str(args.batch_size),
            "--num_epochs",
            str(args.num_epochs),
            "--seed",
            str(seed),
            "--out_folder",
            str(out_dir),
        ]

        metrics_cmd = [
            sys.executable,
            "scripts/get_classification_metrics.py",
            "--model_path",
            str(out_dir / "anti-spoofing_feat_model.pth"),
            "-d",
            args.eval_dataset,
            "--superclass_lut",
            args.superclass_lut,
        ]

        ood_cmd = [
            sys.executable,
            "scripts/ood_detector.py",
            "--model_path",
            str(out_dir / "anti-spoofing_feat_model.pth"),
            "-d",
            args.eval_dataset,
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

        run_cmd(train_cmd, cwd=str(repo_root), dry_run=args.dry_run)
        run_cmd(metrics_cmd, cwd=str(repo_root), dry_run=args.dry_run)
        run_cmd(ood_cmd, cwd=str(repo_root), dry_run=args.dry_run)

        if args.dry_run:
            continue

        eval_f1 = extract_macro_f1(out_dir / "eval_in_domain_results.txt")
        eval_sup_f1_path = out_dir / "eval_in_domain_results_sup.txt"
        eval_sup_f1 = extract_macro_f1(eval_sup_f1_path) if eval_sup_f1_path.exists() else None

        with open(out_dir / "ood" / "OOD_summary_mahalanobis.json") as f:
            ood_summary = json.load(f)

        results.append(
            {
                "seed": seed,
                "eval_macro_f1": eval_f1,
                "eval_sup_macro_f1": eval_sup_f1,
                "eval_ood_eer": ood_summary.get("eval_eer"),
                "eval_ood_eer_sup": ood_summary.get("eval_eer_sup"),
            }
        )

    summary_dir = repo_root / args.base_out
    summary_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("Dry run complete. No metrics aggregated.")
        return

    f1_vals = [r["eval_macro_f1"] for r in results]
    ood_eer_vals = [r["eval_ood_eer"] for r in results if r["eval_ood_eer"] is not None]
    sup_f1_vals = [r["eval_sup_macro_f1"] for r in results if r["eval_sup_macro_f1"] is not None]
    sup_eer_vals = [r["eval_ood_eer_sup"] for r in results if r["eval_ood_eer_sup"] is not None]

    f1_mean, f1_std = mean_std(f1_vals)
    eer_mean, eer_std = mean_std(ood_eer_vals)

    sup_f1_mean, sup_f1_std = (None, None)
    if sup_f1_vals:
        sup_f1_mean, sup_f1_std = mean_std(sup_f1_vals)

    sup_eer_mean, sup_eer_std = (None, None)
    if sup_eer_vals:
        sup_eer_mean, sup_eer_std = mean_std(sup_eer_vals)

    csv_path = summary_dir / "multiseed_results.csv"
    with open(csv_path, "w") as f:
        f.write("seed,eval_macro_f1,eval_sup_macro_f1,eval_ood_eer,eval_ood_eer_sup\n")
        for r in results:
            eval_sup_macro = (
                "" if r["eval_sup_macro_f1"] is None else f"{r['eval_sup_macro_f1']:.6f}"
            )
            eval_ood_eer = "" if r["eval_ood_eer"] is None else f"{r['eval_ood_eer']:.6f}"
            eval_ood_eer_sup = (
                "" if r["eval_ood_eer_sup"] is None else f"{r['eval_ood_eer_sup']:.6f}"
            )
            f.write(
                f"{r['seed']},{r['eval_macro_f1']:.6f},{eval_sup_macro},{eval_ood_eer},{eval_ood_eer_sup}\n"
            )

    summary_json = {
        "num_seeds": len(results),
        "eval_macro_f1_mean": f1_mean,
        "eval_macro_f1_std": f1_std,
        "eval_ood_eer_mean": eer_mean,
        "eval_ood_eer_std": eer_std,
        "eval_sup_macro_f1_mean": sup_f1_mean,
        "eval_sup_macro_f1_std": sup_f1_std,
        "eval_ood_eer_sup_mean": sup_eer_mean,
        "eval_ood_eer_sup_std": sup_eer_std,
    }

    json_path = summary_dir / "multiseed_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    md_path = summary_dir / "multiseed_summary.md"
    with open(md_path, "w") as f:
        f.write("# Priority 3 Multi-seed CI Summary\n\n")
        f.write(f"- Seeds: {', '.join(str(s) for s in seeds)}\n")
        f.write(f"- Eval macro-F1: {f1_mean:.4f} ± {f1_std:.4f}\n")
        f.write(f"- Eval OOD EER: {eer_mean * 100:.2f}% ± {eer_std * 100:.2f}%\n")
        if sup_f1_mean is not None:
            f.write(f"- Eval superclass macro-F1: {sup_f1_mean:.4f} ± {sup_f1_std:.4f}\n")
        if sup_eer_mean is not None:
            f.write(
                f"- Eval superclass OOD EER: {sup_eer_mean * 100:.2f}% ± {sup_eer_std * 100:.2f}%\n"
            )
        f.write("\n## Paper-ready text (template)\n")
        f.write(
            "Across multiple random seeds, performance remains stable with low variance, indicating that gains are not due to favorable initialization. "
        )
        f.write("We therefore report mean ± standard deviation for all key metrics.\n")

    print(f"Saved per-seed CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved paper summary markdown: {md_path}")


if __name__ == "__main__":
    main()
