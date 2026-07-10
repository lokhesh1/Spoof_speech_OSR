# Adapted from https://github.com/piotrkawa/audio-deepfake-source-tracing


import argparse
import json
import os
import sys
from pathlib import Path

# Enables running the script from root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))
import math

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from adar import utils
from adar.datasets.dataset import MLAADFD_AR_Dataset, MLAADFDDataset
from adar.datasets.utils import HuggingFaceFeatureExtractor
from adar.models.mahalanobis import HierMahalanobisOODDetector, MahalanobisOODDetector
from adar.models.NSD import (
    EnergyOODDetector,
    HierNSDOODDetector,
    MSPOODDetector,
    NSDOODDetector,
)
from adar.models.w2v2_aasist import W2VAASIST, W2VAASIST_HArch, W2VAASIST_HShared


def scale(logits, s):
    n_classess = logits.size(1)
    if s == "auto":
        return logits * math.sqrt(2) * math.log(n_classess - 1)
    else:
        return logits * s


@torch.no_grad()
def run_flat_inference(model, models_args, batch, K, s, feature_extractor, pre_encoded=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, _, labels = batch

    if not pre_encoded:
        if models_args["is_segmented"]:
            n_segments = feat.shape[1]
            feat = feat.view(
                -1, models_args["sampling_rate"]
            )  # [batch_size * num_segments, sampling_rate]

        with torch.no_grad():
            feat = feature_extractor(feat, models_args["sampling_rate"]).float()

        if models_args["is_segmented"]:
            feat = feat.view(
                -1, n_segments, *feat.shape[1:]
            )  # [batch_size, num_segments, *embed_dims]
            feat = feat.mean(dim=1)  # [batch_size, *embed_dims] average over segments

    feat = feat.transpose(1, 2).to(device)

    feats, logits = model(feat)

    if models_args["use_sub_center_arc_margin"]:
        # Aggregate sub-center outputs
        if K > 1:
            logits = torch.reshape(logits, (-1, models_args["num_classes"], K))
            logits, _ = torch.max(logits, axis=2)
            logits = scale(logits, s)

    elif models_args["use_arc_margin"]:
        logits = scale(logits, s)

    return feats, logits, labels


@torch.no_grad()
def run_hier_inference(model, models_args, batch, K, s, feature_extractor, pre_encoded=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, _, labels = batch

    if not pre_encoded:
        if models_args["is_segmented"]:
            n_segments = feat.shape[1]
            feat = feat.view(
                -1, models_args["sampling_rate"]
            )  # [batch_size * num_segments, sampling_rate]

        with torch.no_grad():
            feat = feature_extractor(feat, models_args["sampling_rate"]).float()

        if models_args["is_segmented"]:
            feat = feat.view(
                -1, n_segments, *feat.shape[1:]
            )  # [batch_size, num_segments, *embed_dims]
            feat = feat.mean(dim=1)  # [batch_size, *embed_dims] average over segments

    feat = feat.transpose(1, 2).to(device)

    feats, logits = model(feat)
    sup_logits, sub_logits = logits

    if models_args["use_sub_center_arc_margin"]:
        bs = sup_logits.size(0)

        # Aggregate sub-center outputs
        if K > 1:
            sup_logits = torch.reshape(sup_logits, (bs, -1, K))
            sub_logits = torch.reshape(sub_logits, (bs, -1, K))

            sup_logits, _ = torch.max(sup_logits, axis=2)
            sub_logits, _ = torch.max(sub_logits, axis=2)

            sup_logits = scale(sup_logits, s)
            sub_logits = scale(sub_logits, s)

    elif models_args["use_arc_margin"]:
        sup_logits = scale(sup_logits, s)
        sub_logits = scale(sub_logits, s)

    logits = (sup_logits, sub_logits)

    return feats, logits, labels


def parse_args():
    parser = argparse.ArgumentParser("OOD Detector script")
    parser.add_argument("--ood_only", action="store_true", help="Run only the OOD detection")
    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/trained_models/base/anti-spoofing_feat_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "-d",
        "--path_to_dataset",
        type=str,
        default="data/prepared_ds/",
        help="Path to the dataset",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument(
        "--label_assignment_file",
        type=str,
        default="label_assignment.txt",
        help="Path to the file which lists the class assignments as written in the preprocessing step",
    )
    parser.add_argument(
        "--sup_label_assignment_file",
        type=str,
        default="label_assignment_superclass.txt",
        help="Path to the file which lists the superclass assignments",
    )
    parser.add_argument(
        "--superclass_lut_known",
        type=str,
        default="superclass_mapping_known.csv",
        help="File with known superclass mapping for hierarchical classification",
    )
    parser.add_argument(
        "--superclass_lut_full",
        type=str,
        default="superclass_mapping_test.csv",
        help="File with full superclass mapping for hierarchical classification",
    )
    # Hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=160, help="Hidden size dim")
    parser.add_argument(
        "--confidence_scaling",
        type=str,
        default="local",
        help="Confidence scaling type for H-Arch hierarchy",
        choices=["local", "sup", "none", "avg"],
    )
    parser.add_argument(
        "--ood_method",
        type=str,
        default="nsd",
        help="OOD detection method",
        choices=["nsd", "mahalanobis", "msp", "energy"],
    )

    args = parser.parse_args()
    if not os.path.exists(Path(args.model_path).parent / "ood"):
        os.makedirs(Path(args.model_path).parent / "ood")

    return args


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer_index = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return fpr[eer_index], thresholds[eer_index]


# Methods that return ID scores (higher = more in-distribution) must be negated
# so that the resulting OOD score convention is: higher score = more OOD.
_ID_SCORE_METHODS = {"nsd", "msp", "energy"}


def maybe_invert_scores_for_ood(scores, ood_method):
    if ood_method not in _ID_SCORE_METHODS:
        return scores
    if isinstance(scores, tuple):
        return tuple(-s for s in scores)
    return -scores


def main(args):  # noqa: C901
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path}")
    with open(Path(args.model_path).parent / "args.json") as f:
        model_args = json.load(f)
    hierarchy = model_args.get("hierarchy_type", None)
    pre_encoded = model_args.get("pre_encoded", False)

    if not args.ood_only:
        if not pre_encoded:
            feature_extractor = HuggingFaceFeatureExtractor(
                model_class_name=model_args.get("model_class", "Wav2Vec2Model"),
                layer=model_args.get("model_layer", 5),
                name=model_args.get("hugging_face_path", "facebook/wav2vec2-base"),
            )
            # Freeze the feature extractor
            for param in feature_extractor.model.parameters():
                param.requires_grad = False
            feature_extractor.model.eval()
        else:
            feature_extractor = None

    state_dict = torch.load(args.model_path, weights_only=True)
    s = state_dict.pop("arc_margin_scale", 1.0)
    K = state_dict.pop("arc_margin_k_centers", 1)

    if not args.ood_only:
        if hierarchy is None:
            id_map_known = None

            # Load flat model
            model = W2VAASIST(
                feature_dim=model_args["feat_dim"],
                num_labels=model_args["num_classes"] * K
                if (K > 1 and model_args["use_sub_center_arc_margin"])
                else model_args["num_classes"],
                normalize_before_output=True
                if (model_args["use_sub_center_arc_margin"] or model_args["use_arc_margin"])
                else False,
            )
        elif hierarchy == "H-Shared":
            id_map_known, _ = utils.load_superclass_mapping(
                args.path_to_dataset, args.superclass_lut_known
            )

            num_sup_classes = len(set(id_map_known.values()))  # Number of superclasses
            num_sub_classes = model_args["num_classes"]

            model = W2VAASIST_HShared(
                feature_dim=model_args["feat_dim"],
                num_suplabels=num_sup_classes,
                num_labels=num_sub_classes,
                normalize_before_output=True
                if (model_args["use_sub_center_arc_margin"] or model_args["use_arc_margin"])
                else False,
            )

        elif hierarchy == "H-Arch":
            id_map_known, _ = utils.load_superclass_mapping(
                args.path_to_dataset, args.superclass_lut_known
            )

            model = W2VAASIST_HArch(
                feature_dim=model_args["feat_dim"],
                label_mapping=id_map_known,
                normalize_before_output=True
                if (model_args["use_sub_center_arc_margin"] or model_args["use_arc_margin"])
                else False,
                K=K,
            )

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # ==================
        # Loading datasets
        if hierarchy is None:
            id_map_full = None
        else:
            id_map_full, _ = utils.load_superclass_mapping(
                args.path_to_dataset, args.superclass_lut_full
            )

        if pre_encoded:
            train_dataset = MLAADFDDataset(
                args.path_to_dataset,
                part="train",
                superclass_mapping=id_map_full,
            )
            dev_dataset = MLAADFDDataset(
                args.path_to_dataset,
                part="dev",
                superclass_mapping=id_map_full,
            )
            eval_dataset = MLAADFDDataset(
                args.path_to_dataset,
                part="eval",
                superclass_mapping=id_map_full,
            )
        else:
            train_dataset = MLAADFD_AR_Dataset(
                args.path_to_dataset,
                part="train",
                superclass_mapping=id_map_full,
                empasizer_pre_augmented=model_args["pre_augmented"],
                empasizer_sampling_rate=model_args["sampling_rate"],
                empasizer_musan_path=model_args["musan_path"],
                empasizer_rir_path=model_args["rir_path"],
                segmented=model_args["is_segmented"],
            )
            dev_dataset = MLAADFD_AR_Dataset(
                args.path_to_dataset,
                part="dev",
                superclass_mapping=id_map_full,
                empasizer_pre_augmented=model_args["pre_augmented"],
                empasizer_sampling_rate=model_args["sampling_rate"],
                empasizer_musan_path=model_args["musan_path"],
                empasizer_rir_path=model_args["rir_path"],
                segmented=model_args["is_segmented"],
            )
            eval_dataset = MLAADFD_AR_Dataset(
                args.path_to_dataset,
                part="eval",
                superclass_mapping=id_map_full,
                empasizer_pre_augmented=model_args["pre_augmented"],
                empasizer_sampling_rate=model_args["sampling_rate"],
                empasizer_musan_path=model_args["musan_path"],
                empasizer_rir_path=model_args["rir_path"],
                segmented=model_args["is_segmented"],
            )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

        # Extract logits and hidden states from trained model
        for subset_, loader in zip(
            ["train", "dev", "eval"], [train_loader, dev_loader, eval_loader], strict=False
        ):
            print(f"Running inference for {subset_}")
            num_samples = len(loader.dataset)
            all_feats = np.zeros((num_samples, args.hidden_dim))
            if hierarchy is None:
                all_logits = np.zeros((num_samples, model_args["num_classes"]))
                all_labels = np.zeros(num_samples)
            else:
                if hierarchy == "H-Shared":
                    all_logits = np.zeros((num_samples, num_sub_classes))
                    all_logits_sup = np.zeros((num_samples, num_sup_classes))
                    all_labels = np.zeros(num_samples)
                    all_labels_sup = np.zeros(num_samples)
                elif hierarchy == "H-Arch":
                    all_logits = np.zeros((num_samples, model.max_sublabels))
                    all_logits_sup = np.zeros((num_samples, len(model.label_hierarchy)))
                    all_labels = np.zeros(num_samples)
                    all_labels_sup = np.zeros(num_samples)

            for idx, batch in enumerate(tqdm(loader)):
                sample_num = idx * args.batch_size
                bs = batch[0].shape[0]
                if hierarchy is None:
                    hidden_state, logits, labels = run_flat_inference(
                        model,
                        model_args,
                        batch,
                        K,
                        s,
                        feature_extractor,
                        pre_encoded=pre_encoded,
                    )
                else:
                    hidden_state, logits, labels = run_hier_inference(
                        model,
                        model_args,
                        batch,
                        K,
                        s,
                        feature_extractor,
                        pre_encoded=pre_encoded,
                    )
                    sup_logits, logits = logits
                    sup_labels, labels = labels

                    all_logits_sup[sample_num : sample_num + bs] = sup_logits.detach().cpu().numpy()
                    all_labels_sup[sample_num : sample_num + bs] = sup_labels

                # Store all info
                all_feats[sample_num : sample_num + bs] = hidden_state.detach().cpu().numpy()
                all_logits[sample_num : sample_num + bs] = logits.detach().cpu().numpy()
                all_labels[sample_num : sample_num + bs] = labels

            # Save the info
            if hierarchy is None:
                if not args.ood_only:
                    out_path = Path(args.model_path).parent / "ood" / f"{subset_}_dict.npy"
                    np.save(
                        out_path,
                        {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                        },
                    )
                    print(f"Saved inference results to {out_path}")
                else:
                    if subset_ == "train":
                        train_dict = {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                        }
                    elif subset_ == "dev":
                        dev_dict = {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                        }
                    elif subset_ == "eval":
                        eval_dict = {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                        }
            else:
                if not args.ood_only:
                    out_path = Path(args.model_path).parent / "ood" / f"{subset_}_dict.npy"
                    np.save(
                        out_path,
                        {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                            "sup_logits": all_logits_sup,
                            "sup_labels": all_labels_sup,
                        },
                    )
                    print(f"Saved inference results to {out_path}")
                else:
                    if subset_ == "train":
                        train_dict = {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                            "sup_logits": all_logits_sup,
                            "sup_labels": all_labels_sup,
                        }
                    elif subset_ == "dev":
                        dev_dict = {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                            "sup_logits": all_logits_sup,
                            "sup_labels": all_labels_sup,
                        }
                    elif subset_ == "eval":
                        eval_dict = {
                            "feats": all_feats,
                            "logits": all_logits,
                            "labels": all_labels,
                            "sup_logits": all_logits_sup,
                            "sup_labels": all_labels_sup,
                        }

    train_dict = np.load(
        Path(args.model_path).parent / "ood" / "train_dict.npy", allow_pickle=True
    ).item()
    dev_dict = np.load(
        Path(args.model_path).parent / "ood" / "dev_dict.npy", allow_pickle=True
    ).item()
    eval_dict = np.load(
        Path(args.model_path).parent / "ood" / "eval_dict.npy", allow_pickle=True
    ).item()
    # ==================

    print("Setting up the OOD detector using the training data...")
    if hierarchy is None:
        if args.ood_method == "nsd":
            ood_detector = NSDOODDetector()
        elif args.ood_method == "mahalanobis":
            ood_detector = MahalanobisOODDetector()
        elif args.ood_method == "msp":
            ood_detector = MSPOODDetector()
        elif args.ood_method == "energy":
            ood_detector = EnergyOODDetector()

    else:
        if args.ood_method == "nsd":
            ood_detector = HierNSDOODDetector(
                hierarchy_type=hierarchy,
                confidence_scaling=args.confidence_scaling,
            )
        elif args.ood_method == "mahalanobis":
            ood_detector = HierMahalanobisOODDetector(
                hierarchy_type=hierarchy,
            )
        elif args.ood_method == "msp":
            ood_detector = MSPOODDetector()
        elif args.ood_method == "energy":
            ood_detector = EnergyOODDetector()

    ood_detector.setup(train_dict)

    # Get scores for OOD
    print("Getting OOD scores for the dev set...")
    dev_scores = maybe_invert_scores_for_ood(ood_detector.infer(dev_dict), args.ood_method)

    # Get the systems' labels assigned to OOD samples
    # Convert the system numbers into classes: OOD=1 and KNOWN=0

    with open(utils.resolve_label_path(args.path_to_dataset, args.label_assignment_file)) as f:
        OOD_classes = [
            int(line.split("|")[1]) for line in f.readlines() if line.strip().split("|")[2] == "OOD"
        ]
    dev_ood_labels = [
        1 if int(dev_dict["labels"][k]) in OOD_classes else 0
        for k in range(len(dev_dict["labels"]))
    ]

    if hierarchy is not None:
        with open(
            utils.resolve_label_path(args.path_to_dataset, args.sup_label_assignment_file)
        ) as f:
            OOD_classes_sup = [
                int(line.split("|")[1])
                for line in f.readlines()
                if line.strip().split("|")[2] == "OOD"
            ]
        dev_ood_labels_sup = [
            1 if int(dev_dict["sup_labels"][k]) in OOD_classes_sup else 0
            for k in range(len(dev_dict["sup_labels"]))
        ]

    # Compute a EER threshold over the dev scores
    print("\nComputing the EER threshold over dev set...")
    ood_summary = {"ood_method": args.ood_method}
    if hierarchy is None:
        eer, threshold = compute_eer(dev_ood_labels, dev_scores)
        print(f"DEV EER: {eer * 100:.2f}  | Threshold: {threshold:.2f}")
        ood_summary["dev_eer"] = float(eer)
        ood_summary["dev_threshold"] = float(threshold)
        with open(
            Path(args.model_path).parent / "ood" / f"OOD_dev_threshold_{args.ood_method}.txt",
            "w",
        ) as f:
            f.write(f"EER: {eer * 100:.2f}  | Threshold: {threshold:.2f}")
    else:
        dev_scores_sup, dev_scores = dev_scores
        eer, threshold = compute_eer(dev_ood_labels, dev_scores)
        print(f"GLOBAL DEV EER: {eer * 100:.2f}  | Threshold: {threshold:.2f}")
        ood_summary["dev_eer"] = float(eer)
        ood_summary["dev_threshold"] = float(threshold)
        th_file_path = (
            Path(args.model_path).parent
            / "ood"
            / f"OOD_dev_threshold_{ood_detector.confidence_scaling}_scaling.txt"
            if args.ood_method == "nsd"
            else Path(args.model_path).parent / "ood" / f"OOD_dev_threshold_{args.ood_method}.txt"
        )
        with open(th_file_path, "w") as f:
            f.write(f"EER: {eer * 100:.2f}  | Threshold: {threshold:.2f}")

        eer_sup, threshold_sup = compute_eer(dev_ood_labels_sup, dev_scores_sup)
        ood_summary["dev_eer_sup"] = float(eer_sup)
        ood_summary["dev_threshold_sup"] = float(threshold_sup)
        th_sup_file_path = (
            Path(args.model_path).parent / "ood" / "OOD_dev_threshold_sup.txt"
            if args.ood_method == "nsd"
            else Path(args.model_path).parent
            / "ood"
            / f"OOD_dev_threshold_sup_{args.ood_method}.txt"
        )

        print(f"SUPERCLASS DEV EER: {eer_sup * 100:.2f}  | Threshold: {threshold_sup:.2f}")
        with open(th_sup_file_path, "w") as f:
            f.write(f"EER: {eer_sup * 100:.2f}  | Threshold: {threshold_sup:.2f}")

    # Set the threshold and compute the OOD accuracy over the eval set
    print("\nComputing eval results using dev threshold...")
    print("Class 1 is OOD, Class 0 is ID")
    eval_scores = maybe_invert_scores_for_ood(ood_detector.infer(eval_dict), args.ood_method)

    eval_ood_labels = [
        1 if int(eval_dict["labels"][k]) in OOD_classes else 0
        for k in range(len(eval_dict["labels"]))
    ]

    if hierarchy is not None:
        eval_ood_labels_sup = [
            1 if int(eval_dict["sup_labels"][k]) in OOD_classes_sup else 0
            for k in range(len(eval_dict["sup_labels"]))
        ]

    if hierarchy is None:
        eval_eer, _ = compute_eer(eval_ood_labels, eval_scores)
        print(f"Eval EER: {eval_eer * 100:.2f}")
        ood_summary["eval_eer"] = float(eval_eer)

        predicts = [1 if eval_scores[k] > threshold else 0 for k in range(len(eval_dict["labels"]))]
        print("OOD classification report for eval data:")
        report = classification_report(eval_ood_labels, predicts)
        report_path = (
            Path(args.model_path).parent / "ood" / f"OOD_eval_results_{args.ood_method}.txt"
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"... also written to {report_path}")

    else:
        eval_scores_sup, eval_scores = eval_scores

        eval_eer, _ = compute_eer(eval_ood_labels, eval_scores)
        print(f"Eval GLOBAL EER: {eval_eer * 100:.2f}")
        ood_summary["eval_eer"] = float(eval_eer)

        predicts = [1 if eval_scores[k] > threshold else 0 for k in range(len(eval_dict["labels"]))]
        print("OOD GLOBAL classification report for eval data:")
        report = classification_report(eval_ood_labels, predicts)
        report_path = (
            Path(args.model_path).parent
            / "ood"
            / f"OOD_eval_results_{ood_detector.confidence_scaling}_scaling.txt"
            if args.ood_method == "nsd"
            else Path(args.model_path).parent / "ood" / f"OOD_eval_results_{args.ood_method}.txt"
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"... also written to {report_path}")

        eval_eer_sup, _ = compute_eer(eval_ood_labels_sup, eval_scores_sup)
        print(f"Eval SUPERCLASS EER: {eval_eer_sup * 100:.2f}")
        ood_summary["eval_eer_sup"] = float(eval_eer_sup)

        predicts_sup = [
            1 if eval_scores_sup[k] > threshold_sup else 0
            for k in range(len(eval_dict["sup_labels"]))
        ]
        print("OOD SUPERCLASS classification report for eval data:")
        report_sup = classification_report(eval_ood_labels_sup, predicts_sup)
        report_path_sup = (
            Path(args.model_path).parent / "ood" / "OOD_eval_results_sup.txt"
            if args.ood_method == "nsd"
            else Path(args.model_path).parent
            / "ood"
            / f"OOD_eval_results_sup_{args.ood_method}.txt"
        )
        with open(report_path_sup, "w") as f:
            f.write(report_sup)
        print(report_sup)
        print(f"... also written to {report_path_sup}")

    summary_suffix = (
        f"_{ood_detector.confidence_scaling}_scaling"
        if (hierarchy is not None and args.ood_method == "nsd")
        else ""
    )
    summary_path = (
        Path(args.model_path).parent / "ood" / f"OOD_summary_{args.ood_method}{summary_suffix}.json"
    )

    with open(summary_path, "w") as f:
        json.dump(ood_summary, f, indent=2)
    print(f"Saved OOD summary to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
