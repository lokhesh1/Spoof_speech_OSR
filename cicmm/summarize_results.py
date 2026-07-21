#!/usr/bin/env python3
"""Print unknown-class F1 (and MacroF1) side by side across all trial results.

Scans ``results/*.json`` (written by ``eval.py``, one file per artifacts
dir / trial) and prints a sorted table. Run from ``cicmm/``::

    python summarize_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args(argv)

    files = sorted(Path(args.results_dir).glob("*.json"))
    if not files:
        print(f"No result files found in {args.results_dir}/")
        return 1

    rows = []
    for f in files:
        r = json.loads(f.read_text())
        cls = r.get("classification", {})
        det = r.get("detection", {})
        rows.append((
            f.stem,
            cls.get("f1_unknown", float("nan")),
            cls.get("macro_f1", float("nan")),
            cls.get("known_top1", float("nan")),
            cls.get("unknown_reject_rate", float("nan")),
            det.get("auroc", float("nan")),
            det.get("eer", float("nan")),
        ))

    rows.sort(key=lambda r: -r[1])  # rank by F1(unknown) descending

    name_w = max(len(r[0]) for r in rows) + 2
    header = (f"{'trial':<{name_w}}{'F1(unknown)':>13}{'MacroF1':>10}"
              f"{'Known-top1':>12}{'Unk-reject':>12}{'Det-AUROC':>11}{'EER':>9}")
    print(header)
    print("-" * len(header))
    for name, f1u, macro, known, reject, auroc, eer in rows:
        print(f"{name:<{name_w}}{f1u:>13.4f}{macro:>10.4f}"
              f"{known:>12.4f}{reject:>12.4f}{auroc:>11.4f}{eer:>9.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
