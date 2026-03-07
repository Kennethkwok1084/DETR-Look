#!/usr/bin/env python3
"""
Build a diagnostic metrics table image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_overall(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data.get("overall", {})
    return {
        "FlickerRate": overall.get("FlickerRate", 0.0),
        "AvgTrackLength": overall.get("AvgTrackLength", 0.0),
        "ReacquisitionCount": overall.get("ReacquisitionCount", 0.0),
        "FPS": overall.get("FPS", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline diag metrics JSON.")
    parser.add_argument("--warm", required=True, help="Warm-start diag metrics JSON.")
    parser.add_argument("--output", required=True, help="Output table image path.")
    parser.add_argument("--title", default="Diagnostic Stability Metrics")
    args = parser.parse_args()

    baseline = _load_overall(Path(args.baseline))
    warm = _load_overall(Path(args.warm))

    columns = ["Method", "FlickerRate", "AvgTrackLength", "ReacqCount", "FPS"]
    rows = [
        [
            "Baseline",
            f"{baseline['FlickerRate']:.4f}",
            f"{baseline['AvgTrackLength']:.2f}",
            f"{baseline['ReacquisitionCount']:.0f}",
            f"{baseline['FPS']:.2f}",
        ],
        [
            "Warm-start",
            f"{warm['FlickerRate']:.4f}",
            f"{warm['AvgTrackLength']:.2f}",
            f"{warm['ReacquisitionCount']:.0f}",
            f"{warm['FPS']:.2f}",
        ],
    ]

    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.4)
    ax.set_title(args.title, fontsize=12, pad=8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[OK] table -> {output_path}")


if __name__ == "__main__":
    main()
