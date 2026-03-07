#!/usr/bin/env python3
"""
Build an ablation table image from two metrics JSON files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _load_metrics(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "IDSW": data.get("IDSW", 0.0),
        "Frag": data.get("Frag", 0.0),
        "FPS": data.get("FPS", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON.")
    parser.add_argument("--warm", required=True, help="Warm-start metrics JSON.")
    parser.add_argument("--output", required=True, help="Output table image path.")
    parser.add_argument("--title", default="Ablation (IDSW/Frag/FPS)")
    args = parser.parse_args()

    baseline = _load_metrics(Path(args.baseline))
    warm = _load_metrics(Path(args.warm))

    columns = ["Method", "IDSW", "Frag", "FPS"]
    rows = [
        ["Baseline", f"{baseline['IDSW']:.2f}", f"{baseline['Frag']:.2f}", f"{baseline['FPS']:.2f}"],
        ["Warm-start", f"{warm['IDSW']:.2f}", f"{warm['Frag']:.2f}", f"{warm['FPS']:.2f}"],
    ]

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    ax.set_title(args.title, fontsize=12, pad=8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[OK] table -> {output_path}")


if __name__ == "__main__":
    main()
