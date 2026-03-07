#!/usr/bin/env python3
"""
Evaluate MOT metrics (IDSW/Frag/IDF1) with motmetrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


import numpy as np


if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
def _load_fps(fps_json: Optional[str]) -> Optional[float]:
    if not fps_json:
        return None
    path = Path(fps_json)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("avg_fps", 0.0))


def _filter_class(df, class_id: Optional[int]):
    if class_id is None:
        return df
    if "ClassId" not in df.columns:
        return df
    return df[df["ClassId"] == class_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True, help="Ground truth MOT files.")
    parser.add_argument("--pred-dir", required=True, help="Prediction MOT files.")
    parser.add_argument("--output", required=True, help="Output JSON metrics path.")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--fps-json", default=None, help="FPS summary JSON from run.")
    args = parser.parse_args()

    try:
        import motmetrics as mm
    except Exception as exc:
        raise RuntimeError("motmetrics not installed. Try: pip install motmetrics") from exc

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    accs = []
    names: List[str] = []
    for pred_file in sorted(pred_dir.glob("*.txt")):
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            print(f"[WARN] Missing GT for {pred_file.name}")
            continue
        gt = mm.io.loadtxt(gt_file, fmt="mot16")
        pred = mm.io.loadtxt(pred_file, fmt="mot16")
        gt = _filter_class(gt, args.class_id)
        pred = _filter_class(pred, args.class_id)

        acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=args.iou_thresh)
        accs.append(acc)
        names.append(pred_file.stem)

    if not accs:
        raise RuntimeError("No sequences evaluated. Check gt/pred directories.")

    mh = mm.metrics.create()
    metrics = ["num_switches", "num_fragmentations", "idf1"]
    summary = mh.compute_many(accs, metrics=metrics, names=names, generate_overall=True)

    overall = summary.loc["OVERALL"]
    result = {
        "IDSW": float(overall["num_switches"]),
        "Frag": float(overall["num_fragmentations"]),
        "IDF1": float(overall["idf1"]),
    }

    fps = _load_fps(args.fps_json)
    if fps is not None:
        result["FPS"] = fps

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[OK] metrics -> {output_path}")


if __name__ == "__main__":
    main()
