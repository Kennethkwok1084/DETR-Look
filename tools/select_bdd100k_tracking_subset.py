#!/usr/bin/env python3
"""
Select a minimal BDD100K tracking subset from label files.

The script scans per-video JSON label files and ranks sequences by
frame count and object density. It outputs a JSON list for reuse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_objects(frame: Dict) -> Iterable[Dict]:
    # Tracking labels usually store objects under "objects".
    # Keep a fallback for "labels" in case of variant formats.
    objects = frame.get("objects")
    if objects is None:
        objects = frame.get("labels", [])
    return objects or []


def _extract_stats(path: Path, classes: Optional[set]) -> Dict:
    data = _load_json(path)
    video_id = data.get("name") or path.stem
    frames = data.get("frames", [])
    frame_count = len(frames)

    total_objects = 0
    target_objects = 0
    for frame in frames:
        for obj in _iter_objects(frame):
            box = obj.get("box2d")
            if not box:
                continue
            total_objects += 1
            if classes is None:
                target_objects += 1
            else:
                category = (obj.get("category") or "").strip().lower()
                if category in classes:
                    target_objects += 1

    return {
        "video_id": video_id,
        "frames": frame_count,
        "objects_total": total_objects,
        "objects_target": target_objects,
    }


def _sort_key(item: Dict) -> tuple:
    return (item["objects_target"], item["frames"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Path to BDD100K tracking labels (per-video JSON).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of sequences to select.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Minimum number of frames per sequence.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000000,
        help="Maximum number of frames per sequence.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Filter by category names (e.g., car truck bus).",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    classes = None
    if args.classes:
        classes = {c.strip().lower() for c in args.classes if c.strip()}

    stats: List[Dict] = []
    for path in sorted(labels_dir.glob("*.json")):
        item = _extract_stats(path, classes)
        if not (args.min_frames <= item["frames"] <= args.max_frames):
            continue
        stats.append(item)

    stats.sort(key=_sort_key, reverse=True)
    selected = stats[: max(0, args.top_k)]

    output = {
        "labels_dir": str(labels_dir),
        "classes": sorted(classes) if classes else None,
        "selected": selected,
        "total_candidates": len(stats),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[OK] Selected {len(selected)} sequences -> {output_path}")


if __name__ == "__main__":
    main()
