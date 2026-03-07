#!/usr/bin/env python3
"""
Convert BDD100K tracking labels (per-video JSON) to MOT format.
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
    objects = frame.get("objects")
    if objects is None:
        objects = frame.get("labels", [])
    return objects or []


def _load_class_map(path: Optional[str], classes: Optional[List[str]]) -> Dict[str, int]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k).lower(): int(v) for k, v in data.items()}
        if isinstance(data, list):
            return {str(name).lower(): idx for idx, name in enumerate(data, start=1)}
        raise ValueError("class map must be a dict or list")
    if classes:
        return {name.lower(): idx for idx, name in enumerate(classes, start=1)}
    return {}


def _bbox_xyxy_to_xywh(box: Dict) -> Optional[List[float]]:
    if not box:
        return None
    x1 = float(box.get("x1", 0.0))
    y1 = float(box.get("y1", 0.0))
    x2 = float(box.get("x2", 0.0))
    y2 = float(box.get("y2", 0.0))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return [x1, y1, w, h]


def _normalize_frames(data, label_path: Path) -> tuple[str, List[Dict]]:
    if isinstance(data, dict):
        video_id = data.get("name") or label_path.stem
        frames = data.get("frames", [])
        return video_id, frames
    if isinstance(data, list):
        if data:
            video_id = data[0].get("videoName") or label_path.stem
        else:
            video_id = label_path.stem
        return video_id, data
    raise ValueError("Unsupported label JSON format.")


def _convert_video(label_path: Path, output_dir: Path, class_map: Dict[str, int], classes: Optional[set]) -> None:
    data = _load_json(label_path)
    video_id, frames = _normalize_frames(data, label_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.txt"

    lines: List[str] = []
    for frame_offset, frame in enumerate(frames):
        frame_index = frame.get("frameIndex")
        if frame_index is None:
            frame_id = frame_offset + 1
        else:
            frame_id = int(frame_index) + 1
        for obj in _iter_objects(frame):
            box = obj.get("box2d")
            if not box:
                continue
            category = (obj.get("category") or "").strip().lower()
            if classes is not None and category not in classes:
                continue
            track_id = obj.get("id")
            if track_id is None:
                continue
            xywh = _bbox_xyxy_to_xywh(box)
            if xywh is None:
                continue
            class_id = class_map.get(category, -1)
            line = [
                str(frame_id),
                str(int(track_id)),
                f"{xywh[0]:.2f}",
                f"{xywh[1]:.2f}",
                f"{xywh[2]:.2f}",
                f"{xywh[3]:.2f}",
                "1",  # conf
                str(class_id),
                "-1",  # visibility (unused)
            ]
            lines.append(",".join(line))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {video_id} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-dir", required=True, help="BDD100K tracking labels dir (JSON files).")
    parser.add_argument("--output-dir", required=True, help="Output dir for MOT files.")
    parser.add_argument("--classes", nargs="*", default=None, help="Filter by category names.")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping: category -> id.")
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels dir not found: {labels_dir}")

    classes = {c.strip().lower() for c in args.classes} if args.classes else None
    class_map = _load_class_map(args.class_map, args.classes)

    output_dir = Path(args.output_dir)
    for label_path in sorted(labels_dir.glob("*.json")):
        _convert_video(label_path, output_dir, class_map, classes)


if __name__ == "__main__":
    main()
