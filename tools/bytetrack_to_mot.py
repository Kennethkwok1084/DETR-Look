#!/usr/bin/env python3
"""
Convert ByteTrack results to MOT format.

Expected per-video JSON format (recommended):
[
  {
    "frame_id": 1,
    "tracks": [
      {"track_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.9, "cls": 2}
    ]
  }
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_frames(data) -> Iterable[Dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "frames" in data and isinstance(data["frames"], list):
            return data["frames"]
    raise ValueError("Unsupported JSON format for tracking results.")


def _read_bbox(track: Dict, bbox_format: str) -> Optional[List[float]]:
    box = track.get("bbox") or track.get("box") or track.get("tlbr") or track.get("tlwh")
    if not box or len(box) != 4:
        return None
    x, y, w, h = box
    if bbox_format == "tlbr":
        x1, y1, x2, y2 = box
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return [x1, y1, w, h]
    if bbox_format == "tlwh":
        return [float(x), float(y), float(w), float(h)]
    raise ValueError(f"Unknown bbox_format: {bbox_format}")


def _convert_file(input_path: Path, output_dir: Path, bbox_format: str, class_map: Dict[str, int], default_class: int) -> None:
    data = _load_json(input_path)
    frames = _iter_frames(data)

    video_id = input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.txt"

    lines: List[str] = []
    for frame_index, frame in enumerate(frames, start=1):
        frame_id = frame.get("frame_id") or frame.get("frame_index") or frame.get("frame") or frame_index
        tracks = frame.get("tracks") or frame.get("objects") or frame.get("dets") or []
        for track in tracks:
            track_id = track.get("track_id") or track.get("id")
            if track_id is None:
                continue
            xywh = _read_bbox(track, bbox_format)
            if xywh is None:
                continue
            score = track.get("score")
            if score is None:
                score = track.get("conf", 1.0)
            cls = track.get("cls") or track.get("class") or track.get("label")
            if isinstance(cls, str):
                class_id = class_map.get(cls.lower(), default_class)
            elif cls is None:
                class_id = default_class
            else:
                class_id = int(cls)
            line = [
                str(int(frame_id)),
                str(int(track_id)),
                f"{xywh[0]:.2f}",
                f"{xywh[1]:.2f}",
                f"{xywh[2]:.2f}",
                f"{xywh[3]:.2f}",
                f"{float(score):.4f}",
                str(class_id),
                "-1",
            ]
            lines.append(",".join(line))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {video_id} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Single JSON file.")
    parser.add_argument("--input-dir", help="Directory of JSON files.")
    parser.add_argument("--output-dir", required=True, help="Output dir for MOT files.")
    parser.add_argument("--bbox-format", choices=["tlbr", "tlwh"], default="tlbr")
    parser.add_argument("--class-map", default=None, help="Optional JSON mapping: name -> id.")
    parser.add_argument("--default-class", type=int, default=-1)
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        raise ValueError("Either --input or --input-dir is required.")

    class_map: Dict[str, int] = {}
    if args.class_map:
        data = json.loads(Path(args.class_map).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            class_map = {str(k).lower(): int(v) for k, v in data.items()}

    output_dir = Path(args.output_dir)
    if args.input:
        _convert_file(Path(args.input), output_dir, args.bbox_format, class_map, args.default_class)
        return

    input_dir = Path(args.input_dir)
    for input_path in sorted(input_dir.glob("*.json")):
        _convert_file(input_path, output_dir, args.bbox_format, class_map, args.default_class)


if __name__ == "__main__":
    main()
