#!/usr/bin/env python3
"""
Compute diagnostic tracking metrics without GT.

Metrics:
  - FlickerRate: total reappearance gaps / total track frames
  - AvgTrackLength: average frames per track id
  - ReacquisitionCount: total number of gaps per track id
  - FPS: optional from fps.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_tracks(frames: List[Dict]) -> Dict[int, List[int]]:
    tracks: Dict[int, List[int]] = {}
    for frame in frames:
        frame_id = int(frame.get("frame_id", 0))
        for trk in frame.get("tracks", []):
            track_id = trk.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            tracks.setdefault(track_id, []).append(frame_id)
    return tracks


def _compute_from_tracks(tracks: Dict[int, List[int]]) -> Dict[str, float]:
    total_frames = 0
    total_gaps = 0
    track_lengths: List[int] = []
    for frame_ids in tracks.values():
        if not frame_ids:
            continue
        frame_ids = sorted(set(frame_ids))
        track_lengths.append(len(frame_ids))
        total_frames += len(frame_ids)
        for i in range(1, len(frame_ids)):
            if frame_ids[i] - frame_ids[i - 1] > 1:
                total_gaps += 1
    num_tracks = len(track_lengths)
    avg_track_len = sum(track_lengths) / max(1, num_tracks)
    flicker_rate = total_gaps / max(1, total_frames)
    return {
        "FlickerRate": float(flicker_rate),
        "AvgTrackLength": float(avg_track_len),
        "ReacquisitionCount": float(total_gaps),
        "NumTracks": float(num_tracks),
        "TotalTrackFrames": float(total_frames),
    }


def _load_fps(fps_json: Optional[str]) -> Optional[float]:
    if not fps_json:
        return None
    path = Path(fps_json)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("avg_fps", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json-dir", required=True, help="Directory with per-video JSON outputs.")
    parser.add_argument("--output", required=True, help="Output metrics JSON path.")
    parser.add_argument("--fps-json", default=None, help="Optional FPS summary JSON.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_json_dir)
    all_tracks: Dict[int, List[int]] = {}
    per_video: Dict[str, Dict[str, float]] = {}

    for json_path in sorted(pred_dir.glob("*.json")):
        frames = _load_json(json_path)
        tracks = _collect_tracks(frames)
        stats = _compute_from_tracks(tracks)
        per_video[json_path.stem] = stats
        for track_id, frames_list in tracks.items():
            all_tracks.setdefault(track_id, []).extend(frames_list)

    overall = _compute_from_tracks(all_tracks)
    fps = _load_fps(args.fps_json)
    if fps is not None:
        overall["FPS"] = fps

    output = {"overall": overall, "per_video": per_video}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[OK] diagnostic metrics -> {output_path}")


if __name__ == "__main__":
    main()
