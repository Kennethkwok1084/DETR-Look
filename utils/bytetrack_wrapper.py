#!/usr/bin/env python3
"""
Minimal ByteTrack wrapper.

Expected input detections: list of dicts or tuples
  - dict: {"bbox": [x1,y1,x2,y2], "score": float, "cls": int}
  - tuple: (x1, y1, x2, y2, score, cls)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ByteTrackConfig:
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    frame_rate: int = 30
    mot20: bool = False


def _resolve_bytetrack(bypath: Optional[str] = None):
    if bypath:
        sys.path.insert(0, bypath)
    else:
        local_path = Path(__file__).resolve().parents[1] / "third_party" / "bytetrack"
        if local_path.exists():
            sys.path.insert(0, str(local_path))

    try:
        from yolox.tracker.byte_tracker import BYTETracker  # type: ignore
        return BYTETracker
    except Exception as exc:
        raise ImportError(
            "ByteTrack not found. Vendor it under third_party/bytetrack or install it."
        ) from exc


class ByteTrackWrapper:
    def __init__(self, config: Optional[ByteTrackConfig] = None, per_class: bool = False, bytetrack_path: Optional[str] = None):
        self.config = config or ByteTrackConfig()
        self.per_class = per_class
        self._tracker_cls = _resolve_bytetrack(bytetrack_path)
        self._trackers: Dict[int, Any] = {}

    def _build_tracker(self):
        class Args:
            pass

        args = Args()
        args.track_thresh = self.config.track_thresh
        args.match_thresh = self.config.match_thresh
        args.track_buffer = self.config.track_buffer
        args.mot20 = self.config.mot20
        return self._tracker_cls(args, self.config.frame_rate)

    def _get_tracker(self, cls_id: int):
        if cls_id not in self._trackers:
            self._trackers[cls_id] = self._build_tracker()
        return self._trackers[cls_id]

    @staticmethod
    def _normalize_dets(detections: Sequence) -> List[Dict]:
        normalized: List[Dict] = []
        for det in detections:
            if isinstance(det, dict):
                bbox = det.get("bbox")
                score = det.get("score")
                cls_id = det.get("cls", -1)
            else:
                if len(det) < 6:
                    raise ValueError("tuple detections must be (x1,y1,x2,y2,score,cls)")
                bbox = list(det[:4])
                score = float(det[4])
                cls_id = int(det[5])
            if bbox is None or score is None:
                continue
            normalized.append({"bbox": bbox, "score": float(score), "cls": int(cls_id)})
        return normalized

    @staticmethod
    def _to_numpy(dets: List[Dict]):
        import numpy as np

        if not dets:
            return np.zeros((0, 5), dtype=float)
        arr = []
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            arr.append([x1, y1, x2, y2, det["score"]])
        return np.asarray(arr, dtype=float)

    def update(self, detections: Sequence, img_size: Sequence[int]) -> List[Dict]:
        dets = self._normalize_dets(detections)
        if not self.per_class:
            return self._update_tracker(self._get_tracker(-1), dets, img_size, cls_id=-1)

        results: List[Dict] = []
        by_class: Dict[int, List[Dict]] = {}
        for det in dets:
            by_class.setdefault(det["cls"], []).append(det)
        for cls_id, group in by_class.items():
            tracker = self._get_tracker(cls_id)
            results.extend(self._update_tracker(tracker, group, img_size, cls_id=cls_id))
        return results

    def _update_tracker(self, tracker: Any, dets: List[Dict], img_size: Sequence[int], cls_id: int) -> List[Dict]:
        det_arr = self._to_numpy(dets)
        tracks = tracker.update(det_arr, img_size, img_size)
        out: List[Dict] = []
        for t in tracks:
            tlwh = t.tlwh if hasattr(t, "tlwh") else None
            if tlwh is None:
                continue
            x, y, w, h = tlwh
            out.append(
                {
                    "track_id": int(getattr(t, "track_id", -1)),
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "score": float(getattr(t, "score", 1.0)),
                    "cls": int(cls_id),
                }
            )
        return out

    def reset(self) -> None:
        self._trackers.clear()
