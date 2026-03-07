#!/usr/bin/env python3
"""
Minimal tracking pipeline: Deformable DETR + warm-start + ByteTrack.

Outputs:
  - JSON per video (detections + tracks)
  - MOT format per video
  - Visualization frames (optional)
  - FPS summary
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
import yaml

from models.deformable_detr_model import build_deformable_detr_model, post_process_deformable_detr
from utils.image_processor import build_local_image_processor
from utils.bytetrack_wrapper import ByteTrackConfig, ByteTrackWrapper


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned[key[7:]] = value
        else:
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")


def _color_for_id(track_id: int) -> tuple:
    return (
        (track_id * 37) % 255,
        (track_id * 17) % 255,
        (track_id * 29) % 255,
    )


def _draw_tracks(image: Image.Image, tracks: List[Dict], font: ImageFont.ImageFont) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for trk in tracks:
        x1, y1, x2, y2 = trk["bbox"]
        track_id = trk.get("track_id", -1)
        color = _color_for_id(track_id)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"id {track_id}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([x1, y1 - (bbox[3] - bbox[1]) - 4, x1 + (bbox[2] - bbox[0]) + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - (bbox[3] - bbox[1]) - 2), label, fill="white", font=font)
    return image


def _select_warm_ref(detections: List[Dict], image_size: tuple, k: int, tau: float, device: torch.device):
    if not detections:
        return None
    width, height = image_size
    candidates = [d for d in detections if d["score"] >= tau]
    candidates.sort(key=lambda d: d["score"], reverse=True)
    if not candidates:
        return None
    points = []
    for det in candidates[:k]:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) * 0.5 / max(1.0, float(width))
        cy = (y1 + y2) * 0.5 / max(1.0, float(height))
        points.append([cx, cy])
    if not points:
        return None
    return torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)


def _detections_from_results(results: Dict, score_thresh: float) -> List[Dict]:
    scores = results["scores"].detach().cpu().tolist()
    labels = results["labels"].detach().cpu().tolist()
    boxes = results["boxes"].detach().cpu().tolist()
    detections: List[Dict] = []
    for score, label, box in zip(scores, labels, boxes):
        if score < score_thresh:
            continue
        detections.append({"bbox": box, "score": float(score), "cls": int(label)})
    return detections


def _write_mot(video_id: str, frames: List[Dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id}.txt"
    lines: List[str] = []
    for frame in frames:
        frame_id = frame["frame_id"]
        for trk in frame["tracks"]:
            x1, y1, x2, y2 = trk["bbox"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            line = [
                str(int(frame_id)),
                str(int(trk.get("track_id", -1))),
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{w:.2f}",
                f"{h:.2f}",
                f"{float(trk.get('score', 1.0)):.4f}",
                str(int(trk.get("cls", -1))),
                "-1",
            ]
            lines.append(",".join(line))
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path.")
    parser.add_argument("--images-root", required=True, help="Tracking images root (val).")
    parser.add_argument("--seq-list", default=None, help="Sequence list JSON from selector.")
    parser.add_argument("--video-ids", nargs="*", default=None, help="Override video id list.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--warm-k", type=int, default=30)
    parser.add_argument("--warm-tau", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()

    images_root = Path(args.images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_json_dir = output_dir / "pred_json"
    pred_mot_dir = output_dir / "pred_mot"
    vis_dir = output_dir / "vis"
    pred_json_dir.mkdir(parents=True, exist_ok=True)
    pred_mot_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    if args.font_path and Path(args.font_path).exists():
        font = ImageFont.truetype(args.font_path, 18)
    else:
        font = ImageFont.load_default()

    config = _load_config(args.config)
    model = build_deformable_detr_model(config)
    _load_checkpoint(model, args.checkpoint)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    processor = build_local_image_processor(config)

    tracker = ByteTrackWrapper(ByteTrackConfig(), per_class=args.per_class)

    if args.video_ids:
        video_ids = args.video_ids
    elif args.seq_list:
        seq_data = json.loads(Path(args.seq_list).read_text(encoding="utf-8"))
        video_ids = [item["video_id"] for item in seq_data.get("selected", [])]
    else:
        video_ids = sorted([p.name for p in images_root.iterdir() if p.is_dir()])

    fps_records: List[Dict] = []
    for video_id in video_ids:
        frames_dir = images_root / video_id
        if not frames_dir.exists():
            print(f"[WARN] Missing frames dir: {frames_dir}")
            continue

        frame_paths = sorted([p for p in frames_dir.glob("*.jpg")])
        if not frame_paths:
            frame_paths = sorted([p for p in frames_dir.glob("*.png")])
        if not frame_paths:
            print(f"[WARN] No frames in {frames_dir}")
            continue
        if args.max_frames and args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]

        tracker.reset()
        frames_json: List[Dict] = []
        seq_vis_dir = vis_dir / video_id if args.save_vis else None
        if seq_vis_dir is not None:
            seq_vis_dir.mkdir(parents=True, exist_ok=True)
        prev_ref = None
        start_time = time.perf_counter()

        for frame_index, frame_path in enumerate(frame_paths, start=1):
            image = Image.open(frame_path).convert("RGB")
            width, height = image.size
            sample = processor(image)["pixel_values"].squeeze(0).to(device)

            with torch.no_grad():
                outputs = model([sample], warm_ref_points=prev_ref if args.warm_start else None)
                results = post_process_deformable_detr(
                    outputs,
                    target_sizes=torch.tensor([[height, width]], device=device),
                    threshold=None,
                )[0]

            detections = _detections_from_results(results, args.score_thresh)
            prev_ref = _select_warm_ref(detections, (width, height), args.warm_k, args.warm_tau, device)

            tracks = tracker.update(detections, img_size=(height, width))
            frame_entry = {
                "frame_id": frame_index,
                "image_file": frame_path.name,
                "detections": detections,
                "tracks": tracks,
            }
            frames_json.append(frame_entry)

            if args.save_vis and seq_vis_dir is not None:
                vis_img = _draw_tracks(image.copy(), tracks, font)
                vis_img.save(seq_vis_dir / frame_path.name)

        elapsed = time.perf_counter() - start_time
        fps = len(frame_paths) / elapsed if elapsed > 0 else 0.0
        fps_records.append({"video_id": video_id, "fps": fps, "frames": len(frame_paths)})

        json_path = pred_json_dir / f"{video_id}.json"
        json_path.write_text(json.dumps(frames_json, ensure_ascii=True, indent=2), encoding="utf-8")
        _write_mot(video_id, frames_json, pred_mot_dir)
        print(f"[OK] {video_id}: {len(frame_paths)} frames, fps={fps:.2f}")

    avg_fps = sum(item["fps"] for item in fps_records) / max(1, len(fps_records))
    fps_summary = {"sequences": fps_records, "avg_fps": avg_fps}
    (output_dir / "fps.json").write_text(json.dumps(fps_summary, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
