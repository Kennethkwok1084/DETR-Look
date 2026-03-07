#!/usr/bin/env python3
"""
Minimal tracking pipeline using HuggingFace Deformable DETR (no custom ops).

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
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deformable_detr.modeling_deformable_detr import inverse_sigmoid

from utils.bytetrack_wrapper import ByteTrackConfig, ByteTrackWrapper


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


def _detections_from_results(results: Dict) -> List[Dict]:
    scores = results["scores"].detach().cpu().tolist()
    labels = results["labels"].detach().cpu().tolist()
    boxes = results["boxes"].detach().cpu().tolist()
    detections: List[Dict] = []
    for score, label, box in zip(scores, labels, boxes):
        detections.append({"bbox": box, "score": float(score), "cls": int(label)})
    return detections


def _apply_warm_ref(reference_points, warm_ref_points, eps=1e-4):
    if warm_ref_points is None:
        return reference_points
    if warm_ref_points.dim() == 2:
        warm_ref_points = warm_ref_points.unsqueeze(0)
    if warm_ref_points.dim() != 3:
        return reference_points
    if warm_ref_points.size(0) == 1 and reference_points.size(0) > 1:
        warm_ref_points = warm_ref_points.expand(reference_points.size(0), -1, -1)
    if warm_ref_points.size(0) != reference_points.size(0):
        return reference_points
    warm_ref_points = warm_ref_points.to(device=reference_points.device, dtype=reference_points.dtype)
    k = min(reference_points.size(1), warm_ref_points.size(1))
    if k <= 0:
        return reference_points
    reference_points = reference_points.clone()
    reference_points[:, :k, :2] = warm_ref_points[:, :k, :2].clamp(eps, 1 - eps)
    return reference_points


def _forward_with_warm(model, pixel_values, pixel_mask, warm_ref_points=None):
    base = model.model
    output_attentions = False
    output_hidden_states = False
    return_dict = True

    batch_size, _, height, width = pixel_values.shape
    device = pixel_values.device

    if pixel_mask is None:
        pixel_mask = torch.ones((batch_size, height, width), dtype=torch.long, device=device)

    features, position_embeddings_list = base.backbone(pixel_values, pixel_mask)

    sources = []
    masks = []
    for level, (source, mask) in enumerate(features):
        sources.append(base.input_proj[level](source))
        masks.append(mask)
        if mask is None:
            raise ValueError("No attention mask was provided")

    if base.config.num_feature_levels > len(sources):
        _len_sources = len(sources)
        for level in range(_len_sources, base.config.num_feature_levels):
            if level == _len_sources:
                source = base.input_proj[level](features[-1][0])
            else:
                source = base.input_proj[level](sources[-1])
            mask = torch.nn.functional.interpolate(pixel_mask[None].to(pixel_values.dtype), size=source.shape[-2:]).to(
                torch.bool
            )[0]
            pos_l = base.backbone.position_embedding(source, mask).to(source.dtype)
            sources.append(source)
            masks.append(mask)
            position_embeddings_list.append(pos_l)

    query_embeds = None
    if not base.config.two_stage:
        query_embeds = base.query_position_embeddings.weight

    source_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes_list = []
    for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
        _, _, h, w = source.shape
        spatial_shape = (h, w)
        spatial_shapes_list.append(spatial_shape)
        source = source.flatten(2).transpose(1, 2)
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed = pos_embed + base.level_embed[level].view(1, 1, -1)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        source_flatten.append(source)
        mask_flatten.append(mask)
    source_flatten = torch.cat(source_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=source_flatten.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.stack([base.get_valid_ratio(m, dtype=source_flatten.dtype) for m in masks], 1)

    encoder_outputs = base.encoder(
        inputs_embeds=source_flatten,
        attention_mask=mask_flatten,
        position_embeddings=lvl_pos_embed_flatten,
        spatial_shapes=spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    if return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    _, _, num_channels = encoder_outputs[0].shape
    enc_outputs_class = None
    enc_outputs_coord_logits = None
    if base.config.two_stage:
        object_query_embedding, output_proposals = base.gen_encoder_output_proposals(
            encoder_outputs[0], ~mask_flatten, spatial_shapes_list
        )
        enc_outputs_class = base.decoder.class_embed[-1](object_query_embedding)
        delta_bbox = base.decoder.bbox_embed[-1](object_query_embedding)
        enc_outputs_coord_logits = delta_bbox + output_proposals

        topk = base.config.two_stage_num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords_logits = torch.gather(
            enc_outputs_coord_logits,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )
        topk_coords_logits = topk_coords_logits.detach()
        reference_points = topk_coords_logits.sigmoid()
        pos_trans_out = base.pos_trans_norm(base.pos_trans(base.get_proposal_pos_embed(topk_coords_logits)))
        query_embed, target = torch.split(pos_trans_out, num_channels, dim=2)
    else:
        query_embed, target = torch.split(query_embeds, num_channels, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        target = target.unsqueeze(0).expand(batch_size, -1, -1)
        reference_points = base.reference_points(query_embed).sigmoid()

    reference_points = _apply_warm_ref(reference_points, warm_ref_points)
    init_reference_points = reference_points

    decoder_outputs = base.decoder(
        inputs_embeds=target,
        position_embeddings=query_embed,
        encoder_hidden_states=encoder_outputs[0],
        encoder_attention_mask=mask_flatten,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        spatial_shapes_list=spatial_shapes_list,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = decoder_outputs.intermediate_hidden_states
    init_reference = init_reference_points
    inter_references = decoder_outputs.intermediate_reference_points

    outputs_classes = []
    outputs_coords = []
    for level in range(hidden_states.shape[1]):
        reference = init_reference if level == 0 else inter_references[:, level - 1]
        reference = inverse_sigmoid(reference)
        outputs_class = model.class_embed[level](hidden_states[:, level])
        delta_bbox = model.bbox_embed[level](hidden_states[:, level])
        if reference.shape[-1] == 4:
            outputs_coord_logits = delta_bbox + reference
        elif reference.shape[-1] == 2:
            delta_bbox[..., :2] += reference
            outputs_coord_logits = delta_bbox
        else:
            raise ValueError(f"reference.shape[-1] should be 4 or 2, got {reference.shape[-1]}")
        outputs_coord = outputs_coord_logits.sigmoid()
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    logits = torch.stack(outputs_classes)[-1]
    pred_boxes = torch.stack(outputs_coords)[-1]

    return SimpleNamespace(logits=logits, pred_boxes=pred_boxes)


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
    parser.add_argument("--model-name", default="SenseTime/deformable-detr")
    parser.add_argument("--images-root", required=True, help="Tracking images root (val).")
    parser.add_argument("--video-ids", nargs="*", default=None, help="Video id list.")
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

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = DeformableDetrForObjectDetection.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    tracker = ByteTrackWrapper(ByteTrackConfig(), per_class=args.per_class)

    if args.video_ids:
        video_ids = args.video_ids
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
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            pixel_mask = inputs.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)

            with torch.no_grad():
                outputs = _forward_with_warm(model, pixel_values, pixel_mask, prev_ref if args.warm_start else None)
                target_sizes = torch.tensor([[height, width]], device=device)
                results = processor.post_process_object_detection(
                    outputs, threshold=args.score_thresh, target_sizes=target_sizes
                )[0]

            detections = _detections_from_results(results)
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
