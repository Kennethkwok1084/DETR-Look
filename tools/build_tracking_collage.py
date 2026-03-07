#!/usr/bin/env python3
"""
Build a two-column collage for baseline vs warm-start frames.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_frame_list(baseline_dir: Path, ours_dir: Path, max_frames: int) -> List[str]:
    baseline_frames = sorted([p.name for p in baseline_dir.glob("*.jpg")])
    if not baseline_frames:
        baseline_frames = sorted([p.name for p in baseline_dir.glob("*.png")])
    if not baseline_frames:
        raise FileNotFoundError(f"No frames found in {baseline_dir}")

    frames = []
    for name in baseline_frames:
        if (ours_dir / name).exists():
            frames.append(name)
        if len(frames) >= max_frames:
            break
    if not frames:
        raise FileNotFoundError("No matching frames between baseline and ours.")
    return frames


def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    new_width = int(img.width * (height / img.height))
    return img.resize((new_width, height), Image.BILINEAR)


def _draw_titles(
    canvas: Image.Image,
    titles: Tuple[str, str],
    font: ImageFont.ImageFont,
    padding: int,
    left_x: int,
    right_x: int,
) -> int:
    draw = ImageDraw.Draw(canvas)
    title_height = 0
    for title in titles:
        if not title:
            continue
        bbox = draw.textbbox((0, 0), title, font=font)
        title_height = max(title_height, bbox[3] - bbox[1])
    if title_height == 0:
        return 0
    draw.text((left_x + padding, padding), titles[0], fill=(255, 255, 255), font=font)
    draw.text((right_x + padding, padding), titles[1], fill=(255, 255, 255), font=font)
    return title_height + padding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, help="Dir with baseline visualized frames.")
    parser.add_argument("--ours-dir", required=True, help="Dir with warm-start visualized frames.")
    parser.add_argument("--output", required=True, help="Output collage path.")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--spacing", type=int, default=8)
    parser.add_argument("--title-left", default="Baseline")
    parser.add_argument("--title-right", default="Warm-start")
    parser.add_argument("--font-path", default=None)
    parser.add_argument("--font-size", type=int, default=20)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    ours_dir = Path(args.ours_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _load_frame_list(baseline_dir, ours_dir, args.max_frames)

    baseline_imgs: List[Image.Image] = []
    ours_imgs: List[Image.Image] = []
    for name in frames:
        baseline_imgs.append(Image.open(baseline_dir / name).convert("RGB"))
        ours_imgs.append(Image.open(ours_dir / name).convert("RGB"))

    # Normalize heights per row for alignment
    row_heights = []
    resized_rows = []
    for base, ours in zip(baseline_imgs, ours_imgs):
        row_height = max(base.height, ours.height)
        base_resized = _resize_to_height(base, row_height)
        ours_resized = _resize_to_height(ours, row_height)
        row_heights.append(row_height)
        resized_rows.append((base_resized, ours_resized))

    col1_width = max(img.width for img, _ in resized_rows)
    col2_width = max(img.width for _, img in resized_rows)
    spacing = args.spacing

    # Title space
    if args.font_path and Path(args.font_path).exists():
        font = ImageFont.truetype(args.font_path, args.font_size)
    else:
        font = ImageFont.load_default()
    title_padding = spacing
    title_height = 0
    if args.title_left or args.title_right:
        title_height = args.font_size + title_padding

    total_height = sum(row_heights) + spacing * (len(row_heights) - 1) + title_height
    total_width = col1_width + col2_width + spacing
    canvas = Image.new("RGB", (total_width, total_height), (10, 10, 10))

    y = 0
    if title_height > 0:
        _draw_titles(
            canvas,
            (args.title_left, args.title_right),
            font,
            title_padding // 2,
            0,
            col1_width + spacing,
        )
        y += title_height

    for (base, ours), row_h in zip(resized_rows, row_heights):
        x1 = 0
        x2 = col1_width + spacing
        canvas.paste(base, (x1, y))
        canvas.paste(ours, (x2, y))
        y += row_h + spacing

    canvas.save(output_path)
    print(f"[OK] collage saved to {output_path}")


if __name__ == "__main__":
    main()
