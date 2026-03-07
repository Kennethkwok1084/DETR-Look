#!/usr/bin/env python3
"""
Generate a simple flowchart for warm-start + ByteTrack pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _box(ax, xy, text, font, width=2.4, height=0.9):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#222222",
        facecolor="#f2f2f2",
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        fontproperties=font,
    )


def _arrow(ax, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.2, color="#222222")
    ax.add_patch(arrow)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--font-path", default="output/simsum.ttc", help="Chinese font path.")
    parser.add_argument("--title", default="跨帧 Warm-start + ByteTrack 流程")
    args = parser.parse_args()

    font_path = Path(args.font_path)
    if not font_path.exists():
        raise FileNotFoundError(f"Font not found: {font_path}")
    font = FontProperties(fname=str(font_path))

    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)

    _box(ax, (0.5, 1.6), "输入帧图像 t", font)
    _box(ax, (3.4, 1.6), "Deformable DETR 检测", font)
    _box(ax, (6.4, 2.4), "跨帧缓存\n(top-K ref)", font)
    _box(ax, (6.4, 0.8), "ByteTrack 在线关联", font)
    _box(ax, (9.0, 1.6), "轨迹输出 + 指标", font)

    _arrow(ax, (2.9, 2.05), (3.4, 2.05))
    _arrow(ax, (5.8, 2.05), (6.4, 2.85))
    _arrow(ax, (5.8, 2.05), (6.4, 1.25))
    _arrow(ax, (8.8, 1.25), (9.0, 2.05))

    ax.set_title(args.title, fontsize=12, pad=8, fontproperties=font)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"[OK] flowchart -> {output_path}")


if __name__ == "__main__":
    main()
