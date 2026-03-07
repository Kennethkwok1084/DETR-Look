#!/usr/bin/env python3
"""
Generate pre-training figures for thesis/report use.

Outputs:
  - coco_conversion_flowchart.png
  - deformable_detr_simplified.png
  - training_strategy_overview.png
  - cases_success_collage.jpg
  - cases_failure_collage.jpg
  - case_selection.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image, ImageDraw, ImageFont
import yaml


DEFAULT_COCO_JSON = "data/traffic_coco/bdd100k_det/annotations/instances_val.json"
DEFAULT_IMAGE_ROOT = "data/traffic_coco/bdd100k_det/images/val"
DEFAULT_OUTPUT_DIR = "outputs/figures"
DEFAULT_CN_FONT = "outputs/simsun.ttc"
DEFAULT_EN_FONT = "outputs/times.ttf"


COLORS = {
    "vehicle": (255, 0, 0),
    "traffic_sign": (0, 180, 0),
    "traffic_light": (0, 80, 255),
    "default": (255, 255, 255),
}

LABEL_MAP_ZH = {
    "vehicle": "车辆",
    "traffic_sign": "交通标志",
    "traffic_light": "交通灯",
}


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_classes_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("COARSE_CLASSES", {})


def add_text_lines(ax, x, y, w, h, lines):
    if not lines:
        return
    count = len(lines)
    for idx, line in enumerate(lines):
        text = line.get("text", "")
        fontsize = line.get("fontsize", 10)
        font_props = line.get("font_props")
        cy = y + h * (1 - (idx + 0.5) / count)
        ax.text(
            x + w / 2,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            fontproperties=font_props,
        )


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text=None,
    lines=None,
    fc="#FFFFFF",
    ec="#222222",
    fontsize=10,
    font_props=None,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    if lines is not None:
        add_text_lines(ax, x, y, w, h, lines)
    elif text is not None:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#111111",
            fontproperties=font_props,
            wrap=True,
        )


def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color="#333333",
    )
    ax.add_patch(arrow)


def init_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def generate_coco_flowchart(output_path: Path, cn_font, en_font) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    init_axes(ax)

    add_box(
        ax,
        0.05,
        0.68,
        0.18,
        0.12,
        lines=[
            {"text": "BDD100K", "font_props": en_font, "fontsize": 10},
            {"text": "原始数据", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#F6F1E1",
    )
    add_box(
        ax,
        0.05,
        0.48,
        0.18,
        0.12,
        lines=[
            {"text": "CCTSDB", "font_props": en_font, "fontsize": 10},
            {"text": "原始数据", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#F6F1E1",
    )
    add_box(
        ax,
        0.05,
        0.28,
        0.18,
        0.12,
        lines=[
            {"text": "TT100K", "font_props": en_font, "fontsize": 10},
            {"text": "原始数据", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#F6F1E1",
    )

    add_box(
        ax,
        0.30,
        0.48,
        0.20,
        0.16,
        lines=[
            {"text": "convert_to_coco.py", "font_props": en_font, "fontsize": 9},
            {"text": "解析 + 归一化", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#E8F4FF",
    )

    add_box(
        ax,
        0.55,
        0.60,
        0.18,
        0.12,
        lines=[
            {"text": "COCO", "font_props": en_font, "fontsize": 10},
            {"text": "标注", "font_props": cn_font, "fontsize": 10},
            {"text": "JSON", "font_props": en_font, "fontsize": 10},
        ],
        fc="#E9F7EF",
    )
    add_box(
        ax,
        0.55,
        0.40,
        0.18,
        0.12,
        lines=[
            {"text": "图像", "font_props": cn_font, "fontsize": 10},
            {"text": "训练/验证", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#E9F7EF",
    )

    add_box(
        ax,
        0.75,
        0.48,
        0.18,
        0.16,
        lines=[
            {"text": "校验 + 统计", "font_props": cn_font, "fontsize": 10},
            {"text": "validate_coco.py", "font_props": en_font, "fontsize": 9},
        ],
        fc="#FFF2E6",
    )

    add_box(
        ax,
        0.75,
        0.15,
        0.18,
        0.14,
        lines=[
            {"text": "统一数据集", "font_props": cn_font, "fontsize": 10},
            {"text": "COCO", "font_props": en_font, "fontsize": 10},
        ],
        fc="#F2E9FF",
    )

    add_arrow(ax, 0.23, 0.74, 0.30, 0.58)
    add_arrow(ax, 0.23, 0.54, 0.30, 0.56)
    add_arrow(ax, 0.23, 0.34, 0.30, 0.54)

    add_arrow(ax, 0.50, 0.56, 0.55, 0.66)
    add_arrow(ax, 0.50, 0.56, 0.55, 0.46)

    add_arrow(ax, 0.73, 0.66, 0.75, 0.56)
    add_arrow(ax, 0.73, 0.46, 0.75, 0.52)

    add_arrow(ax, 0.84, 0.48, 0.84, 0.29)

    ax.text(
        0.5,
        0.96,
        "多源数据",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=cn_font,
    )
    ax.text(
        0.5,
        0.92,
        "COCO",
        ha="center",
        va="center",
        fontsize=13,
        fontproperties=en_font,
    )
    ax.text(
        0.5,
        0.88,
        "化流程图",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=cn_font,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_deformable_detr_diagram(output_path: Path, cn_font, en_font) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    init_axes(ax)

    add_box(
        ax,
        0.03,
        0.40,
        0.12,
        0.20,
        lines=[{"text": "输入图像", "font_props": cn_font, "fontsize": 10}],
        fc="#F6F1E1",
    )
    add_box(
        ax,
        0.19,
        0.35,
        0.14,
        0.30,
        lines=[
            {"text": "CNN", "font_props": en_font, "fontsize": 10},
            {"text": "骨干网络", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#E8F4FF",
    )
    add_box(
        ax,
        0.37,
        0.35,
        0.16,
        0.30,
        lines=[{"text": "多尺度特征", "font_props": cn_font, "fontsize": 10}],
        fc="#E9F7EF",
    )
    add_box(
        ax,
        0.57,
        0.35,
        0.18,
        0.30,
        lines=[
            {"text": "Deformable Transformer", "font_props": en_font, "fontsize": 9},
            {"text": "编码/解码", "font_props": cn_font, "fontsize": 10},
        ],
        fc="#FFF2E6",
    )
    add_box(
        ax,
        0.80,
        0.35,
        0.16,
        0.30,
        lines=[{"text": "预测头", "font_props": cn_font, "fontsize": 10}],
        fc="#F2E9FF",
    )

    add_box(
        ax,
        0.45,
        0.72,
        0.16,
        0.12,
        lines=[
            {"text": "Positional Encoding", "font_props": en_font, "fontsize": 8},
            {"text": "位置编码", "font_props": cn_font, "fontsize": 9},
        ],
        fc="#EFEFEF",
    )
    add_box(
        ax,
        0.57,
        0.08,
        0.18,
        0.16,
        lines=[
            {"text": "Object Queries", "font_props": en_font, "fontsize": 8},
            {"text": "对象查询", "font_props": cn_font, "fontsize": 9},
        ],
        fc="#EFEFEF",
    )

    add_arrow(ax, 0.15, 0.50, 0.19, 0.50)
    add_arrow(ax, 0.33, 0.50, 0.37, 0.50)
    add_arrow(ax, 0.53, 0.50, 0.57, 0.50)
    add_arrow(ax, 0.75, 0.50, 0.80, 0.50)

    add_arrow(ax, 0.53, 0.72, 0.57, 0.64)
    add_arrow(ax, 0.66, 0.24, 0.66, 0.35)

    add_box(
        ax,
        0.80,
        0.05,
        0.16,
        0.18,
        lines=[
            {"text": "输出", "font_props": cn_font, "fontsize": 9},
            {"text": "类别 + 边框", "font_props": cn_font, "fontsize": 9},
        ],
        fc="#E9F7EF",
    )
    add_arrow(ax, 0.88, 0.35, 0.88, 0.23)

    ax.text(
        0.5,
        0.96,
        "Deformable DETR",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=en_font,
    )
    ax.text(
        0.5,
        0.92,
        "简版结构图",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=cn_font,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_training_strategy(output_path: Path, cn_font, en_font) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    init_axes(ax)

    center = Circle((0.5, 0.5), 0.12, edgecolor="#333333", facecolor="#F6F6F6", linewidth=1.5)
    ax.add_patch(center)
    ax.text(0.5, 0.5, "训练循环", ha="center", va="center", fontsize=11, fontproperties=cn_font)

    add_box(
        ax,
        0.08,
        0.70,
        0.26,
        0.18,
        lines=[
            {"text": "PR", "font_props": en_font, "fontsize": 10},
            {"text": "渐进式分辨率", "font_props": cn_font, "fontsize": 10},
            {"text": "分阶段调整", "font_props": cn_font, "fontsize": 9},
        ],
        fc="#E8F4FF",
    )
    add_box(
        ax,
        0.66,
        0.70,
        0.26,
        0.18,
        lines=[
            {"text": "AMP", "font_props": en_font, "fontsize": 10},
            {"text": "自动混合精度", "font_props": cn_font, "fontsize": 10},
            {"text": "autocast + scaler", "font_props": en_font, "fontsize": 8},
        ],
        fc="#FFF2E6",
    )
    add_box(
        ax,
        0.30,
        0.14,
        0.40,
        0.18,
        lines=[
            {"text": "多源数据", "font_props": cn_font, "fontsize": 10},
            {"text": "COCO", "font_props": en_font, "fontsize": 10},
            {"text": "可选合并", "font_props": cn_font, "fontsize": 9},
        ],
        fc="#E9F7EF",
    )

    add_arrow(ax, 0.34, 0.70, 0.44, 0.58)
    add_arrow(ax, 0.66, 0.70, 0.56, 0.58)
    add_arrow(ax, 0.50, 0.32, 0.50, 0.38)

    ax.text(
        0.5,
        0.96,
        "训练策略总览",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=cn_font,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_coco(coco_json: Path):
    with coco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    return images, categories, anns_by_image


def get_font(size: int, font_path: str | None = None):
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_boxes(
    image: Image.Image,
    anns,
    categories,
    show_labels=True,
    font_path: str | None = None,
    label_map: dict | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = get_font(16, font_path=font_path)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        x2, y2 = x + w, y + h
        cat_name = categories.get(ann["category_id"], "object")
        color = COLORS.get(cat_name, COLORS["default"])
        draw.rectangle([x, y, x2, y2], outline=color, width=3)
        if show_labels:
            text = label_map.get(cat_name, cat_name) if label_map else cat_name
            text_bbox = draw.textbbox((x, y), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            draw.rectangle([x, y - text_h - 4, x + text_w + 6, y], fill=color)
            draw.text((x + 3, y - text_h - 2), text, fill="white", font=font)
    return image


def compute_difficulty(anns) -> dict:
    if not anns:
        return {"num": 0, "small_ratio": 0.0, "score": 0.0}
    areas = [ann["bbox"][2] * ann["bbox"][3] for ann in anns]
    small_thresh = 32 * 32
    small = sum(1 for a in areas if a < small_thresh)
    num = len(areas)
    small_ratio = small / max(num, 1)
    score = num * (1.0 + 4.0 * small_ratio)
    return {"num": num, "small_ratio": small_ratio, "score": score}


def select_cases(images, anns_by_image, image_root: Path, count=6, seed=42):
    random.seed(seed)
    stats = []
    for img_id, img_info in images.items():
        img_path = image_root / img_info["file_name"]
        if not img_path.exists():
            continue
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue
        diff = compute_difficulty(anns)
        stats.append((diff["score"], diff["num"], diff["small_ratio"], img_id, img_info))
    stats.sort(key=lambda x: x[0])

    easy = stats[:count]
    hard = stats[-count:]

    return easy, hard


def make_collage(
    selections,
    anns_by_image,
    categories,
    image_root: Path,
    output_path: Path,
    title: str,
    tile_size=(640, 360),
    cn_font_path=None,
):
    cols = 3
    rows = math.ceil(len(selections) / cols)
    pad = 12
    title_h = 50
    tile_w, tile_h = tile_size

    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = rows * tile_h + (rows + 1) * pad + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(24, font_path=cn_font_path)
    draw.text((pad, 8), title, fill=(20, 20, 20), font=title_font)

    for idx, (_, num, small_ratio, img_id, img_info) in enumerate(selections):
        img_path = image_root / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        anns = anns_by_image.get(img_id, [])
        image = draw_boxes(
            image,
            anns,
            categories,
            show_labels=True,
            font_path=cn_font_path,
            label_map=LABEL_MAP_ZH,
        )
        image = image.resize(tile_size, Image.Resampling.LANCZOS)

        row = idx // cols
        col = idx % cols
        x = pad + col * (tile_w + pad)
        y = title_h + pad + row * (tile_h + pad)
        canvas.paste(image, (x, y))

        meta_text = f"目标数={num} 小目标占比={small_ratio:.2f}"
        meta_font = get_font(16, font_path=cn_font_path)
        meta_bbox = draw.textbbox((0, 0), meta_text, font=meta_font)
        meta_w = meta_bbox[2] - meta_bbox[0]
        meta_h = meta_bbox[3] - meta_bbox[1]
        draw.rectangle(
            [x + 8, y + 8, x + 8 + meta_w + 8, y + 8 + meta_h + 6],
            fill=(0, 0, 0),
        )
        draw.text((x + 12, y + 10), meta_text, fill=(255, 255, 255), font=meta_font)

    canvas.save(output_path, quality=92)


def generate_case_collages(output_dir: Path, coco_json: Path, image_root: Path, seed=42, cn_font_path=None):
    images, categories, anns_by_image = load_coco(coco_json)
    easy, hard = select_cases(images, anns_by_image, image_root, count=6, seed=seed)

    make_collage(
        easy,
        anns_by_image,
        categories,
        image_root,
        output_dir / "cases_success_collage.jpg",
        title="成功案例（易样本）",
        cn_font_path=cn_font_path,
    )
    make_collage(
        hard,
        anns_by_image,
        categories,
        image_root,
        output_dir / "cases_failure_collage.jpg",
        title="失败案例（难样本）",
        cn_font_path=cn_font_path,
    )

    selection_record = {
        "success_cases": [
            {
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "num_objects": num,
                "small_ratio": round(small_ratio, 4),
                "score": round(score, 4),
            }
            for score, num, small_ratio, img_id, img_info in easy
        ],
        "failure_cases": [
            {
                "image_id": img_id,
                "file_name": img_info["file_name"],
                "num_objects": num,
                "small_ratio": round(small_ratio, 4),
                "score": round(score, 4),
            }
            for score, num, small_ratio, img_id, img_info in hard
        ],
        "notes": "样本按难度分数排序（小目标占比越高越难）。",
    }
    with (output_dir / "case_selection.json").open("w", encoding="utf-8") as f:
        json.dump(selection_record, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pre-training figures.")
    parser.add_argument("--coco-json", default=DEFAULT_COCO_JSON, help="COCO annotation JSON path")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT, help="COCO image root path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cn-font", default=DEFAULT_CN_FONT, help="Chinese font path (SimSun)")
    parser.add_argument("--en-font", default=DEFAULT_EN_FONT, help="English font path (Times)")
    return parser.parse_args()


def resolve_font(font_path: str, label: str):
    path = Path(font_path)
    if not path.exists():
        raise FileNotFoundError(f"{label} font not found: {path}")
    try:
        font_manager.fontManager.addfont(str(path))
    except Exception:
        pass
    font_props = font_manager.FontProperties(fname=str(path))
    try:
        name = font_props.get_name()
        print(f"[INFO] {label} font: {name} ({path})")
    except Exception:
        print(f"[INFO] {label} font: {path}")
    return font_props


def generate_progressive_resizing_cn(output_path: Path, cn_font, en_font) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    init_axes(ax)

    stages = [
        {
            "title": "阶段1",
            "resolution": "输入: 640×640",
            "epoch": "epoch: 1-19",
            "goal": "目标: 快速收敛/稳定损失",
            "color": "#E8F4FF",
        },
        {
            "title": "阶段2",
            "resolution": "输入: 800×800",
            "epoch": "epoch: 20-39",
            "goal": "目标: 兼顾速度与精度",
            "color": "#FFF2E6",
        },
        {
            "title": "阶段3",
            "resolution": "输入: 960×960",
            "epoch": "epoch: 40+",
            "goal": "目标: 提升小目标AP",
            "color": "#E9F7EF",
        },
    ]

    box_w = 0.26
    box_h = 0.55
    y = 0.25
    xs = [0.06, 0.37, 0.68]

    for idx, stage in enumerate(stages):
        add_box(
            ax,
            xs[idx],
            y,
            box_w,
            box_h,
            fc=stage["color"],
            lines=[
                {"text": stage["title"], "font_props": cn_font, "fontsize": 11},
                {"text": stage["resolution"], "font_props": cn_font, "fontsize": 11},
                {"text": stage["epoch"], "font_props": en_font, "fontsize": 10},
                {"text": stage["goal"], "font_props": cn_font, "fontsize": 10},
            ],
        )
        if idx < len(stages) - 1:
            add_arrow(ax, xs[idx] + box_w, y + box_h / 2, xs[idx + 1], y + box_h / 2)

    ax.text(
        0.5,
        0.92,
        "Progressive Resizing",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=en_font,
    )
    ax.text(
        0.5,
        0.88,
        "分阶段训练示意",
        ha="center",
        va="center",
        fontsize=14,
        fontproperties=cn_font,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)
    cn_font = resolve_font(args.cn_font, "Chinese")
    en_font = resolve_font(args.en_font, "English")

    generate_coco_flowchart(output_dir / "coco_conversion_flowchart.png", cn_font, en_font)
    generate_deformable_detr_diagram(output_dir / "deformable_detr_simplified.png", cn_font, en_font)
    generate_training_strategy(output_dir / "training_strategy_overview.png", cn_font, en_font)
    generate_progressive_resizing_cn(output_dir / "progressive_resizing_cn.png", cn_font, en_font)

    coco_json = Path(args.coco_json)
    image_root = Path(args.image_root)
    if coco_json.exists() and image_root.exists():
        generate_case_collages(
            output_dir,
            coco_json,
            image_root,
            seed=args.seed,
            cn_font_path=args.cn_font,
        )


if __name__ == "__main__":
    main()
