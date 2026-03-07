#!/usr/bin/env python3
"""
COCO 格式 Ground Truth 可视化脚本
读取 COCO JSON 标注文件，在原图上绘制 GT 边界框与类别标签
支持类别过滤、置信度显示、批量可视化
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

from PIL import Image, ImageDraw, ImageFont
import yaml

# 默认颜色方案（对应 classes.yaml 的3个粗粒度类别）
DEFAULT_COLORS = {
    "vehicle": (255, 0, 0),         # 红色
    "traffic_sign": (0, 255, 0),    # 绿色
    "traffic_light": (0, 0, 255),   # 蓝色
    "default": (255, 255, 255)      # 白色
}


class COCOGroundTruthVisualizer:
    """COCO GT 可视化器"""
    
    def __init__(
        self,
        coco_json: str,
        image_root: str,
        classes_yaml: str = None,
        colors: Dict[str, Tuple[int, int, int]] = None
    ):
        """
        初始化可视化器
        
        Args:
            coco_json: COCO 标注文件路径
            image_root: 图片根目录
            classes_yaml: 类别映射文件（可选）
            colors: 自定义颜色映射（可选）
        """
        self.image_root = Path(image_root)
        self.colors = colors if colors else DEFAULT_COLORS
        
        print(f"[INFO] 加载 COCO 标注: {coco_json}")
        with open(coco_json, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        # 构建索引
        self._build_indices()
        
        # 加载类别配置
        self.coarse_classes = None
        if classes_yaml and Path(classes_yaml).exists():
            self.coarse_classes = self._load_class_config(classes_yaml)
            print(f"[INFO] 已加载类别配置: {classes_yaml}")
        
        print(f"[INFO] 数据集信息:")
        print(f"  - 图片数量: {len(self.images)}")
        print(f"  - 标注数量: {len(self.annotations)}")
        print(f"  - 类别数量: {len(self.categories)}")
    
    def _build_indices(self):
        """构建快速索引"""
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        
        # 按图片ID分组标注
        self.annotations_by_image = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # 保存原始标注列表
        self.annotations = self.coco_data["annotations"]
    
    def _load_class_config(self, yaml_path: str) -> Dict:
        """加载 classes.yaml 配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('COARSE_CLASSES', {})
    
    def get_category_color(self, category_name: str) -> Tuple[int, int, int]:
        """根据类别名称获取颜色"""
        # 如果有粗粒度映射，尝试使用映射后的类别颜色
        if self.coarse_classes:
            for coarse_id, coarse_name in self.coarse_classes.items():
                if category_name == coarse_name:
                    return self.colors.get(coarse_name, self.colors["default"])
        
        # 否则直接查找
        return self.colors.get(category_name, self.colors["default"])
    
    def visualize_single(
        self,
        image_id: int,
        output_path: str = None,
        show_labels: bool = True,
        show_area: bool = False,
        font_size: int = 20,
        line_width: int = 3,
        category_filter: List[str] = None
    ) -> Optional[Image.Image]:
        """
        可视化单张图片的 GT
        
        Args:
            image_id: 图片ID
            output_path: 保存路径（可选）
            show_labels: 是否显示标签
            show_area: 是否显示面积
            font_size: 字体大小
            line_width: 线宽
            category_filter: 类别过滤列表（只显示指定类别）
        
        Returns:
            annotated_image: 标注后的图片
        """
        # 获取图片信息
        if image_id not in self.images:
            print(f"[WARNING] 图片ID {image_id} 不存在")
            return None
        
        img_info = self.images[image_id]
        img_path = self.image_root / img_info["file_name"]
        
        if not img_path.exists():
            print(f"[WARNING] 图片文件不存在: {img_path}")
            return None
        
        # 加载图片
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # 加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 获取该图片的所有标注
        annotations = self.annotations_by_image.get(image_id, [])
        
        # 绘制每个标注
        drawn_count = 0
        for ann in annotations:
            category_id = ann["category_id"]
            category_name = self.categories[category_id]
            
            # 类别过滤
            if category_filter and category_name not in category_filter:
                continue
            
            # COCO bbox 格式: [x, y, width, height]
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            # 获取颜色
            color = self.get_category_color(category_name)
            
            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            
            # 画标签
            if show_labels:
                text = category_name
                if show_area:
                    area = ann.get("area", w * h)
                    text += f" ({int(area)}px²)"
                
                # 获取文本边界框
                bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # 画背景矩形
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                    fill=color
                )
                draw.text((x1 + 2, y1 - text_height - 2), text, fill="white", font=font)
            
            drawn_count += 1
        
        print(f"[INFO] 图片 {img_info['file_name']}: 绘制 {drawn_count}/{len(annotations)} 个标注")
        
        # 保存图片
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            print(f"[INFO] 保存到: {output_path}")
        
        return image
    
    def visualize_batch(
        self,
        output_dir: str,
        max_images: int = None,
        category_filter: List[str] = None,
        **viz_kwargs
    ):
        """
        批量可视化所有图片
        
        Args:
            output_dir: 输出目录
            max_images: 最大处理数量
            category_filter: 类别过滤列表
            **viz_kwargs: 传递给 visualize_single 的其他参数
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_ids = list(self.images.keys())
        if max_images:
            image_ids = image_ids[:max_images]
        
        print(f"[INFO] 开始批量可视化 {len(image_ids)} 张图片")
        
        success_count = 0
        for idx, img_id in enumerate(image_ids, 1):
            img_info = self.images[img_id]
            print(f"[{idx}/{len(image_ids)}] 处理: {img_info['file_name']}")
            
            try:
                output_path = output_dir / f"{Path(img_info['file_name']).stem}_gt.jpg"
                result = self.visualize_single(
                    image_id=img_id,
                    output_path=str(output_path),
                    category_filter=category_filter,
                    **viz_kwargs
                )
                
                if result is not None:
                    success_count += 1
            
            except Exception as e:
                print(f"[ERROR] 处理失败: {e}")
                continue
        
        print(f"[INFO] 批量可视化完成: 成功 {success_count}/{len(image_ids)} 张")
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            "total_images": len(self.images),
            "total_annotations": len(self.annotations),
            "categories": {}
        }
        
        # 按类别统计
        for ann in self.annotations:
            cat_id = ann["category_id"]
            cat_name = self.categories[cat_id]
            
            if cat_name not in stats["categories"]:
                stats["categories"][cat_name] = 0
            stats["categories"][cat_name] += 1
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print("数据集统计")
        print("="*50)
        print(f"总图片数: {stats['total_images']}")
        print(f"总标注数: {stats['total_annotations']}")
        print(f"\n各类别分布:")
        
        for cat_name, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats['total_annotations'] * 100
            print(f"  {cat_name:20s}: {count:6d} ({percentage:5.2f}%)")
        
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="COCO GT 可视化工具")
    parser.add_argument("--coco_json", type=str, required=True,
                        help="COCO 标注文件路径")
    parser.add_argument("--image_root", type=str, required=True,
                        help="图片根目录")
    parser.add_argument("--output_dir", type=str, default="outputs/demo_gt",
                        help="输出目录 (默认: outputs/demo_gt)")
    parser.add_argument("--image_id", type=int,
                        help="单张图片ID（可选，不指定则批量处理）")
    parser.add_argument("--max_images", type=int,
                        help="最大处理图片数量")
    parser.add_argument("--classes_yaml", type=str, default="../configs/classes.yaml",
                        help="类别映射文件")
    parser.add_argument("--category_filter", type=str, nargs="+",
                        help="只显示指定类别（如: vehicle traffic_sign）")
    parser.add_argument("--no_labels", action="store_true",
                        help="不显示标签文字")
    parser.add_argument("--show_area", action="store_true",
                        help="显示标注面积")
    parser.add_argument("--font_size", type=int, default=20,
                        help="字体大小")
    parser.add_argument("--line_width", type=int, default=3,
                        help="边框线宽")
    parser.add_argument("--stats", action="store_true",
                        help="只打印统计信息，不生成可视化")
    
    args = parser.parse_args()
    
    # 初始化可视化器
    visualizer = COCOGroundTruthVisualizer(
        coco_json=args.coco_json,
        image_root=args.image_root,
        classes_yaml=args.classes_yaml if Path(args.classes_yaml).exists() else None
    )
    
    # 打印统计信息
    if args.stats:
        visualizer.print_statistics()
        return
    
    # 执行可视化
    if args.image_id is not None:
        # 单张可视化
        print(f"[INFO] 单张可视化模式: image_id={args.image_id}")
        output_path = Path(args.output_dir) / f"image_{args.image_id}_gt.jpg"
        visualizer.visualize_single(
            image_id=args.image_id,
            output_path=str(output_path),
            show_labels=not args.no_labels,
            show_area=args.show_area,
            font_size=args.font_size,
            line_width=args.line_width,
            category_filter=args.category_filter
        )
    else:
        # 批量可视化
        print(f"[INFO] 批量可视化模式")
        visualizer.visualize_batch(
            output_dir=args.output_dir,
            max_images=args.max_images,
            show_labels=not args.no_labels,
            show_area=args.show_area,
            font_size=args.font_size,
            line_width=args.line_width,
            category_filter=args.category_filter
        )
    
    # 打印统计信息
    visualizer.print_statistics()


if __name__ == "__main__":
    main()
