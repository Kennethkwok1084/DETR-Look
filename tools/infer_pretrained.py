#!/usr/bin/env python3
"""
DETR 预训练模型推理脚本
使用 Hugging Face transformers.pipeline 加载预训练模型对交通场景图片进行目标检测
支持批量推理、单张推理、keep_labels 过滤，输出可视化结果
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
import yaml

# 默认颜色映射（COCO 91类子集映射到交通相关类别）
TRAFFIC_COLORS = {
    "car": (255, 0, 0),          # 红色
    "truck": (255, 165, 0),      # 橙色
    "bus": (255, 255, 0),        # 黄色
    "traffic light": (0, 255, 0), # 绿色
    "stop sign": (0, 0, 255),    # 蓝色
    "motorcycle": (128, 0, 128),  # 紫色
    "bicycle": (0, 255, 255),     # 青色
    "person": (255, 192, 203),    # 粉色
    "default": (255, 255, 255)    # 默认白色
}


class PretrainedDETRInference:
    """预训练 DETR 模型推理封装类（使用 pipeline）"""
    
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        device: int = -1,
        confidence_threshold: float = 0.8,
        keep_labels: Optional[List[str]] = None,
        classes_yaml: str = None
    ):
        """
        初始化推理器
        
        Args:
            model_name: HF 模型名称
            device: 设备 ID (-1=CPU, 0=GPU0, ...)
            confidence_threshold: 置信度阈值
            keep_labels: 保留的类别列表（过滤用）
            classes_yaml: 类别映射文件路径（可选）
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.keep_labels = set(keep_labels) if keep_labels else None
        
        print(f"[INFO] 加载模型: {model_name}")
        print(f"[INFO] 使用设备: {'GPU' if device >= 0 else 'CPU'}")
        if self.keep_labels:
            print(f"[INFO] 类别过滤: {', '.join(sorted(self.keep_labels))}")
        
        # 使用 pipeline 加载模型
        self.pipe = pipeline(
            "object-detection",
            model=model_name,
            device=device
        )
        
        # 加载类别映射（如果提供）
        self.coarse_mapping = None
        if classes_yaml and Path(classes_yaml).exists():
            self.coarse_mapping = self._load_class_mapping(classes_yaml)
            print(f"[INFO] 已加载类别映射: {classes_yaml}")
        
        print(f"[INFO] 模型加载完成")
    
    def _load_class_mapping(self, yaml_path: str) -> Dict:
        """加载 classes.yaml 的粗粒度映射"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('BDD100K_MAPPING', {})
    
    def infer_single(self, image_path: str) -> Tuple[List[Dict], Image.Image]:
        """
        对单张图片进行推理
        
        Args:
            image_path: 图片路径
        
        Returns:
            detections: 检测结果列表 [{"label": str, "score": float, "box": [x1,y1,x2,y2]}]
            image: PIL Image 对象
        """
        image = Image.open(image_path).convert("RGB")
        
        # 使用 pipeline 推理
        raw_results = self.pipe(image, threshold=self.confidence_threshold)
        
        # 格式化结果并应用 keep_labels 过滤
        detections = []
        for result in raw_results:
            label = result["label"]
            
            # keep_labels 过滤
            if self.keep_labels and label not in self.keep_labels:
                continue
            
            # 转换 box 格式: {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
            box = result["box"]
            detections.append({
                "label": label,
                "score": result["score"],
                "box": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
            })
        
        return detections, image
    
    def visualize(
        self,
        image: Image.Image,
        detections: List[Dict],
        output_path: str = None,
        show_labels: bool = True,
        font_size: int = 20
    ) -> Image.Image:
        """
        在图片上绘制检测框
        
        Args:
            image: PIL Image 对象
            detections: 检测结果列表
            output_path: 保存路径（可选）
            show_labels: 是否显示标签
            font_size: 字体大小
        
        Returns:
            annotated_image: 标注后的图片
        """
        draw = ImageDraw.Draw(image)
        
        # 尝试加载字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            label = det["label"]
            score = det["score"]
            x1, y1, x2, y2 = det["box"]
            
            # 获取颜色
            color = TRAFFIC_COLORS.get(label, TRAFFIC_COLORS["default"])
            
            # 画框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 画标签
            if show_labels:
                text = f"{label}: {score:.2f}"
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
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            print(f"[INFO] 保存可视化结果: {output_path}")
        
        return image
    
    def infer_batch(
        self,
        image_dir: str,
        output_dir: str,
        save_json: bool = True,
        save_log: bool = True,
        extensions: List[str] = None
    ):
        """
        批量推理图片目录
        
        Args:
            image_dir: 图片目录
            output_dir: 输出目录
            save_json: 是否保存 JSON 结果
            save_log: 是否保存日志文件
            extensions: 支持的图片扩展名
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        if save_log:
            log_dir = Path("outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"inference_{Path(output_dir).name}.log"
            log_lines = []
        
        # 收集所有图片
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.rglob(f"*{ext}"))
            image_files.extend(image_dir.rglob(f"*{ext.upper()}"))
        
        msg = f"[INFO] 发现 {len(image_files)} 张图片"
        print(msg)
        if save_log:
            log_lines.append(msg)
        
        all_results = {}
        for idx, img_path in enumerate(image_files, 1):
            msg = f"[{idx}/{len(image_files)}] 处理: {img_path.name}"
            print(msg)
            if save_log:
                log_lines.append(msg)
            
            try:
                # 推理
                detections, image = self.infer_single(str(img_path))
                
                # 可视化保存
                output_path = output_dir / f"{img_path.stem}_pred.jpg"
                self.visualize(image, detections, str(output_path))
                
                # 记录结果
                all_results[img_path.name] = {
                    "num_detections": len(detections),
                    "detections": detections
                }
                
                msg = f"  ✓ 检测到 {len(detections)} 个目标"
                print(msg)
                if save_log:
                    log_lines.append(msg)
                
            except Exception as e:
                msg = f"[ERROR] 处理失败 {img_path.name}: {e}"
                print(msg)
                if save_log:
                    log_lines.append(msg)
                continue
        
        # 保存 JSON
        if save_json:
            json_path = output_dir / "detections.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            msg = f"[INFO] 保存检测结果: {json_path}"
            print(msg)
            if save_log:
                log_lines.append(msg)
        
        # 保存日志
        if save_log:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_lines))
            print(f"[INFO] 保存日志: {log_file}")
        
        print(f"[INFO] 批量推理完成，共处理 {len(all_results)} 张图片")


def main():
    parser = argparse.ArgumentParser(description="DETR 预训练模型推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, help="图片目录（批量推理）")
    parser.add_argument("--output_dir", type=str, default="outputs/demo_pred",
                        help="输出目录 (默认: outputs/demo_pred)")
    parser.add_argument("--model", type=str, default="facebook/detr-resnet-50",
                        help="HF 模型名称")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="置信度阈值 (默认: 0.8)")
    parser.add_argument("--device", type=int, default=-1,
                        help="设备 ID (-1=CPU, 0=GPU0, ...)")
    parser.add_argument("--keep_labels", type=str, nargs="+",
                        help="保留的类别列表（如: car truck bus）")
    parser.add_argument("--classes_yaml", type=str, default="../configs/classes.yaml",
                        help="类别映射文件")
    parser.add_argument("--no_json", action="store_true",
                        help="不保存 JSON 结果文件")
    parser.add_argument("--no_log", action="store_true",
                        help="不保存日志文件")
    
    args = parser.parse_args()
    
    # 参数检查
    if not args.image and not args.image_dir:
        parser.error("必须指定 --image 或 --image_dir")
    
    # 初始化推理器
    inferencer = PretrainedDETRInference(
        model_name=args.model,
        device=args.device,
        confidence_threshold=args.threshold,
        keep_labels=args.keep_labels,
        classes_yaml=args.classes_yaml if Path(args.classes_yaml).exists() else None
    )
    
    # 执行推理
    if args.image:
        # 单张推理
        print(f"[INFO] 单张推理模式: {args.image}")
        detections, image = inferencer.infer_single(args.image)
        
        output_path = Path(args.output_dir) / f"{Path(args.image).stem}_pred.jpg"
        inferencer.visualize(image, detections, str(output_path))
        
        print(f"\n检测结果 ({len(detections)} 个目标):")
        for det in detections:
            print(f"  - {det['label']}: {det['score']:.3f}")
    
    elif args.image_dir:
        # 批量推理
        print(f"[INFO] 批量推理模式: {args.image_dir}")
        inferencer.infer_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            save_json=not args.no_json,
            save_log=not args.no_log
        )


if __name__ == "__main__":
    main()
