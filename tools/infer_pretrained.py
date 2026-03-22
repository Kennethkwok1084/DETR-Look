#!/usr/bin/env python3
"""
交通场景模型推理脚本
支持两种模式：
1. 本地 checkpoint 推理（推荐，Deformable DETR 主线）
2. Hugging Face 预训练 DETR 演示推理
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_image_processor, build_model
from utils import load_checkpoint

# 默认颜色映射
TRAFFIC_COLORS = {
    "vehicle": (255, 120, 40),
    "traffic_sign": (40, 120, 255),
    "traffic_light": (60, 200, 120),
    "car": (255, 0, 0),
    "truck": (255, 165, 0),
    "bus": (255, 255, 0),
    "traffic light": (0, 255, 0),
    "stop sign": (0, 0, 255),
    "default": (255, 255, 255),
}


def _resolve_repo_path(path_value: Optional[str]) -> Optional[Path]:
    """将传入路径解析为仓库内绝对路径。"""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class PretrainedDETRInference:
    """统一推理封装类，兼容本地 checkpoint 与 HF 预训练模型。"""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        device: int = -1,
        confidence_threshold: float = 0.8,
        keep_labels: Optional[List[str]] = None,
        classes_yaml: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.keep_labels = set(keep_labels) if keep_labels else None
        self.classes_yaml = _resolve_repo_path(classes_yaml)
        self.config_path = _resolve_repo_path(config_path)
        self.checkpoint_path = _resolve_repo_path(checkpoint_path)
        self.mode = "local" if self.config_path and self.checkpoint_path else "hf"

        self.coarse_mapping: Dict[str, str] = {}
        self.id_to_label: Dict[int, str] = {}
        if self.classes_yaml and self.classes_yaml.exists():
            self.coarse_mapping, self.id_to_label = self._load_class_mappings(self.classes_yaml)

        if self.keep_labels:
            print(f"[INFO] 类别过滤: {', '.join(sorted(self.keep_labels))}")

        if self.mode == "local":
            self._init_local_model(device)
        else:
            self._init_hf_pipeline(device)

    def _init_hf_pipeline(self, device: int):
        print(f"[INFO] 加载 HF 演示模型: {self.model_name}")
        print(f"[INFO] 使用设备: {'GPU' if device >= 0 else 'CPU'}")
        self.pipe = pipeline(
            "object-detection",
            model=self.model_name,
            device=device
        )
        self.model_type = "detr"
        self.torch_device = torch.device('cuda' if device >= 0 and torch.cuda.is_available() else 'cpu')
        print("[INFO] HF 模型加载完成")

    def _init_local_model(self, device: int):
        if not self.config_path or not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {self.checkpoint_path}")

        print(f"[INFO] 加载本地配置: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_type = self.config.get('model', {}).get('type', 'detr').lower()
        self.torch_device = torch.device('cuda' if device >= 0 and torch.cuda.is_available() else 'cpu')
        self.config.setdefault('device', {})
        self.config['device']['type'] = 'cuda' if self.torch_device.type == 'cuda' else 'cpu'
        print(f"[INFO] 加载本地模型: {self.model_type}")
        print(f"[INFO] 使用设备: {self.torch_device}")

        self.model = build_model(self.config).to(self.torch_device)
        load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.model,
            device=str(self.torch_device),
            restore_rng_state=False,
        )
        self.model.eval()

        if self.model_type == 'detr':
            self.image_processor = build_image_processor(self.config)
            self.deformable_transform = None
        elif self.model_type in ('deformable_detr', 'deformable-detr'):
            from dataset.deformable_dataset import make_deformable_transforms
            self.image_processor = None
            self.deformable_transform = make_deformable_transforms('val', self.config)
        else:
            raise ValueError(f"不支持的本地模型类型: {self.model_type}")

        if not self.id_to_label:
            num_classes = int(self.config.get('dataset', {}).get('num_classes', 0))
            self.id_to_label = {idx: str(idx) for idx in range(num_classes)}

        print("[INFO] 本地 checkpoint 加载完成")

    @staticmethod
    def _load_class_mappings(yaml_path: Path) -> Tuple[Dict[str, str], Dict[int, str]]:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        coarse_classes = config.get('COARSE_CLASSES', {})
        id_to_label = {int(k): str(v) for k, v in coarse_classes.items()}

        coarse_mapping: Dict[str, str] = {}
        for mapping_key in ('BDD100K_MAPPING', 'CCTSDB_MAPPING'):
            for src, dst in config.get(mapping_key, {}).items():
                coarse_mapping[str(src).lower()] = str(dst)

        tt_target = config.get('TT100K_TARGET')
        if tt_target:
            coarse_mapping['stop sign'] = str(tt_target)
            coarse_mapping['traffic sign'] = str(tt_target)

        return coarse_mapping, id_to_label

    def _map_label(self, raw_label: str) -> str:
        normalized = raw_label.strip().lower()
        if normalized in self.coarse_mapping:
            return self.coarse_mapping[normalized]
        if 'sign' in normalized:
            return 'traffic_sign'
        return raw_label

    def _filter_and_format(self, detections: List[Dict]) -> List[Dict]:
        formatted = []
        for det in detections:
            label = det["label"]
            if self.keep_labels and label not in self.keep_labels:
                continue
            formatted.append(det)
        return formatted

    def _infer_single_hf(self, image: Image.Image) -> List[Dict]:
        raw_results = self.pipe(image, threshold=self.confidence_threshold)
        detections = []
        for result in raw_results:
            mapped_label = self._map_label(result["label"])
            box = result["box"]
            detections.append({
                "label": mapped_label,
                "score": float(result["score"]),
                "box": [float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])],
            })
        return self._filter_and_format(detections)

    def _infer_single_local_detr(self, image: Image.Image) -> List[Dict]:
        encoding = self.image_processor(images=[image], return_tensors='pt')
        pixel_values = encoding['pixel_values'].to(self.torch_device)
        pixel_mask = encoding.get('pixel_mask')
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.torch_device)

        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = torch.tensor([[image.height, image.width]], device=self.torch_device)
        processed = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label, box in zip(processed['scores'], processed['labels'], processed['boxes']):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "label": self.id_to_label.get(int(label.item()), str(int(label.item()))),
                "score": float(score.item()),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })
        return self._filter_and_format(detections)

    def _infer_single_local_deformable(self, image: Image.Image) -> List[Dict]:
        target = {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64),
            'image_id': torch.tensor([0], dtype=torch.int64),
            'area': torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((0,), dtype=torch.int64),
            'orig_size': torch.as_tensor([image.height, image.width]),
            'size': torch.as_tensor([image.height, image.width]),
        }
        image_tensor, _ = self.deformable_transform(image.copy(), target)
        image_tensor = image_tensor.to(self.torch_device)

        with torch.no_grad():
            outputs = self.model([image_tensor])

        target_sizes = torch.tensor([[image.height, image.width]], device=self.torch_device)
        processed = self.model.postprocess(outputs, target_sizes)[0]

        detections = []
        for score, label, box in zip(processed['scores'], processed['labels'], processed['boxes']):
            if float(score.item()) < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "label": self.id_to_label.get(int(label.item()), str(int(label.item()))),
                "score": float(score.item()),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            })
        return self._filter_and_format(detections)

    def infer_single(self, image_path: str) -> Tuple[List[Dict], Image.Image]:
        image = Image.open(image_path).convert("RGB")

        if self.mode == "hf":
            detections = self._infer_single_hf(image)
        elif self.model_type == 'detr':
            detections = self._infer_single_local_detr(image)
        else:
            detections = self._infer_single_local_deformable(image)

        return detections, image

    def visualize(
        self,
        image: Image.Image,
        detections: List[Dict],
        output_path: str = None,
        show_labels: bool = True,
        font_size: int = 20
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for det in detections:
            label = det["label"]
            score = det["score"]
            x1, y1, x2, y2 = det["box"]

            color = TRAFFIC_COLORS.get(label, TRAFFIC_COLORS["default"])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            if show_labels:
                text = f"{label}: {score:.2f}"
                bbox = draw.textbbox((x1, y1), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
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
        extensions: Optional[List[str]] = None
    ):
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_lines: List[str] = []
        if save_log:
            log_dir = Path("outputs/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"inference_{Path(output_dir).name}.log"

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
                detections, image = self.infer_single(str(img_path))
                output_path = output_dir / f"{img_path.stem}_pred.jpg"
                self.visualize(image, detections, str(output_path))
                all_results[img_path.name] = {
                    "num_detections": len(detections),
                    "detections": detections
                }
                msg = f"  ✓ 检测到 {len(detections)} 个目标"
            except Exception as e:
                msg = f"[ERROR] 处理失败 {img_path.name}: {e}"
            print(msg)
            if save_log:
                log_lines.append(msg)

        if save_json:
            json_path = output_dir / "detections.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 保存检测结果: {json_path}")

        if save_log:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_lines))
            print(f"[INFO] 保存日志: {log_file}")

        print(f"[INFO] 批量推理完成，共处理 {len(all_results)} 张图片")


def main():
    parser = argparse.ArgumentParser(description="交通场景模型推理")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, help="图片目录（批量推理）")
    parser.add_argument("--output_dir", type=str, default="outputs/demo_pred", help="输出目录")
    parser.add_argument("--model", type=str, default="facebook/detr-resnet-50", help="HF 模型名称（仅 HF 模式）")
    parser.add_argument("--config", type=str, help="本地模型配置文件（提供后启用本地模型模式）")
    parser.add_argument("--checkpoint", type=str, help="本地模型 checkpoint 路径")
    parser.add_argument("--threshold", type=float, default=0.8, help="置信度阈值")
    parser.add_argument("--device", type=int, default=-1, help="设备 ID (-1=CPU, 0=GPU0, ...)")
    parser.add_argument("--keep_labels", type=str, nargs="+", help="保留的类别列表")
    parser.add_argument("--classes_yaml", type=str, default="configs/classes.yaml", help="类别映射文件")
    parser.add_argument("--no_json", action="store_true", help="不保存 JSON 结果文件")
    parser.add_argument("--no_log", action="store_true", help="不保存日志文件")

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("必须指定 --image 或 --image_dir")

        inferencer = PretrainedDETRInference(
            model_name=args.model,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            confidence_threshold=args.threshold,
            keep_labels=args.keep_labels,
            classes_yaml=args.classes_yaml,
        )

    if args.image:
        print(f"[INFO] 单张推理模式: {args.image}")
        detections, image = inferencer.infer_single(args.image)
        output_path = Path(args.output_dir) / f"{Path(args.image).stem}_pred.jpg"
        inferencer.visualize(image, detections, str(output_path))
        print(f"\n检测结果 ({len(detections)} 个目标):")
        for det in detections:
            print(f"  - {det['label']}: {det['score']:.3f}")
    else:
        print(f"[INFO] 批量推理模式: {args.image_dir}")
        inferencer.infer_batch(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            save_json=not args.no_json,
            save_log=not args.no_log
        )


if __name__ == "__main__":
    main()
