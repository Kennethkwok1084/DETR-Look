#!/usr/bin/env python3
"""
BDD100K 到 COCO 格式转换脚本

配置职责分工（配置驱动设计）：
- 类别映射定义：从 configs/classes.yaml 读取（配置驱动）
  · COARSE_CLASSES: 粗粒度类别定义（ID -> 名称）
  · BDD100K_MAPPING: BDD100K原始类别到粗粒度类别的映射
  · MAPPING_OPTIONS: 可选映射开关（bike/motor并入）
- 映射记录：转换后输出 mapping.json，记录配置快照和统计信息
- 验证机制：启动时校验配置合法性（ID连续性、映射完整性）

注意：修改类别映射只需修改 configs/classes.yaml
"""

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm


def load_classes_config(config_path: str) -> Dict:
    """加载并验证类别配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 验证必需字段
    required_fields = ['COARSE_CLASSES', 'MAPPING_OPTIONS']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"配置文件缺少必需字段: {field}")
    
    return config


def validate_common_config(config: Dict) -> Tuple[Dict[str, int], Dict[str, bool]]:
    """
    验证通用配置合法性并返回 coarse_id

    Returns:
        coarse_id: {class_name: class_id} 映射
        mapping_options: 映射选项
    """
    coarse_classes = config['COARSE_CLASSES']
    mapping_options = config['MAPPING_OPTIONS']

    # 验证1: COARSE_CLASSES的ID必须从0开始连续
    class_ids = sorted(coarse_classes.keys())
    expected_ids = list(range(len(class_ids)))
    if class_ids != expected_ids:
        raise ValueError(
            f"COARSE_CLASSES的ID必须从0开始连续\n"
            f"  期望: {expected_ids}\n"
            f"  实际: {class_ids}"
        )

    # 生成 coarse_id 映射 {name: id}
    coarse_id = {name: idx for idx, name in coarse_classes.items()}

    return coarse_id, mapping_options


def validate_mapping(mapping: Dict[str, str], coarse_id: Dict[str, int], name: str) -> Dict[str, str]:
    """验证映射字典的value是否存在于COARSE_CLASSES中"""
    invalid_mappings = []
    for original_class, coarse_class in mapping.items():
        if coarse_class not in coarse_id:
            invalid_mappings.append(f"  '{original_class}' -> '{coarse_class}' (不存在)")

    if invalid_mappings:
        raise ValueError(
            f"{name}包含无效的粗粒度类别:\n" + "\n".join(invalid_mappings)
        )
    return dict(mapping)


def build_bdd_class_map(
    config: Dict,
    coarse_id: Dict[str, int],
    mapping_options: Dict[str, bool],
) -> Dict[str, str]:
    """构建BDD100K类别映射"""
    bdd_mapping = config.get('BDD100K_MAPPING')
    if not bdd_mapping:
        raise ValueError("配置文件缺少 BDD100K_MAPPING")
    bdd_class_map = validate_mapping(bdd_mapping, coarse_id, "BDD100K_MAPPING")

    # 根据MAPPING_OPTIONS动态添加bike/motor映射
    if mapping_options.get('include_bike', False):
        bike_target = mapping_options.get('bike_target', 'vehicle')
        if bike_target not in coarse_id:
            raise ValueError(
                f"MAPPING_OPTIONS.bike_target='{bike_target}' 不存在于COARSE_CLASSES中"
            )
        bdd_class_map['bike'] = bike_target

    if mapping_options.get('include_motor', False):
        motor_target = mapping_options.get('motor_target', 'vehicle')
        if motor_target not in coarse_id:
            raise ValueError(
                f"MAPPING_OPTIONS.motor_target='{motor_target}' 不存在于COARSE_CLASSES中"
            )
        bdd_class_map['motor'] = motor_target

    return bdd_class_map


def build_cctsdb_class_map(config: Dict, coarse_id: Dict[str, int]) -> Dict[str, str]:
    """构建CCTSDB类别映射"""
    cctsdb_mapping = config.get('CCTSDB_MAPPING')
    if not cctsdb_mapping:
        raise ValueError("配置文件缺少 CCTSDB_MAPPING")
    return validate_mapping(cctsdb_mapping, coarse_id, "CCTSDB_MAPPING")


def resolve_tt100k_target(config: Dict, coarse_id: Dict[str, int]) -> str:
    """解析TT100K的目标粗粒度类别"""
    target = config.get('TT100K_TARGET', 'traffic_sign')
    if target not in coarse_id:
        raise ValueError(f"TT100K_TARGET='{target}' 不存在于COARSE_CLASSES中")
    return target


def print_config_summary(coarse_id: Dict[str, int], class_map: Dict[str, str]) -> None:
    """打印配置摘要"""
    print("\n" + "="*60)
    print("📋 配置摘要")
    print("="*60)
    
    print("\n粗粒度类别:")
    for name, idx in sorted(coarse_id.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {name}")
    
    print(f"\n映射规则 ({len(class_map)} 个):")
    for original, coarse in sorted(class_map.items()):
        class_id = coarse_id[coarse]
        print(f"  '{original}' -> '{coarse}' (ID: {class_id})")
    
    print("="*60 + "\n")


def print_tt100k_summary(coarse_id: Dict[str, int], target: str) -> None:
    """打印TT100K配置摘要"""
    print("\n" + "="*60)
    print("📋 配置摘要")
    print("="*60)

    print("\n粗粒度类别:")
    for name, idx in sorted(coarse_id.items(), key=lambda x: x[1]):
        print(f"  [{idx}] {name}")

    print(f"\nTT100K目标类别: '{target}' (ID: {coarse_id[target]})")
    print("="*60 + "\n")


def resolve_image_name(img_src_dir: Path, name: str) -> str:
    """根据实际文件补全图片后缀（优先jpg，其次png）"""
    candidate = img_src_dir / name
    if candidate.exists():
        return name
    if Path(name).suffix:
        return name
    for ext in [".jpg", ".png"]:
        candidate = img_src_dir / f"{name}{ext}"
        if candidate.exists():
            return f"{name}{ext}"
    return name


def load_per_image_labels(label_dir: Path) -> List[Dict]:
    """加载单图JSON标注并转换为统一格式"""
    json_files = sorted(label_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"标注目录为空: {label_dir}")

    annotations: List[Dict] = []
    multi_frame = 0
    for label_path in tqdm(json_files, desc=f"Loading {label_dir.name} labels"):
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        name = data.get("name", label_path.stem)
        frames = data.get("frames", [])
        if len(frames) > 1:
            multi_frame += 1

        labels: List[Dict] = []
        if frames:
            # 单图标注通常只有一帧，默认取第一帧
            for obj in frames[0].get("objects", []):
                if "box2d" not in obj:
                    continue
                labels.append({
                    "category": obj.get("category"),
                    "box2d": obj.get("box2d"),
                })

        annotations.append({
            "name": name,
            "labels": labels,
        })

    if multi_frame > 0:
        print(f"⚠️  检测到 {multi_frame} 个多帧标注文件，已默认使用首帧")

    return annotations


def get_image_size(image_path: Path) -> Tuple[int, int]:
    """读取图片尺寸（依赖opencv-python）"""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("读取图片尺寸需要安装 opencv-python") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    height, width = image.shape[:2]
    return width, height


def parse_cctsdb_xml(xml_path: Path) -> Dict:
    """解析CCTSDB XML标注"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename") or xml_path.stem
    size_node = root.find("size")
    width = int(size_node.findtext("width")) if size_node is not None else 0
    height = int(size_node.findtext("height")) if size_node is not None else 0

    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")
        if box is None:
            continue
        labels.append({
            "category": name,
            "box2d": {
                "x1": float(box.findtext("xmin")),
                "y1": float(box.findtext("ymin")),
                "x2": float(box.findtext("xmax")),
                "y2": float(box.findtext("ymax")),
            },
        })

    return {
        "name": filename,
        "width": width,
        "height": height,
        "labels": labels,
    }

def convert_bdd_to_coco(
    src_dir: str,
    dst_dir: str,
    split: str,
    coarse_id: Dict[str, int],
    class_map: Dict[str, str],
    min_area: float = 0.0,
) -> Tuple[Dict, Dict]:
    """
    将BDD100K格式转换为COCO格式
    
    Args:
        src_dir: BDD100K数据集根目录
        dst_dir: 输出COCO格式数据集目录
        split: 数据集划分 (train/val/test)
        coarse_id: 粗粒度类别ID映射 {class_name: class_id}
        class_map: BDD类别映射 {original_class: coarse_class}
        min_area: 最小bbox面积过滤阈值
        
    Returns:
        coco_dict: COCO格式的标注字典
        stats: 转换统计信息
    """
    # 路径设置（兼容多种图像目录结构）
    img_src_candidates = [
        Path(src_dir) / "images" / "100k" / split,
        Path(src_dir) / "images" / split,
        Path(src_dir) / split,
    ]
    img_src_dir = next((p for p in img_src_candidates if p.exists()), img_src_candidates[0])
    
    # BDD100K标注文件路径fallback（支持多种官方格式）
    label_candidates = [
        Path(src_dir) / "labels" / f"det_{split}.json",              # 旧版: det_train.json
        Path(src_dir) / "labels" / "det_20" / f"det_{split}.json",   # 新版子目录: det_20/det_train.json
        Path(src_dir) / "labels" / f"det_20_{split}.json",           # 扁平命名: det_20_train.json
    ]

    # 单图JSON标注路径（逐图文件）
    label_dir_candidates = [
        Path(src_dir) / "labels" / "bdd100k" / split,
        Path(src_dir) / "labels" / "bd100k" / split,
    ]

    label_src_file = next((p for p in label_candidates if p.exists()), None)
    label_src_dir = next((p for p in label_dir_candidates if p.exists()), None)

    img_dst_dir = Path(dst_dir) / "images" / split
    ann_dst_dir = Path(dst_dir) / "annotations"
    
    # 创建输出目录
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    ann_dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查源文件是否存在
    if label_src_file is None and label_src_dir is None:
        raise FileNotFoundError(
            f"标注文件不存在: {label_candidates[0]}，且未找到逐图标注目录"
        )
    if not img_src_dir.exists():
        raise FileNotFoundError(
            f"图像目录不存在: {img_src_candidates[0]}，且未找到可用的图像目录结构"
        )
    
    # 加载BDD100K标注
    if label_src_file is not None:
        print(f"📂 加载 {split} 集标注: {label_src_file}")
        with open(label_src_file, 'r') as f:
            bdd_annotations = json.load(f)
    else:
        print(f"📂 加载 {split} 集逐图标注: {label_src_dir}")
        bdd_annotations = load_per_image_labels(label_src_dir)
    
    # 初始化COCO格式
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    # 创建类别列表
    for class_name, class_id in coarse_id.items():
        coco_dict["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "traffic",
        })
    
    # 统计信息
    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "class_counts": defaultdict(int),
        "original_class_counts": defaultdict(int),
        "filtered_annotations": 0,
        "unmapped_classes": set(),
    }
    
    annotation_id = 0
    
    # 处理每张图像
    print(f"🔄 转换 {split} 集...")
    for img_idx, img_data in enumerate(tqdm(bdd_annotations, desc=f"Processing {split}")):
        img_name = resolve_image_name(img_src_dir, img_data["name"])
        
        # 添加图像信息
        coco_dict["images"].append({
            "id": img_idx,
            "file_name": img_name,
            "width": 1280,  # BDD100K默认分辨率
            "height": 720,
        })
        stats["total_images"] += 1
        
        # 复制图像文件
        src_img = img_src_dir / img_name
        dst_img = img_dst_dir / img_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)
        
        # 处理标注
        if "labels" not in img_data:
            continue
            
        for label in img_data["labels"]:
            if "box2d" not in label:
                continue
            
            original_category = label["category"]
            stats["original_class_counts"][original_category] += 1
            
            # 类别映射
            if original_category not in class_map:
                stats["unmapped_classes"].add(original_category)
                continue
            
            coarse_category = class_map[original_category]
            category_id = coarse_id[coarse_category]
            
            # 提取bbox
            box = label["box2d"]
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]
            w, h = x2 - x1, y2 - y1
            
            # 面积过滤
            area = w * h
            if area < min_area:
                stats["filtered_annotations"] += 1
                continue
            
            # 添加标注
            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": img_idx,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0,
            })
            
            stats["total_annotations"] += 1
            stats["class_counts"][coarse_category] += 1
            annotation_id += 1
    
    # 保存COCO格式标注
    ann_file = ann_dst_dir / f"instances_{split}.json"
    print(f"💾 保存标注文件: {ann_file}")
    with open(ann_file, 'w') as f:
        json.dump(coco_dict, f)
    
    return coco_dict, stats


def convert_cctsdb_to_coco(
    src_dir: str,
    dst_dir: str,
    split: str,
    coarse_id: Dict[str, int],
    class_map: Dict[str, str],
    min_area: float = 0.0,
) -> Tuple[Dict, Dict]:
    """将CCTSDB XML标注转换为COCO格式"""
    img_src_candidates = [
        Path(src_dir) / "images" / split,
        Path(src_dir) / split,
    ]
    img_src_dir = next((p for p in img_src_candidates if p.exists()), img_src_candidates[0])

    label_dir_candidates = [
        Path(src_dir) / "labels" / "xml" / split,
        Path(src_dir) / "labels" / "xml",
        Path(src_dir) / "xml" / split,
        Path(src_dir) / "xml",
        Path(src_dir) / "labels" / split,
    ]
    label_src_dir = next((p for p in label_dir_candidates if p.exists()), None)

    img_dst_dir = Path(dst_dir) / "images" / split
    ann_dst_dir = Path(dst_dir) / "annotations"
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    ann_dst_dir.mkdir(parents=True, exist_ok=True)

    if not img_src_dir.exists():
        raise FileNotFoundError(
            f"图像目录不存在: {img_src_candidates[0]}，且未找到可用的图像目录结构"
        )
    if label_src_dir is None:
        raise FileNotFoundError("未找到CCTSDB XML标注目录")

    image_files = {p.stem: p.name for p in img_src_dir.glob("*.*")}

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for class_name, class_id in coarse_id.items():
        coco_dict["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "traffic",
        })

    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "class_counts": defaultdict(int),
        "original_class_counts": defaultdict(int),
        "filtered_annotations": 0,
        "unmapped_classes": set(),
    }
    annotation_id = 0

    xml_files = sorted(label_src_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"标注目录为空: {label_src_dir}")

    print(f"📂 加载 {split} 集XML标注: {label_src_dir}")
    for img_idx, xml_path in enumerate(tqdm(xml_files, desc=f"Processing {split}")):
        data = parse_cctsdb_xml(xml_path)
        img_name = resolve_image_name(img_src_dir, data["name"])
        if Path(img_name).stem not in image_files:
            continue

        width = data.get("width", 0)
        height = data.get("height", 0)
        if width == 0 or height == 0:
            width, height = get_image_size(img_src_dir / img_name)

        coco_dict["images"].append({
            "id": img_idx,
            "file_name": img_name,
            "width": width,
            "height": height,
        })
        stats["total_images"] += 1

        src_img = img_src_dir / img_name
        dst_img = img_dst_dir / img_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        for label in data.get("labels", []):
            original_category = label["category"]
            stats["original_class_counts"][original_category] += 1
            if original_category not in class_map:
                stats["unmapped_classes"].add(original_category)
                continue

            coarse_category = class_map[original_category]
            category_id = coarse_id[coarse_category]
            box = label["box2d"]
            x1, y1 = box["x1"], box["y1"]
            x2, y2 = box["x2"], box["y2"]
            w, h = x2 - x1, y2 - y1
            area = w * h
            if area < min_area:
                stats["filtered_annotations"] += 1
                continue

            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": img_idx,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0,
            })
            stats["total_annotations"] += 1
            stats["class_counts"][coarse_category] += 1
            annotation_id += 1

    ann_file = ann_dst_dir / f"instances_{split}.json"
    print(f"💾 保存标注文件: {ann_file}")
    with open(ann_file, 'w') as f:
        json.dump(coco_dict, f)

    return coco_dict, stats


def convert_tt100k_to_coco(
    src_dir: str,
    dst_dir: str,
    split: str,
    coarse_id: Dict[str, int],
    target_class: str,
    min_area: float = 0.0,
) -> Tuple[Dict, Dict]:
    """将TT100K官方标注转换为COCO格式"""
    ann_file = Path(src_dir) / "annotations_all.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"TT100K标注文件不存在: {ann_file}")

    with open(ann_file, 'r', encoding='utf-8') as f:
        ann_data = json.load(f)

    imgs = ann_data.get("imgs", {})
    img_dst_dir = Path(dst_dir) / "images" / split
    ann_dst_dir = Path(dst_dir) / "annotations"
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    ann_dst_dir.mkdir(parents=True, exist_ok=True)

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for class_name, class_id in coarse_id.items():
        coco_dict["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "traffic",
        })

    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "class_counts": defaultdict(int),
        "original_class_counts": defaultdict(int),
        "filtered_annotations": 0,
        "unmapped_classes": set(),
    }
    annotation_id = 0
    image_id = 0

    split_prefix = f"{split}/"
    # 使用数值排序确保 image_id 按图像 ID 顺序生成（而非字符串字典序）
    # 添加安全兜底：无法转 int 时退回字符串排序
    def safe_numeric_sort(item):
        try:
            return (0, int(item[0]))  # (优先级, 数值)
        except (ValueError, TypeError):
            return (1, item[0])  # (优先级, 字符串) - 非数字key放后面
    
    sorted_imgs = sorted(imgs.items(), key=safe_numeric_sort)
    for img_key, img_info in tqdm(sorted_imgs, desc=f"Processing {split}"):
        path = img_info.get("path", "")
        if not path.startswith(split_prefix):
            continue

        img_path = Path(src_dir) / path
        if not img_path.exists():
            continue

        width = img_info.get("width")
        height = img_info.get("height")
        if width is None or height is None:
            width, height = get_image_size(img_path)

        file_name = Path(path).name
        coco_dict["images"].append({
            "id": image_id,
            "file_name": file_name,
            "width": int(width),
            "height": int(height),
        })
        stats["total_images"] += 1

        dst_img = img_dst_dir / file_name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        for obj in img_info.get("objects", []):
            bbox = obj.get("bbox")
            if not bbox:
                continue
            original_category = obj.get("category", "unknown")
            stats["original_class_counts"][original_category] += 1

            x1, y1 = float(bbox["xmin"]), float(bbox["ymin"])
            x2, y2 = float(bbox["xmax"]), float(bbox["ymax"])
            w, h = x2 - x1, y2 - y1
            area = w * h
            if area < min_area:
                stats["filtered_annotations"] += 1
                continue

            coco_dict["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": coarse_id[target_class],
                "bbox": [x1, y1, w, h],
                "area": area,
                "iscrowd": 0,
            })
            stats["total_annotations"] += 1
            stats["class_counts"][target_class] += 1
            annotation_id += 1
        
        image_id += 1

    ann_out = ann_dst_dir / f"instances_{split}.json"
    print(f"💾 保存标注文件: {ann_out}")
    with open(ann_out, 'w') as f:
        json.dump(coco_dict, f)

    return coco_dict, stats

def save_mapping_info(
    dst_dir: str,
    all_stats: Dict[str, Dict],
    coarse_id: Dict[str, int],
    class_map: Dict[str, str],
    config_path: str,
    config_content: Dict,
    mapping_key: str,
) -> None:
    """保存映射信息和统计摘要"""
    # 将配置转为YAML字符串保存，避免JSON序列化时int key变成字符串
    config_yaml = yaml.dump(config_content, allow_unicode=True, sort_keys=False)
    
    mapping_info = {
        "class_mapping": {
            mapping_key: class_map,
            "coarse_to_id": coarse_id,
        },
        "statistics": {},
        "config_snapshot": {
            "path": config_path,
            "content_yaml": config_yaml,  # YAML原文保持完整结构
            "content_dict": config_content,  # dict便于程序读取
        },
    }
    
    # 汇总各split的统计信息
    for split, stats in all_stats.items():
        mapping_info["statistics"][split] = {
            "total_images": stats["total_images"],
            "total_annotations": stats["total_annotations"],
            "filtered_annotations": stats["filtered_annotations"],
            "class_counts": dict(stats["class_counts"]),
            "original_class_counts": dict(stats["original_class_counts"]),
            "unmapped_classes": list(stats["unmapped_classes"]),
        }
    
    # 保存mapping.json
    mapping_file = Path(dst_dir) / "mapping.json"
    print(f"\n📋 保存映射信息: {mapping_file}")
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 转换摘要")
    print("="*60)
    
    for split, stats in all_stats.items():
        print(f"\n【{split.upper()} 集】")
        print(f"  总图像数: {stats['total_images']:,}")
        print(f"  总标注数: {stats['total_annotations']:,}")
        print(f"  过滤标注数: {stats['filtered_annotations']:,}")
        print(f"  类别分布:")
        for class_name, count in sorted(stats['class_counts'].items()):
            class_id = coarse_id[class_name]
            print(f"    [{class_id}] {class_name}: {count:,}")
        
        if stats['unmapped_classes']:
            print(f"  ⚠️  未映射类别: {', '.join(sorted(stats['unmapped_classes']))}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="将BDD100K数据集转换为COCO格式，支持粗粒度类别映射"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="BDD100K数据集根目录（包含images和labels目录）",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="COCO格式输出目录",
    )
    parser.add_argument(
        "--split-train",
        type=str,
        default="train",
        help="训练集名称（默认: train）",
    )
    parser.add_argument(
        "--split-val",
        type=str,
        default="val",
        help="验证集名称（默认: val）",
    )
    parser.add_argument(
        "--split-test",
        type=str,
        default="test",
        help="测试集名称（默认: test）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/classes.yaml",
        help="类别配置文件路径",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=None,
        help="最小bbox面积阈值（像素），未指定时从配置文件读取",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="要转换的数据集划分",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bdd100k",
        choices=["bdd100k", "cctsdb", "tt100k"],
        help="数据集类型",
    )
    
    args = parser.parse_args()
    
    # 加载并验证配置
    print(f"📖 加载类别配置: {args.config}")
    try:
        config = load_classes_config(args.config)
        coarse_id, mapping_options = validate_common_config(config)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 配置错误: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 从配置读取min_area（如果命令行未显式指定）
    if args.min_area is None:
        args.min_area = mapping_options.get('min_area', 0.0)
    # 否则使用命令行值（包括显式指定的 0.0）
    
    # 打印配置摘要
    if args.dataset == "bdd100k":
        class_map = build_bdd_class_map(config, coarse_id, mapping_options)
        print_config_summary(coarse_id, class_map)
        converter = convert_bdd_to_coco
        mapping_key = "bdd100k_to_coarse"
        target_class = None
    elif args.dataset == "cctsdb":
        class_map = build_cctsdb_class_map(config, coarse_id)
        print_config_summary(coarse_id, class_map)
        converter = convert_cctsdb_to_coco
        mapping_key = "cctsdb_to_coarse"
        target_class = None
    else:
        target_class = resolve_tt100k_target(config, coarse_id)
        class_map = {"__all__": target_class}
        print_tt100k_summary(coarse_id, target_class)
        converter = convert_tt100k_to_coco
        mapping_key = "tt100k_to_coarse"
    
    # 打印转换信息
    print("="*60)
    print(f"🚀 {args.dataset.upper()} → COCO 转换工具")
    print("="*60)
    print(f"源目录: {args.src}")
    print(f"目标目录: {args.dst}")
    print(f"最小面积: {args.min_area} 像素²")
    print(f"转换划分: {', '.join(args.splits)}")
    print("="*60 + "\n")
    
    # 转换各个split
    all_stats = {}
    split_map = {
        "train": args.split_train,
        "val": args.split_val,
        "test": args.split_test,
    }
    
    for split_key in args.splits:
        split_name = split_map[split_key]
        try:
            if args.dataset == "tt100k":
                _, stats = converter(
                    src_dir=args.src,
                    dst_dir=args.dst,
                    split=split_name,
                    coarse_id=coarse_id,
                    target_class=target_class,
                    min_area=args.min_area,
                )
            else:
                _, stats = converter(
                    src_dir=args.src,
                    dst_dir=args.dst,
                    split=split_name,
                    coarse_id=coarse_id,
                    class_map=class_map,
                    min_area=args.min_area,
                )
            all_stats[split_name] = stats
        except FileNotFoundError as e:
            print(f"⚠️  跳过 {split_name} 集: {e}")
            continue
    
    # 保存映射信息和打印摘要
    if all_stats:
        save_mapping_info(
            args.dst,
            all_stats,
            coarse_id,
            class_map,
            args.config,
            config,
            mapping_key,
        )
        print(f"\n✅ 转换完成！输出目录: {args.dst}")
    else:
        print("\n❌ 未成功转换任何数据集")


if __name__ == "__main__":
    main()
