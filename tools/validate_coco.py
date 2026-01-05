#!/usr/bin/env python3
"""
COCO格式数据集验证脚本
使用pycocotools验证转换后的数据集是否正确
"""

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO


def validate_coco_dataset(ann_file: str) -> dict:
    """
    验证COCO格式数据集
    
    Args:
        ann_file: COCO格式标注文件路径
        
    Returns:
        验证统计信息
    """
    print(f"📂 加载标注文件: {ann_file}")
    
    # 加载COCO数据
    coco = COCO(ann_file)
    
    # 获取类别信息
    cats = coco.loadCats(coco.getCatIds())
    cat_dict = {c['id']: c['name'] for c in cats}
    
    # 统计信息
    stats = {
        "annotation_file": ann_file,
        "num_images": len(coco.imgs),
        "num_annotations": len(coco.anns),
        "num_categories": len(cats),
        "categories": cat_dict,
        "category_counts": {},
    }
    
    # 统计每个类别的标注数
    for cat_id in cat_dict.keys():
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        stats["category_counts"][cat_dict[cat_id]] = len(ann_ids)
    
    return stats


def print_validation_report(stats: dict) -> None:
    """打印验证报告"""
    print("\n" + "="*60)
    print("📊 COCO数据集验证报告")
    print("="*60)
    print(f"\n文件: {stats['annotation_file']}")
    print(f"\n总图像数: {stats['num_images']:,}")
    print(f"总标注数: {stats['num_annotations']:,}")
    print(f"类别数量: {stats['num_categories']}")
    
    print(f"\n类别映射:")
    for cat_id, cat_name in sorted(stats['categories'].items()):
        print(f"  [{cat_id}] {cat_name}")
    
    print(f"\n类别分布:")
    for cat_name, count in sorted(stats['category_counts'].items()):
        percentage = (count / stats['num_annotations'] * 100) if stats['num_annotations'] > 0 else 0
        print(f"  {cat_name:20s}: {count:7,} ({percentage:5.2f}%)")
    
    # 数据完整性检查
    print(f"\n✅ 数据完整性检查:")
    checks = []
    
    if stats['num_images'] > 0:
        checks.append("✓ 包含图像数据")
    else:
        checks.append("✗ 无图像数据")
    
    if stats['num_annotations'] > 0:
        checks.append("✓ 包含标注数据")
    else:
        checks.append("✗ 无标注数据")
    
    if stats['num_categories'] > 0:
        checks.append("✓ 包含类别定义")
    else:
        checks.append("✗ 无类别定义")
    
    # 检查类别ID连续性
    cat_ids = sorted(stats['categories'].keys())
    if cat_ids == list(range(len(cat_ids))):
        checks.append("✓ 类别ID连续（从0开始）")
    else:
        checks.append("⚠ 类别ID不连续")
    
    for check in checks:
        print(f"  {check}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="验证COCO格式数据集"
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        required=True,
        help="COCO标注文件路径",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="检查图像文件是否存在",
    )
    
    args = parser.parse_args()
    
    # 验证数据集
    stats = validate_coco_dataset(args.ann_file)
    
    # 打印报告
    print_validation_report(stats)
    
    # 检查图像文件（可选）
    if args.check_images:
        print("\n🔍 检查图像文件...")
        ann_path = Path(args.ann_file)
        img_dir = ann_path.parent.parent / "images" / ann_path.stem.replace("instances_", "")
        
        if not img_dir.exists():
            print(f"⚠️  图像目录不存在: {img_dir}")
        else:
            with open(args.ann_file, 'r') as f:
                data = json.load(f)
            
            missing = 0
            for img in data['images'][:100]:  # 检查前100张
                img_file = img_dir / img['file_name']
                if not img_file.exists():
                    print(f"  ✗ 缺失: {img['file_name']}")
                    missing += 1
            
            if missing == 0:
                print(f"  ✓ 前100张图像文件完整")
            else:
                print(f"  ⚠️  发现 {missing} 个缺失文件")


if __name__ == "__main__":
    main()
