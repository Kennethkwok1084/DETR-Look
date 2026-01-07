#!/usr/bin/env python3
"""
完整验证：检查第二轮关键修复
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """检查关键导入"""
    print("=" * 80)
    print("1. 检查导入")
    print("=" * 80)
    
    try:
        from tools.train_detr_optimized import CocoDetrDataset, DETR_MEAN, DETR_STD
        print("✅ train_detr_optimized.py 可导入")
        print(f"   DETR_MEAN: {DETR_MEAN}")
        print(f"   DETR_STD: {DETR_STD}")
    except Exception as e:
        print(f"❌ train_detr_optimized.py 导入失败: {e}")
        return False
    
    try:
        from tools.benchmark_dataloader import benchmark_dataloader
        print("✅ benchmark_dataloader.py 可导入")
    except Exception as e:
        print(f"❌ benchmark_dataloader.py 导入失败: {e}")
        return False
    
    return True


def check_dataset_mapping():
    """检查 Category ID 映射"""
    print("\n" + "=" * 80)
    print("2. 检查 Category ID 映射")
    print("=" * 80)
    
    try:
        from tools.train_detr_optimized import CocoDetrDataset
        from pycocotools.coco import COCO
        
        ann_file = "data/traffic_coco/bdd100k_det/annotations/instances_train.json"
        if not Path(ann_file).exists():
            print(f"⚠️  标注文件不存在: {ann_file}")
            return True  # 跳过，不算失败
        
        coco = COCO(ann_file)
        cat_ids = sorted(coco.getCatIds())
        
        print(f"原始 category_id: {cat_ids}")
        
        # 检查是否连续
        is_continuous = cat_ids == list(range(len(cat_ids)))
        if is_continuous:
            print("✅ Category ID 已经连续，映射为恒等映射")
        else:
            print(f"⚠️  Category ID 不连续，需要映射")
        
        # 创建 dataset 检查映射
        ds = CocoDetrDataset(
            "data/traffic_coco/bdd100k_det/images/train",
            ann_file,
            min_size=800,
            max_size=1333
        )
        
        print(f"映射后 ID: {list(range(ds.num_classes))}")
        print(f"反向映射: {ds.reverse_cat_id_map}")
        
        # 验证反向映射
        for i in range(ds.num_classes):
            original = ds.reverse_cat_id_map[i]
            if original != cat_ids[i]:
                print(f"❌ 反向映射错误: {i} -> {original}, 期望 {cat_ids[i]}")
                return False
        
        print("✅ Category ID 映射正确")
        return True
        
    except Exception as e:
        print(f"❌ Category ID 映射检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_bbox_format():
    """检查 Bbox 格式"""
    print("\n" + "=" * 80)
    print("3. 检查 Bbox 格式")
    print("=" * 80)
    
    try:
        from tools.train_detr_optimized import CocoDetrDataset
        import torch
        
        ann_file = "data/traffic_coco/bdd100k_det/annotations/instances_train.json"
        if not Path(ann_file).exists():
            print(f"⚠️  标注文件不存在: {ann_file}")
            return True
        
        ds = CocoDetrDataset(
            "data/traffic_coco/bdd100k_det/images/train",
            ann_file,
            min_size=800,
            max_size=1333
        )
        
        # 检查第一张图
        img, target = ds[0]
        
        print(f"图像形状: {img.shape}")
        print(f"orig_size: {target['orig_size'].tolist()}")
        print(f"size: {target['size'].tolist()}")
        
        if len(target['boxes']) > 0:
            boxes = target['boxes']
            print(f"Boxes 形状: {boxes.shape}")
            print(f"Boxes 范围: [{boxes.min():.3f}, {boxes.max():.3f}]")
            
            # 检查归一化
            if boxes.min() < 0 or boxes.max() > 1:
                print(f"❌ Boxes 未正确归一化到 [0, 1]")
                return False
            
            print("✅ Boxes 格式正确（归一化 cxcywh）")
        else:
            print("⚠️  第一张图没有标注框")
        
        return True
        
    except Exception as e:
        print(f"❌ Bbox 格式检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_target_sizes():
    """检查 orig_size vs size"""
    print("\n" + "=" * 80)
    print("4. 检查坐标系（orig_size vs size）")
    print("=" * 80)
    
    try:
        from tools.train_detr_optimized import CocoDetrDataset
        
        ann_file = "data/traffic_coco/bdd100k_det/annotations/instances_train.json"
        if not Path(ann_file).exists():
            print(f"⚠️  标注文件不存在: {ann_file}")
            return True
        
        ds = CocoDetrDataset(
            "data/traffic_coco/bdd100k_det/images/train",
            ann_file,
            min_size=800,
            max_size=1333
        )
        
        img, target = ds[0]
        orig_h, orig_w = target['orig_size'].tolist()
        new_h, new_w = target['size'].tolist()
        
        print(f"原始尺寸: {orig_h} x {orig_w}")
        print(f"Resize后: {new_h} x {new_w}")
        
        # 检查是否不同
        if orig_h == new_h and orig_w == new_w:
            print("⚠️  orig_size == size（可能没有 resize）")
        else:
            print("✅ orig_size != size（正确保存了原始尺寸）")
        
        # 检查 evaluate() 函数源码
        import inspect
        from tools.train_detr_optimized import evaluate
        source = inspect.getsource(evaluate)
        
        if "orig_size" in source and "target_sizes" in source:
            # 查找 target_sizes 赋值
            if 'target_sizes = torch.stack([l["orig_size"]' in source:
                print("✅ evaluate() 正确使用 orig_size 作为 target_sizes")
            elif 'target_sizes = torch.stack([l["size"]' in source:
                print("❌ evaluate() 错误使用 size 作为 target_sizes（应该用 orig_size）")
                return False
            else:
                print("⚠️  无法确认 target_sizes 赋值")
        else:
            print("❌ evaluate() 可能未正确使用 orig_size")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 坐标系检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_files():
    """检查文件状态"""
    print("\n" + "=" * 80)
    print("5. 检查文件状态")
    print("=" * 80)
    
    files = {
        "✅ 可用": [
            "tools/train_detr_optimized.py",
            "tools/benchmark_dataloader.py",
            "tools/run_torchvision_training.sh",
            "docs/DETR_TRAINING_GUIDE_CURRENT.md",
            "docs/DETR_TRAINING_README.md",
            "docs/FIXES_2026_01_06.md",
        ],
        "❌ 已标记不可用": [
            "tools/train_detr_torchvision.py.BROKEN",
            "tools/smoke_test_torchvision.py.BROKEN",
        ],
        "📄 已过时": [
            "docs/TORCHVISION_DETR_GUIDE.md.OUTDATED",
            "docs/TORCHVISION_DETR_SUMMARY.md.OUTDATED",
        ]
    }
    
    all_ok = True
    for category, paths in files.items():
        print(f"\n{category}:")
        for path in paths:
            exists = Path(path).exists()
            status = "✓" if exists else "✗"
            print(f"  [{status}] {path}")
            if not exists and "可用" in category:
                all_ok = False
    
    return all_ok


def main():
    """运行所有检查"""
    print("\n🔍 DETR 训练脚本验证")
    print("检查所有关键修复是否正确\n")
    
    checks = [
        ("导入检查", check_imports),
        ("Category ID 映射", check_dataset_mapping),
        ("Bbox 格式", check_bbox_format),
        ("坐标系（orig_size vs size）", check_target_sizes),
        ("文件状态", check_files),
    ]
    
    results = []
    for name, func in checks:
        try:
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 检查异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("验证总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n🎉 所有检查通过！脚本已准备就绪。")
        print("\n下一步：运行冒烟测试")
        print("  python tools/train_detr_optimized.py \\")
        print("    --train-img data/traffic_coco/bdd100k_det/images/train \\")
        print("    --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \\")
        print("    --subset 100 --num-epochs 1 --batch-size 4")
        return 0
    else:
        print("\n⚠️  部分检查失败，请查看上述错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
