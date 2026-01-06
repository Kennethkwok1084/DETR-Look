#!/usr/bin/env python3
"""
快速冒烟测试脚本
验证COCO数据集可以正常加载，适合在命令行直接执行
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def quick_smoke_test(ann_file: str):
    """快速冒烟测试"""
    from pycocotools.coco import COCO
    
    print(f"🔥 冒烟测试: {ann_file}\n")
    
    try:
        # 加载COCO数据
        coco = COCO(ann_file)
        
        # 获取统计信息
        cats = coco.loadCats(coco.getCatIds())
        cat_dict = {c['id']: c['name'] for c in cats}
        
        # 打印结果
        print(f"✅ 加载成功!")
        print(f"   图像数: {len(coco.imgs):,}")
        print(f"   标注数: {len(coco.anns):,}")
        print(f"   类别数: {len(cats)}")
        print(f"   类别映射: {cat_dict}")
        
        # 检查每个类别的标注数
        print(f"\n   类别分布:")
        for cat_id, cat_name in sorted(cat_dict.items()):
            ann_ids = coco.getAnnIds(catIds=[cat_id])
            print(f"     [{cat_id}] {cat_name}: {len(ann_ids):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 默认测试验证集
        ann_file = "data/traffic_coco/bdd100k_det/annotations/instances_val.json"
    else:
        ann_file = sys.argv[1]
    
    success = quick_smoke_test(ann_file)
    sys.exit(0 if success else 1)
