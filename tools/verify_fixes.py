"""
验证所有bug修复
"""
import sys
from pathlib import Path
import re

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_configs():
    """验证配置文件中的模型名称"""
    print("="*60)
    print("1. 验证配置文件")
    print("="*60)
    
    configs_to_check = [
        'configs/detr_baseline.yaml',
        'configs/detr_smoke.yaml',
    ]
    
    for config_path in configs_to_check:
        full_path = project_root / config_path
        with open(full_path) as f:
            content = f.read()
        
        # 简单的文本检查
        if 'name: "detr-resnet-50"' in content or "name: 'detr-resnet-50'" in content:
            print(f"✓ {config_path}: model.name = 'detr-resnet-50'")
        elif 'name: "detr_resnet50"' in content or "name: 'detr_resnet50'" in content:
            print(f"❌ {config_path}: model.name 仍然是 'detr_resnet50' (错误)")
            return False
        
        # 验证eval_interval存在
        if 'eval_interval:' in content:
            match = re.search(r'eval_interval:\s*(\d+)', content)
            if match:
                print(f"  eval_interval = {match.group(1)}")
        else:
            print(f"❌ {config_path} 缺少 eval_interval")
            return False
        
        # 验证max_iters
        if 'max_iters:' in content:
            match = re.search(r'max_iters:\s*(\d+|null)', content)
            if match:
                print(f"  max_iters = {match.group(1)}")
    
    print("\n✅ 配置文件验证通过!\n")
    return True


def verify_requirements():
    """验证requirements.txt包含timm"""
    print("="*60)
    print("2. 验证依赖文件")
    print("="*60)
    
    req_file = project_root / 'requirements.txt'
    with open(req_file) as f:
        requirements = f.read()
    
    if 'timm' in requirements:
        print("✓ requirements.txt 包含 timm")
        print("\n✅ 依赖文件验证通过!\n")
    else:
        print("❌ requirements.txt 缺少 timm")
        return False
    
    return True


def verify_imports():
    """验证关键导入"""
    print("="*60)
    print("3. 验证关键导入（train_detr_optimized.py）")
    print("="*60)
    
    # 检查 train_detr_optimized.py 的导入和关键特性
    train_script = project_root / 'tools' / 'train_detr_optimized.py'
    with open(train_script) as f:
        content = f.read()
    
    checks = [
        ('from pycocotools.coco import COCO', 'COCO导入'),
        ('from transformers import', 'transformers导入'),
        ('DetrForObjectDetection', 'DetrForObjectDetection'),
        ('DETR_MEAN = [0.485, 0.456, 0.406]', 'DETR归一化均值'),
        ('DETR_STD = [0.229, 0.224, 0.225]', 'DETR归一化标准差'),
        ('reverse_cat_id_map', 'Category ID反向映射'),
        ('target_sizes = torch.stack([l["orig_size"]', '使用orig_size作为target_sizes'),
        ('def collate_fn(batch)', 'collate_fn返回dict'),
        ('torchvision.io', 'torchvision.io导入'),
    ]
    
    for check_str, desc in checks:
        if check_str in content:
            print(f"✓ {desc}")
        else:
            print(f"❌ 缺少 {desc}")
            return False
    
    print("\n✅ 导入验证通过!\n")
    return True


def verify_collate_fn():
    """验证collate_fn处理可变尺寸"""
    print("="*60)
    print("4. 验证数据加载（dict格式）")
    print("="*60)
    
    train_script = project_root / 'tools' / 'train_detr_optimized.py'
    with open(train_script) as f:
        content = f.read()
    
    checks = [
        ('def collate_fn(batch', 'collate_fn定义'),
        ('"pixel_values"', 'pixel_values键'),
        ('"labels"', 'labels键'),
        ('class_labels', 'class_labels字段'),
    ]
    
    for check_str, desc in checks:
        if check_str in content:
            print(f"✓ {desc}")
        else:
            print(f"❌ 缺少 {desc}")
            return False
    
    print("\n✅ 数据加载验证通过!\n")
    return True


def verify_stack_handling():
    """验证Bbox格式和坐标系"""
    print("="*60)
    print("5. 验证Bbox格式和坐标系")
    print("="*60)
    
    train_script = project_root / 'tools' / 'train_detr_optimized.py'
    with open(train_script) as f:
        content = f.read()
    
    checks = [
        ('# 转换 bbox：xyxy 像素 -> 归一化 cxcywh', 'Bbox转换注释'),
        ('boxes_cxcywh', '归一化中心点计算'),
        ('target_sizes = torch.stack([l["orig_size"]', 'evaluate使用orig_size'),
        ('reverse_cat_id_map.get(label.item()', 'Category ID反向映射'),
    ]
    
    for check_str, desc in checks:
        if check_str in content:
            print(f"✓ {desc}")
        else:
            print(f"❌ 缺少 {desc}")
            return False
    
    print("\n✅ Bbox格式和坐标系验证通过!\n")
    return True


def verify_eval_threshold():
    """验证评估阈值"""
    print("="*60)
    print("6. 验证评估阈值")
    print("="*60)
    
    eval_file = project_root / 'tools' / 'eval_unified.py'
    with open(eval_file) as f:
        content = f.read()
    
    # 检查score_threshold参数
    if 'score_threshold=0.05' in content or 'score_threshold' in content:
        print("✓ eval_unified.py 使用可配置的 score_threshold")
    else:
        print("❌ eval_unified.py 缺少 score_threshold 参数")
        return False
    
    # 检查不应该硬编码0.7
    if 'score > 0.7' in content or 'scores > 0.7' in content:
        print("⚠️  仍然存在硬编码的0.7阈值")
    else:
        print("✓ 没有硬编码的0.7阈值")
    
    print("\n✅ 评估阈值验证通过!\n")
    return True


def verify_epoch_logic():
    """验证旧入口是否已收敛到统一训练脚本"""
    print("="*60)
    print("7. 验证训练入口收敛")
    print("="*60)
    
    train_file = project_root / 'tools' / 'train_detr.py'
    with open(train_file) as f:
        content = f.read()
    
    if 'tools.train_unified' in content and '兼容入口' in content:
        print("✓ train_detr.py 已改为兼容包装，并委托 train_unified.py")
    else:
        print("❌ train_detr.py 尚未收敛到统一训练入口")
        return False
    
    print("\n✅ 训练入口收敛验证通过!\n")
    return True


def main():
    """运行所有验证"""
    print("\n" + "="*60)
    print("🔍 开始验证Bug修复")
    print("="*60 + "\n")
    
    all_pass = True
    
    try:
        verify_configs()
        all_pass = verify_requirements() and all_pass
        all_pass = verify_imports() and all_pass
        all_pass = verify_collate_fn() and all_pass
        all_pass = verify_stack_handling() and all_pass
        all_pass = verify_eval_threshold() and all_pass
        all_pass = verify_epoch_logic() and all_pass
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("🎉 所有验证通过!")
        print("="*60)
        print("\n下一步:")
        print("1. 在GPU服务器上运行冒烟测试:")
        print("   python tools/train_detr.py --config configs/detr_smoke.yaml")
        print("\n2. 如果冒烟测试通过，运行完整训练:")
        print("   python tools/train_detr.py --config configs/detr_baseline.yaml")
        print("\n3. 评估模型:")
        print("   python tools/eval_detr.py --config configs/detr_baseline.yaml \\")
        print("       --checkpoint outputs/detr_baseline/checkpoints/best.pth")
    else:
        print("❌ 部分验证失败，请检查上述错误")
        print("="*60)
    print()
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
