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
    print("3. 验证关键导入")
    print("="*60)
    
    # 检查train_detr.py的导入
    train_script = project_root / 'tools' / 'train_detr.py'
    with open(train_script) as f:
        content = f.read()
    
    checks = [
        ('from pycocotools.coco import COCO', 'COCO导入'),
        ('from tools.eval_detr import evaluate', 'evaluate函数导入'),
        ('from transformers import DetrImageProcessor', 'DetrImageProcessor导入'),
        ('eval_interval', 'eval_interval配置'),
        ('coco_gt = COCO(val_ann_file)', 'COCO GT加载'),
        ('val_metrics = evaluate(', '验证调用'),
        ("config['dataset']['val_ann']", '正确的配置键名val_ann'),
        ('image_processor = DetrImageProcessor', 'ImageProcessor初始化'),
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
    print("4. 验证数据加载")
    print("="*60)
    
    dataset_file = project_root / 'dataset' / 'coco_dataset.py'
    with open(dataset_file) as f:
        content = f.read()
    
    if 'def collate_fn(batch' in content and 'return list(images), list(targets)' in content:
        print("✓ collate_fn 返回列表（支持可变尺寸）")
    else:
        print("❌ collate_fn实现有问题")
        return False
    
    print("\n✅ 数据加载验证通过!\n")
    return True


def verify_stack_handling():
    """验证可变尺寸图像处理（使用DetrImageProcessor）"""
    print("="*60)
    print("5. 验证可变尺寸图像处理")
    print("="*60)
    
    files_to_check = [
        ('tools/train_detr.py', 'train_detr.py'),
        ('tools/eval_detr.py', 'eval_detr.py'),
    ]
    
    for file_path, name in files_to_check:
        full_path = project_root / file_path
        with open(full_path) as f:
            content = f.read()
        
        # 检查使用DetrImageProcessor而非torch.stack
        if 'DetrImageProcessor' in content and 'pixel_values' in content and 'pixel_mask' in content:
            print(f"✓ {name} 使用 DetrImageProcessor 处理可变尺寸")
        else:
            print(f"❌ {name} 未正确使用 DetrImageProcessor")
            return False
    
    print("\n✅ 可变尺寸处理验证通过!\n")
    return True


def verify_eval_threshold():
    """验证评估阈值"""
    print("="*60)
    print("6. 验证评估阈值")
    print("="*60)
    
    eval_file = project_root / 'tools' / 'eval_detr.py'
    with open(eval_file) as f:
        content = f.read()
    
    # 检查score_threshold参数
    if 'score_threshold=0.05' in content or 'score_threshold' in content:
        print("✓ eval_detr.py 使用可配置的score_threshold")
    else:
        print("❌ eval_detr.py 缺少score_threshold参数")
        return False
    
    # 检查不应该硬编码0.7
    if 'score > 0.7' in content or 'scores > 0.7' in content:
        print("⚠️  仍然存在硬编码的0.7阈值")
    else:
        print("✓ 没有硬编码的0.7阈值")
    
    print("\n✅ 评估阈值验证通过!\n")
    return True


def verify_epoch_logic():
    """验证epoch停止逻辑"""
    print("="*60)
    print("7. 验证Epoch停止逻辑")
    print("="*60)
    
    train_file = project_root / 'tools' / 'train_detr.py'
    with open(train_file) as f:
        content = f.read()
    
    # 检查改进的停止逻辑
    if 'max_iters <= 200' in content and 'epoch >= 2' in content:
        print("✓ 包含改进的epoch停止逻辑（只在max_iters<=200时2epoch停止）")
    else:
        print("❌ 缺少改进的epoch停止逻辑")
        return False
    
    print("\n✅ Epoch逻辑验证通过!\n")
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
