"""
验证可变尺寸修复
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("\n" + "="*60)
    print("🔍 验证可变尺寸修复")
    print("="*60 + "\n")
    
    all_pass = True
    
    # 检查train_detr.py
    print("1. 检查 train_detr.py")
    print("-" * 60)
    
    train_file = project_root / 'tools' / 'train_detr.py'
    with open(train_file) as f:
        train_content = f.read()
    
    train_checks = [
        ('from transformers import DetrImageProcessor', '✓ 导入DetrImageProcessor'),
        ('image_processor = DetrImageProcessor.from_pretrained', '✓ 初始化ImageProcessor'),
        ('def train_one_epoch(\n    model,\n    dataloader,\n    optimizer,\n    device,\n    epoch,\n    image_processor,', '✓ train_one_epoch函数签名包含image_processor'),
        ('encoding = image_processor(', '✓ 使用ImageProcessor处理图像'),
        ('pixel_values = encoding[\'pixel_values\']', '✓ 提取pixel_values'),
        ('pixel_mask = encoding[\'pixel_mask\']', '✓ 提取pixel_mask'),
        ("val_ann_file = root_dir / config['dataset']['val_ann']", '✓ 使用正确的配置键val_ann'),
        ('image_processor=image_processor,', '✓ 传递image_processor到train_one_epoch'),
    ]
    
    for check, msg in train_checks:
        if check in train_content:
            print(msg)
        else:
            print(f"❌ 缺少: {msg}")
            all_pass = False
    
    print()
    
    # 检查eval_detr.py
    print("2. 检查 eval_detr.py")
    print("-" * 60)
    
    eval_file = project_root / 'tools' / 'eval_detr.py'
    with open(eval_file) as f:
        eval_content = f.read()
    
    eval_checks = [
        ('from transformers import DetrImageProcessor', '✓ 导入DetrImageProcessor'),
        ('def evaluate(model, dataloader, device, coco_gt, logger, score_threshold=0.05, image_processor=None):', '✓ evaluate函数签名包含image_processor和score_threshold'),
        ('if image_processor is None:', '✓ ImageProcessor默认初始化'),
        ('encoding = image_processor(', '✓ 使用ImageProcessor处理图像'),
        ('pixel_values = encoding[\'pixel_values\']', '✓ 提取pixel_values'),
        ('pixel_mask = encoding[\'pixel_mask\']', '✓ 提取pixel_mask'),
        ('keep = max_scores > score_threshold', '✓ 使用score_threshold参数而非硬编码'),
    ]
    
    for check, msg in eval_checks:
        if check in eval_content:
            print(msg)
        else:
            print(f"❌ 缺少: {msg}")
            all_pass = False
    
    print()
    
    # 检查不应该存在的内容
    print("3. 检查已移除的错误代码")
    print("-" * 60)
    
    bad_patterns = [
        ('torch.stack(images)', 'train_detr.py', '❌ 仍使用torch.stack而非ImageProcessor'),
        ("config['dataset']['val_ann_file']", 'train_detr.py', '❌ 仍使用错误的配置键val_ann_file'),
        ('keep = max_scores > 0.05', 'eval_detr.py', '❌ 仍硬编码0.05而非使用参数'),
    ]
    
    for pattern, filename, msg in bad_patterns:
        content = train_content if 'train' in filename else eval_content
        if pattern in content:
            print(msg)
            all_pass = False
        else:
            print(f"✓ 已移除: {pattern}")
    
    print()
    
    print("="*60)
    if all_pass:
        print("🎉 所有可变尺寸修复验证通过!")
        print("="*60)
        print("\n关键改进:")
        print("1. ✅ 使用DetrImageProcessor自动处理padding和pixel_mask")
        print("2. ✅ 支持真正的可变尺寸图像（不会因torch.stack失败）")
        print("3. ✅ 修复配置键名错误（val_ann_file → val_ann）")
        print("4. ✅ 评估阈值参数真正生效（不再硬编码0.05）")
        print("\n下一步:")
        print("  python tools/verify_fixes.py  # 运行完整验证")
    else:
        print("❌ 部分验证失败，请检查上述错误")
        print("="*60)
    print()
    
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
