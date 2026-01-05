#!/usr/bin/env python3
"""
快速测试脚本
验证数据加载、模型构建等基础功能
"""

import sys
from pathlib import Path

import torch
import yaml

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset import build_dataloader
from models import build_detr_model


def test_dataloader(config):
    """测试数据加载器"""
    print("\n" + "="*60)
    print("🧪 测试数据加载器")
    print("="*60)
    
    try:
        # 构建训练集dataloader
        train_loader = build_dataloader(config, 'train', batch_size=2, num_workers=0)
        
        # 取一个batch
        images, targets = next(iter(train_loader))
        
        print(f"✅ 数据加载成功!")
        print(f"   Batch size: {len(images)}")
        print(f"   Image shape: {images[0].shape if len(images) > 0 else 'N/A'}")
        print(f"   Target keys: {targets[0].keys() if len(targets) > 0 else 'N/A'}")
        
        if len(targets) > 0:
            print(f"   第一张图的目标数: {len(targets[0]['boxes'])}")
        
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model(config):
    """测试模型构建"""
    print("\n" + "="*60)
    print("🧪 测试模型构建")
    print("="*60)
    
    try:
        model = build_detr_model(config)
        print(f"✅ 模型构建成功!")
        return True
    except Exception as e:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(config):
    """测试前向传播"""
    print("\n" + "="*60)
    print("🧪 测试前向传播")
    print("="*60)
    
    try:
        # 构建模型
        model = build_detr_model(config)
        model.eval()
        
        # 准备假数据
        batch_size = 2
        images = torch.randn(batch_size, 3, 480, 640)
        
        # 前向传播（推理模式）
        with torch.no_grad():
            outputs = model(images)
        
        print(f"✅ 前向传播成功!")
        print(f"   输出logits shape: {outputs.logits.shape}")
        print(f"   输出boxes shape: {outputs.pred_boxes.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🚀 DETR 训练框架功能测试")
    print("="*60)
    
    # 加载配置
    config_path = project_root / "configs" / "detr_smoke.yaml"
    print(f"\n📖 加载配置: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 运行测试
    results = {
        "数据加载器": test_dataloader(config),
        "模型构建": test_model(config),
        "前向传播": test_forward_pass(config),
    }
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！可以开始训练了。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
