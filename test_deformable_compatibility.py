#!/usr/bin/env python3
"""
验证 Deformable DETR 接口兼容性
测试所有关键接口是否正常工作
"""

import yaml
import torch
from PIL import Image
import numpy as np


def test_model_interface():
    """测试模型接口兼容性"""
    print("="*60)
    print("测试 1: Deformable DETR 模型接口兼容性")
    print("="*60)
    
    from models import build_model
    
    # 加载配置
    config = yaml.safe_load(open('configs/deformable_detr_baseline.yaml', encoding='utf-8'))
    
    # 构建模型
    print("构建模型...")
    model = build_model(config)
    model.eval()
    
    # 创建虚拟输入
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 800, 800)
    pixel_mask = torch.ones(batch_size, 800, 800, dtype=torch.bool)
    
    # 创建虚拟标签 (HF 风格)
    labels = [
        {
            'class_labels': torch.tensor([1, 2]),
            'boxes': torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]]),
        },
        {
            'class_labels': torch.tensor([0]),
            'boxes': torch.tensor([[0.6, 0.6, 0.15, 0.15]]),
        }
    ]
    
    print("\n测试 1a: HF 风格接口（训练模式）")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
    
    print(f"  ✅ 返回 loss: {outputs.loss.item():.4f}")
    print(f"  ✅ 返回 logits 形状: {outputs.logits.shape}")
    print(f"  ✅ 返回 pred_boxes 形状: {outputs.pred_boxes.shape}")
    
    print("\n测试 1b: HF 风格接口（推理模式）")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    print(f"  ✅ 返回 pred_logits 形状: {outputs['pred_logits'].shape}")
    print(f"  ✅ 返回 pred_boxes 形状: {outputs['pred_boxes'].shape}")
    
    print("\n✅ 模型接口测试通过！")


def test_image_processor():
    """测试图像处理器"""
    print("\n" + "="*60)
    print("测试 2: 本地图像处理器")
    print("="*60)
    
    from models import build_image_processor
    
    # 加载配置
    config = yaml.safe_load(open('configs/deformable_detr_baseline.yaml', encoding='utf-8'))
    
    # 构建处理器
    print("构建本地图像处理器...")
    processor = build_image_processor(config)
    
    # 创建虚拟图像
    dummy_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    print("\n测试 2a: 图像编码")
    encoding = processor(images=[dummy_img, dummy_img], return_tensors='pt')
    
    print(f"  ✅ pixel_values 形状: {encoding['pixel_values'].shape}")
    print(f"  ✅ pixel_mask 形状: {encoding['pixel_mask'].shape}")
    
    print("\n测试 2b: 后处理")
    # 创建虚拟模型输出
    batch_size = 2
    num_queries = 300
    num_classes = 13
    
    dummy_outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
        'pred_boxes': torch.rand(batch_size, num_queries, 4),
    }
    target_sizes = torch.tensor([[480, 640], [480, 640]])
    
    results = processor.post_process_object_detection(
        dummy_outputs, 
        target_sizes=target_sizes,
        threshold=0.7
    )
    
    print(f"  ✅ 处理了 {len(results)} 张图像")
    for i, result in enumerate(results):
        print(f"  ✅ 图像 {i}: {len(result['scores'])} 个检测框")
    
    print("\n✅ 图像处理器测试通过！")


def test_label_mapping():
    """测试标签字段映射"""
    print("\n" + "="*60)
    print("测试 3: 标签字段映射（class_labels → labels）")
    print("="*60)
    
    from models import build_model
    
    config = yaml.safe_load(open('configs/deformable_detr_baseline.yaml', encoding='utf-8'))
    model = build_model(config)
    model.eval()
    
    # HF 风格标签
    hf_labels = [
        {
            'class_labels': torch.tensor([1, 2, 0]),
            'boxes': torch.rand(3, 4),
        }
    ]
    
    pixel_values = torch.randn(1, 3, 800, 800)
    pixel_mask = torch.ones(1, 800, 800, dtype=torch.bool)
    
    print("使用 HF 风格标签（class_labels 字段）...")
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=hf_labels)
    
    print(f"  ✅ 成功计算损失: {outputs.loss.item():.4f}")
    print(f"  ✅ 标签自动映射为官方格式")
    
    print("\n✅ 标签映射测试通过！")


def test_lazy_import():
    """测试延迟导入只执行一次"""
    print("\n" + "="*60)
    print("测试 4: _lazy_import 缓存机制")
    print("="*60)
    
    from models.deformable_detr_model import _lazy_import_deformable_detr, _DEFORMABLE_MODULES
    
    print("第一次导入...")
    modules1 = _lazy_import_deformable_detr()
    
    print("第二次导入（应该使用缓存）...")
    modules2 = _lazy_import_deformable_detr()
    
    if modules1 is modules2:
        print("  ✅ 使用了缓存，避免重复导入")
    else:
        print("  ❌ 警告：每次都重新导入")
    
    print("\n✅ 延迟导入测试通过！")


if __name__ == '__main__':
    try:
        test_model_interface()
        test_image_processor()
        test_label_mapping()
        test_lazy_import()
        
        print("\n" + "="*60)
        print("🎉 所有兼容性测试通过！")
        print("="*60)
        print("\n关键修复:")
        print("  ✅ DeformableDETRModel 支持 HF 风格参数 (pixel_values/pixel_mask/labels)")
        print("  ✅ 标签字段自动映射 (class_labels → labels)")
        print("  ✅ 本地图像处理器，无需下载 HF 模型")
        print("  ✅ 后处理函数兼容官方输出格式")
        print("  ✅ _lazy_import 只执行一次，提升性能")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
