#!/usr/bin/env python3
"""测试模型构建"""
import yaml
from models import build_model

# 测试 DETR
print("=" * 60)
print("测试 DETR 模型构建（向后兼容性）")
print("=" * 60)
with open('configs/detr_baseline.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

model = build_model(config)
param_count = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅ DETR 模型构建成功")
print(f"   参数量: {param_count:.1f}M")
print()

# 测试 Deformable DETR（需要 CUDA 扩展）
print("=" * 60)
print("测试 Deformable DETR 模型构建")
print("=" * 60)
try:
    with open('configs/deformable_detr_baseline.yaml', 'r', encoding='utf-8') as f:
        config2 = yaml.safe_load(f)
    
    model2 = build_model(config2)
    param_count2 = sum(p.numel() for p in model2.parameters()) / 1e6
    print(f"✅ Deformable DETR 模型构建成功")
    print(f"   参数量: {param_count2:.1f}M")
except Exception as e:
    print(f"⚠️  Deformable DETR 构建失败（预期需要编译 CUDA 扩展）")
    print(f"   错误: {type(e).__name__}: {e}")
