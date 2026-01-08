#!/usr/bin/env python3
"""测试两种模型构建"""

import yaml
from models import build_model

print("=" * 60)
print("测试 DETR 和 Deformable DETR 模型构建")
print("=" * 60)

# 测试 DETR
print("\n1️⃣  测试 DETR (HuggingFace)")
print("-" * 60)
detr_config = yaml.safe_load(open('configs/detr_baseline.yaml', encoding='utf-8'))
detr_model = build_model(detr_config)
detr_params = sum(p.numel() for p in detr_model.parameters())
print(f"✅ DETR 模型: {detr_params/1e6:.1f}M 参数\n")

# 测试 Deformable DETR
print("2️⃣  测试 Deformable DETR (官方实现)")
print("-" * 60)
deformable_config = yaml.safe_load(open('configs/deformable_detr_baseline.yaml', encoding='utf-8'))
deformable_model = build_model(deformable_config)
deformable_params = sum(p.numel() for p in deformable_model.parameters())
print(f"✅ Deformable DETR 模型: {deformable_params/1e6:.1f}M 参数\n")

print("=" * 60)
print("✅ 所有测试通过！迁移完成！")
print("=" * 60)
