#!/usr/bin/env python3
"""测试两种模型构建"""

import yaml
from models import build_model

print("=" * 60)
print("测试 DETR 和 Deformable DETR 模型构建")
print("=" * 60)

results = []

# 测试 DETR
print("\n1️⃣  测试 DETR (HuggingFace)")
print("-" * 60)
try:
    detr_config = yaml.safe_load(open('configs/detr_baseline.yaml', encoding='utf-8'))
    detr_model = build_model(detr_config)
    detr_params = sum(p.numel() for p in detr_model.parameters())
    print(f"✅ DETR 模型: {detr_params/1e6:.1f}M 参数\n")
    results.append(True)
except Exception as e:
    print(f"❌ DETR 构建失败: {e}\n")
    results.append(False)

# 测试 Deformable DETR
print("2️⃣  测试 Deformable DETR (官方实现)")
print("-" * 60)
try:
    deformable_config = yaml.safe_load(open('configs/deformable_detr_baseline.yaml', encoding='utf-8'))
    deformable_model = build_model(deformable_config)
    deformable_params = sum(p.numel() for p in deformable_model.parameters())
    print(f"✅ Deformable DETR 模型: {deformable_params/1e6:.1f}M 参数\n")
    results.append(True)
except ImportError as e:
    if "CUDA" in str(e) or "MultiScaleDeformableAttention" in str(e):
        print(f"⚠️  需要编译 CUDA 扩展")
        print(f"   错误: {e}")
        print(f"   请运行: cd third_party/deformable_detr/models/ops && python setup.py build install\n")
    else:
        print(f"❌ Deformable DETR 导入失败: {e}\n")
    results.append(False)
except Exception as e:
    print(f"❌ Deformable DETR 构建失败: {e}\n")
    import traceback
    traceback.print_exc()
    results.append(False)

print("=" * 60)
if all(results):
    print("✅ 所有测试通过！迁移完成！")
else:
    print("⚠️  部分测试失败，请检查上述错误信息")
print("=" * 60)

