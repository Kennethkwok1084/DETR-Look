#!/usr/bin/env python3
"""
测试 Deformable DETR CUDA 扩展是否正确编译和安装
"""

import sys
import torch

print("="*60)
print("测试 Deformable DETR CUDA 扩展")
print("="*60)

# 测试 1: 检查 PyTorch CUDA 可用性
print("\n1. PyTorch CUDA 状态:")
print(f"   PyTorch 版本: {torch.__version__}")
print(f"   CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   当前设备: {torch.cuda.get_device_name(0)}")
else:
    print("   ⚠️  CUDA 不可用，Deformable DETR 无法运行")
    sys.exit(1)

# 测试 2: 导入编译的扩展
print("\n2. 导入 CUDA 扩展:")
try:
    import MultiScaleDeformableAttention as MSDA
    print("   ✅ MultiScaleDeformableAttention 模块导入成功")
    
    # 检查函数
    if hasattr(MSDA, 'ms_deform_attn_forward'):
        print("   ✅ ms_deform_attn_forward 函数存在")
    if hasattr(MSDA, 'ms_deform_attn_backward'):
        print("   ✅ ms_deform_attn_backward 函数存在")
        
except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    print("\n   请确保已编译 CUDA 扩展:")
    print("   cd third_party/deformable_detr/models/ops")
    print("   python setup.py build install")
    sys.exit(1)

# 测试 3: 导入 Python 包装器
print("\n3. 导入 Python 包装器:")
try:
    # 添加路径
    from pathlib import Path
    third_party_path = Path(__file__).parent / "third_party" / "deformable_detr"
    sys.path.insert(0, str(third_party_path))
    
    from models.ops.modules import MSDeformAttn
    print("   ✅ MSDeformAttn 模块导入成功")
    
except ImportError as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)

# 测试 4: 创建模块实例
print("\n4. 创建模块实例:")
try:
    d_model = 256
    n_levels = 4
    n_heads = 8
    n_points = 4
    
    msda = MSDeformAttn(
        d_model=d_model,
        n_levels=n_levels,
        n_heads=n_heads,
        n_points=n_points
    )
    msda = msda.cuda()
    
    print(f"   ✅ MSDeformAttn 实例创建成功")
    print(f"      - d_model: {d_model}")
    print(f"      - n_levels: {n_levels}")
    print(f"      - n_heads: {n_heads}")
    print(f"      - n_points: {n_points}")
    
except Exception as e:
    print(f"   ❌ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 5: 前向传播测试
print("\n5. 前向传播测试:")
try:
    batch_size = 2
    num_queries = 300
    
    # 创建虚拟输入
    # input_spatial_shapes 定义每个特征层的空间尺寸
    input_spatial_shapes = torch.tensor([[50, 50], [25, 25], [13, 13], [7, 7]], dtype=torch.long).cuda()
    
    # 计算每层的起始索引和总长度
    level_sizes = (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).tolist()
    input_level_start_index = torch.tensor([0] + level_sizes[:-1], dtype=torch.long).cumsum(0).cuda()
    total_len = sum(level_sizes)
    
    # 创建输入张量
    query = torch.randn(batch_size, num_queries, d_model).cuda()
    reference_points = torch.rand(batch_size, num_queries, n_levels, 2).cuda()
    input_flatten = torch.randn(batch_size, total_len, d_model).cuda()  # 注意：长度是 total_len
    
    # 前向传播
    with torch.no_grad():
        output = msda(
            query,
            reference_points,
            input_flatten,
            input_spatial_shapes,
            input_level_start_index,
            None
        )
    
    print(f"   ✅ 前向传播成功")
    print(f"      query 形状: {query.shape}")
    print(f"      input_flatten 形状: {input_flatten.shape}")
    print(f"      输出形状: {output.shape}")
    print(f"      空间尺寸: {input_spatial_shapes.tolist()}")
    print(f"      总位置数: {total_len}")
    
except Exception as e:
    print(f"   ❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 所有测试通过！CUDA 扩展工作正常")
print("="*60)
