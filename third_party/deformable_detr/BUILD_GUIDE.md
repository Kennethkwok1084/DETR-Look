# Deformable DETR CUDA 扩展编译指南

## 概述

Deformable DETR 需要编译 CUDA 扩展以支持 Multi-Scale Deformable Attention 操作。

## 环境要求

- CUDA 10.2+ 
- PyTorch (与 CUDA 版本匹配)
- GCC/G++ 编译器 (Linux) 或 MSVC (Windows)

## Linux 编译步骤

```bash
cd third_party/deformable_detr/models/ops
python setup.py build install
```

## Windows 编译步骤

### 前置条件
1. 安装 Visual Studio 2019/2022 (含 C++ 桌面开发工具)
2. 安装 CUDA Toolkit (https://developer.nvidia.com/cuda-downloads)
3. 确保 PyTorch 的 CUDA 版本与 CUDA Toolkit 版本匹配

### 编译命令
```powershell
cd third_party\deformable_detr\models\ops
python setup.py build_ext --inplace
```

## CPU Fallback (不推荐用于训练)

如果无法编译 CUDA 扩展，可以使用 CPU 版本的 Multi-Scale Deformable Attention（速度会非常慢）。

修改 `models/ops/modules/ms_deform_attn.py`，启用 CPU fallback。

## 验证安装

```python
import torch
from models.ops.modules import MSDeformAttn

# 测试 CUDA 扩展
attn = MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4)
if torch.cuda.is_available():
    attn = attn.cuda()
    print("✅ CUDA 扩展工作正常")
else:
    print("⚠️ 使用 CPU 模式")
```

## 常见问题

### 1. NVCC not found
确保 CUDA bin 目录在 PATH 中：
```bash
# Linux
export PATH=/usr/local/cuda/bin:$PATH

# Windows
# 添加 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin 到系统 PATH
```

### 2. PyTorch CUDA 版本不匹配
```bash
# 查看 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 查看系统 CUDA 版本
nvcc --version
```

### 3. 编译错误
- 确保 GCC 版本兼容 (GCC 7-9 for CUDA 11)
- Windows 需要 MSVC 2019+
- 检查 CUDA_HOME 环境变量

## Docker 方案（推荐）

如果本地编译困难，可以使用 Docker：

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

WORKDIR /workspace
COPY . .

RUN cd third_party/deformable_detr/models/ops && \
    python setup.py build install

CMD ["/bin/bash"]
```
