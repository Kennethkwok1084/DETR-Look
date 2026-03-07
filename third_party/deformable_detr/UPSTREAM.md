# Deformable DETR 上游源码信息

## 来源
- **仓库**: https://github.com/fundamentalvision/Deformable-DETR
- **Commit**: 11169a60c33333af00a4849f1808023eba96a931
- **日期**: 2020-12-07 16:23:05 +0800
- **引入日期**: 2026-01-08

## 目录结构
- `models/`: 核心模型代码（backbone, deformable_detr, deformable_transformer 等）
- `models/ops/`: CUDA 扩展（MultiScaleDeformableAttention）
- `util/`: 工具函数（box_ops, misc 等）

## 依赖
- PyTorch >= 1.5.0
- torchvision >= 0.6.0
- CUDA（用于编译 ops/）

## 编译说明
CUDA 扩展需要在使用前编译：
```bash
cd third_party/deformable_detr/models/ops
python setup.py build install
```

## 注意事项
1. 官方代码基于较早的 PyTorch 版本，可能需要适配
2. CUDA 扩展对 CUDA 版本有要求，建议 CUDA 10.2+
3. CPU 模式下无法使用 Deformable Attention
