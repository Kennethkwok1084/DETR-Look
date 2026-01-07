# Torchvision DETR 脚本状态说明

## ⚠️ 当前不可用

以下脚本已标记为 `.BROKEN`，因为当前环境不支持：

- `train_detr_torchvision.py.BROKEN`
- `smoke_test_torchvision.py.BROKEN`

## 原因

1. **环境限制**: 当前 torchvision 0.24.1+cpu 不包含 DETR 模型
2. **接口冲突**: 这些脚本混用了 transformers DETR 的构建和 torchvision 风格的调用接口

## 替代方案

**请使用 `train_detr_optimized.py`**，该脚本：
- ✅ 使用 transformers DETR（可用）
- ✅ 完整的训练/评估/checkpoint 功能
- ✅ C++ 图像解码优化
- ✅ 优化的 DataLoader 参数
- ✅ 正确的 DETR 标签格式（归一化 cxcywh）
- ✅ 官方 post_process_object_detection

## 未来计划

等 CUDA 环境就绪且安装包含 DETR 的 torchvision 版本后：
1. 重新实现纯 torchvision DETR 训练脚本
2. 保持 `train_detr_optimized.py` 作为 transformers 版本的参考实现

## 验证 torchvision DETR 可用性

```bash
python -c "import torchvision.models.detection as d; print(hasattr(d,'detr_resnet50'))"
```

如果输出 `True`，则可以开发纯 torchvision 版本的脚本。

---

**当前推荐**: 使用 `train_detr_optimized.py`  
**日期**: 2026-01-06
