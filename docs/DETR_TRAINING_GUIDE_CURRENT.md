# Deformable DETR 训练实用指南（当前可用版本）

## ⚠️ 重要声明

**当前环境**: torchvision 0.24.1+cpu **不包含** Deformable DETR 模型  
**实际使用**: transformers 库的 `DeformableDetrForObjectDetection`  
**可用脚本**: `tools/train_detr_optimized.py`（已完整修复）

## 🚀 快速开始（3分钟）

### 1. 激活环境
```bash
cd /srv/code/detr_traffic_analysis
source .venv/bin/activate
```

### 2. 验证数据集
```bash
ls data/traffic_coco/bdd100k_det/annotations/instances_train.json
ls data/traffic_coco/bdd100k_det/images/train/ | head
```

### 3. 运行训练

**方式 A：一键启动**（推荐）
```bash
./tools/run_torchvision_training.sh
# 选择模式 1/2/3
```

**方式 B：直接运行**
```bash
# 冒烟测试（100张图，1 epoch）
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --subset 100 --num-epochs 1 --batch-size 4 \
  --output-dir outputs/smoke_test

# 快速训练（2000张图，10 epochs）
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --val-img data/traffic_coco/bdd100k_det/images/val \
  --val-ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --subset 2000 --num-epochs 10 --eval-interval 2 \
  --batch-size 16 --num-workers 12 --amp --pretrained \
  --output-dir outputs/detr_fast

# 完整训练（全部数据，50 epochs）
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --val-img data/traffic_coco/bdd100k_det/images/val \
  --val-ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --num-epochs 50 --eval-interval 5 \
  --batch-size 16 --num-workers 12 --amp --pretrained \
  --output-dir outputs/detr_full
```

## 📋 参数说明

### 必需参数
- `--train-img`: 训练图像目录
- `--train-ann`: 训练标注文件（COCO JSON）

### 重要参数
- `--val-img/--val-ann`: 验证集（启用评估）
- `--num-classes`: 类别数（默认 3）
- `--pretrained`: 使用预训练权重（推荐）
- `--amp`: 启用混合精度（显著加速）
- `--batch-size`: 批次大小（默认 16）
- `--num-workers`: 数据加载进程数（默认 12）
- `--num-epochs`: 训练轮数（默认 50）
- `--eval-interval`: 评估间隔（默认 5 epochs）

### 调试参数
- `--subset N`: 使用前 N 张图像（快速测试）
- `--device`: 设备（默认 cuda）
- `--output-dir`: 输出目录

## 🔧 技术细节

### 数据处理流程

1. **图像加载**（C++ 解码）
   ```python
   from torchvision.io import read_image
   img = read_image(path, mode=ImageReadMode.RGB).float() / 255.0
   ```

2. **Deformable DETR 标准归一化**
   ```python
   DETR_MEAN = [0.485, 0.456, 0.406]
   DETR_STD = [0.229, 0.224, 0.225]
   for c in range(3):
       img[c] = (img[c] - DETR_MEAN[c]) / DETR_STD[c]
   ```

3. **Bbox 转换**（xyxy 像素 → 归一化 cxcywh）
   ```python
   # 计算中心点和宽高
   cx = (x1 + x2) / 2 / img_w
   cy = (y1 + y2) / 2 / img_h
   w = (x2 - x1) / img_w
   h = (y2 - y1) / img_h
   ```

4. **Batch Padding**
   ```python
   # Pad 到 batch 最大尺寸
   max_h, max_w = ...
   padded_img = torch.zeros(3, max_h, max_w)
   padded_img[:, :h, :w] = img
   
   # Pixel mask（1=真实像素，0=padding）
   mask = torch.zeros(max_h, max_w)
   mask[:h, :w] = 1
   ```

### COCO Category ID 处理

**问题**：COCO category_id 不保证连续从 0 开始

**解决方案**：
```python
# 训练时：原始 ID → 连续 [0..N-1]
cat_ids = sorted(coco.getCatIds())
cat_id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}

# 评估时：反向映射回原始 ID
reverse_cat_id_map = {i: cat_id for i, cat_id in enumerate(cat_ids)}
```

### 评估坐标系

**关键**：使用原始图像尺寸（`orig_size`）而非 resize 后尺寸（`size`）

```python
# 错误示例（会导致 mAP 失真）
target_sizes = torch.stack([l["size"] for l in labels])  # resize 后

# 正确做法
target_sizes = torch.stack([l["orig_size"] for l in labels])  # 原始尺寸
```

**原因**：COCO GT 使用原始图像坐标，`post_process_object_detection` 会将预测框缩放到 `target_sizes` 对应的坐标系。

## 📊 输出结构

```
outputs/detr_optimized/
├── config.json          # 训练配置快照
├── metrics.json         # 每 epoch 指标
│   [
│     {
│       "epoch": 1,
│       "train_loss": 2.45,
│       "mAP": 0.234,
│       "AP_small": 0.156,
│       "epoch_time": 1234.5
│     },
│     ...
│   ]
├── last.pth             # 最新 checkpoint
└── best.pth             # 最佳模型（按 mAP）
```

### Checkpoint 内容
```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": 10,
    "iteration": 0,
    "best_map": 0.456
}
```

## 🔍 监控训练

### 查看日志
```bash
# 实时查看训练日志
tail -f outputs/detr_optimized/metrics.json

# 提取 mAP 趋势
cat outputs/detr_optimized/metrics.json | jq '.[].mAP'
```

### 检查输出
```bash
# 查看配置
cat outputs/detr_optimized/config.json

# 验证 checkpoint
python -c "
import torch
ckpt = torch.load('outputs/detr_optimized/best.pth', map_location='cpu')
print(f'Epoch: {ckpt[\"epoch\"]}')
print(f'Best mAP: {ckpt[\"best_map\"]:.4f}')
"
```

## 🐛 常见问题

### Q1: 显存不足
**现象**：CUDA out of memory

**解决**：
```bash
# 方案 1：降低 batch size
--batch-size 8  # 或 4

# 方案 2：降低 workers（减少 CPU 内存）
--num-workers 4

# 方案 3：关闭 AMP（不推荐）
# 去掉 --amp 参数
```

### Q2: 训练速度慢
**现象**：it/s 低于预期

**检查**：
```bash
# 1. GPU 利用率
nvidia-smi dmon -s u

# 2. DataLoader 性能
python tools/benchmark_dataloader.py \
  --batch-size 16 --num-workers 12

# 3. 尝试不同 workers 数量
--num-workers 8   # 或 16
```

### Q3: mAP 为 0 或异常低
**可能原因**：
1. ❌ 预训练权重未加载：确保使用 `--pretrained`
2. ❌ 学习率过大：检查 `--lr`（默认 5e-5）
3. ❌ 训练轮数不足：至少 10 epochs
4. ❌ 数据问题：验证标注格式

### Q4: 离线环境无法加载预训练权重
**解决**：
```bash
# 在线环境下提前下载
python -c "
from transformers import DeformableDetrForObjectDetection
DeformableDetrForObjectDetection.from_pretrained('SenseTime/deformable-detr')
"

# 缓存位置（默认）
~/.cache/huggingface/hub/models--facebook--deformable-detr/
```

### Q5: COCO 评估报错
**检查**：
1. 标注文件格式正确
2. `image_id` 匹配
3. `category_id` 存在于 COCO GT 中

## ✅ 验证清单

训练前检查：
- [ ] 虚拟环境已激活
- [ ] 数据集路径正确
- [ ] 标注文件可加载（`COCO(ann_file)`）
- [ ] 有足够的磁盘空间（至少 10GB）

训练中监控：
- [ ] Loss 正常下降
- [ ] GPU 利用率 > 70%
- [ ] 每 epoch 时间合理
- [ ] Checkpoint 正常保存

训练后验证：
- [ ] mAP 在合理范围（> 0.3 为基线）
- [ ] AP_small 有提升（小目标优化目标）
- [ ] 最佳模型已保存

## 🎯 性能基准（参考）

| 硬件 | Batch Size | Workers | AMP | it/s | 备注 |
|------|-----------|---------|-----|------|------|
| RTX 5090 32GB | 16 | 12 | ✅ | 3-4 | 目标性能 |
| RTX 3090 24GB | 12 | 8 | ✅ | 2-3 | 推荐配置 |
| RTX 3060 12GB | 8 | 4 | ✅ | 1-2 | 最小配置 |

## 📚 相关文档

- [DETR_TRAINING_README.md](DETR_TRAINING_README.md) - 技术细节
- [FIXES_2026_01_06.md](FIXES_2026_01_06.md) - 修复记录
- [develop.md](develop.md) § 3.11 - 性能优化记录

---

**版本**: v2.0（已修复所有严重问题）  
**日期**: 2026-01-06  
**状态**: ✅ 生产可用
