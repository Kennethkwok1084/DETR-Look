# Deformable DETR 稳定收敛训练指南

## 功能支持

✅ **已实现功能**：
- 梯度累积（Gradient Accumulation）
- 梯度裁剪（Gradient Clipping）
- AMP 混合精度（RTX 5090 自动 BF16）
- 预训练权重加载
- 完整评估和 checkpoint 保存

## 稳定训练配置（RTX 5090 32GB）

### 核心参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--batch-size` | 20 | 单批次大小，不贴满显存留余量 |
| `--grad-accum` | 2 | 梯度累积，有效 batch = 20×2 = 40 |
| `--clip-max-norm` | 0.1 | 梯度裁剪，防止梯度爆炸 |
| `--amp` | 必须 | 混合精度，5090 自动 BF16 |
| `--pretrained` | 必须 | 预训练权重，收敛更快更稳 |
| `--lr` | 4e-5 或 5e-5 | 学习率偏小，稳定优先 |

### 完整命令（稳定版）

#### 短训验证（2000 样本，10 epoch）

```bash
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --val-img data/traffic_coco/bdd100k_det/images/val \
  --val-ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --subset 2000 \
  --num-epochs 10 \
  --eval-interval 2 \
  --batch-size 20 \
  --grad-accum 2 \
  --clip-max-norm 0.1 \
  --lr 5e-5 \
  --num-workers 16 \
  --prefetch-factor 2 \
  --amp --pretrained \
  --output-dir outputs/detr_stable_short
```

**预期结果**：
- Loss 平滑下降（无大幅波动）
- mAP 每 2 epoch 稳步提升
- 训练时间约 1-2 小时

#### 全量训练（70K 样本，50 epoch）

```bash
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --val-img data/traffic_coco/bdd100k_det/images/val \
  --val-ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --num-epochs 50 \
  --eval-interval 5 \
  --batch-size 20 \
  --grad-accum 2 \
  --clip-max-norm 0.1 \
  --lr 4e-5 \
  --num-workers 16 \
  --prefetch-factor 2 \
  --amp --pretrained \
  --output-dir outputs/detr_full_stable
```

**预期结果**：
- 训练时间约 30-50 小时
- 最终 mAP@0.5:0.95 > 0.30
- AP_small（小目标）逐步上升

## 梯度累积工作原理

### 有效 Batch Size

```
effective_batch = batch_size × grad_accum
              = 20 × 2
              = 40
```

### 内存占用对比

| 配置 | 显存占用 | 有效Batch | 收敛性 |
|------|----------|-----------|--------|
| batch=40, accum=1 | ~30GB（爆显存） | 40 | ❌ OOM |
| batch=20, accum=2 | ~18GB | 40 | ✅ 稳定 |
| batch=10, accum=4 | ~12GB | 40 | ✅ 更稳 |

### 实现细节

```python
# 损失缩放（防止累积后梯度放大）
loss = loss / grad_accum
loss.backward()

# 每 grad_accum 步更新一次参数
if step % grad_accum == 0:
    # 梯度裁剪（在 optimizer.step() 前）
    if clip_max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
    
    optimizer.step()
    optimizer.zero_grad()
```

## 梯度裁剪作用

### 防止梯度爆炸

```python
# 裁剪前：梯度范数可能很大
grad_norm_before = 15.6

# 裁剪后：限制最大范数
grad_norm_after = min(15.6, 0.1) = 0.1
```

### 推荐值

- `0.1`：严格裁剪，最稳定（推荐）
- `0.5`：适度裁剪，平衡速度和稳定性
- `1.0`：宽松裁剪，允许更大梯度

## 训练输出示例

```
================================================================================
🚀 Deformable DETR 训练（transformers + 优化数据加载）
================================================================================
输出目录: outputs/detr_full_stable
设备: cuda
Batch Size: 20 | Workers: 16
图像尺寸: min=800, max=1333
梯度累积: 2 步 | 有效Batch: 40
梯度裁剪: clip_max_norm=0.1
================================================================================

Epoch [1] Step [50/4375] Loss: 2.1245 (avg: 2.3456) | Speed: 3.45 it/s
  ⏱️  t_load: 0.125s (43.2%) | t_step: 0.164s (56.8%)

Epoch [1] Step [100/4375] Loss: 1.9832 (avg: 2.2103) | Speed: 3.48 it/s
  ⏱️  t_load: 0.123s (42.8%) | t_step: 0.165s (57.2%)
```

## 收敛判断标准

### ✅ 正常收敛

- Loss 平滑下降（无大幅跳跃）
- mAP 每 2-5 epoch 有明显提升
- AP_small 逐步上升（小目标难度高）
- 训练速度稳定（3-4 it/s）

### ❌ 异常情况

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| Loss 爆炸（NaN） | 学习率太大 / 无裁剪 | 降低 lr，增加 clip_max_norm |
| Loss 震荡 | 有效 batch 太小 | 增加 grad_accum |
| mAP 不涨 | lr 太小 / epoch 不够 | 增加 lr 或 epoch |
| 显存 OOM | batch_size 太大 | 减小 batch_size，增加 grad_accum |

## 调优建议

### 1. 优先保证稳定

```bash
# 最稳定配置（牺牲速度）
--batch-size 10 --grad-accum 4 --clip-max-norm 0.1 --lr 3e-5
```

### 2. 平衡速度和稳定

```bash
# 推荐配置
--batch-size 20 --grad-accum 2 --clip-max-norm 0.1 --lr 4e-5
```

### 3. 极限速度（风险较高）

```bash
# 快速但可能不稳定
--batch-size 32 --grad-accum 1 --clip-max-norm 0.5 --lr 1e-4
```

## 实验节奏

1. **短训验证**（必须）
   - 2000 样本，10 epoch
   - 观察 loss 曲线是否平滑
   - 验证 mAP 是否上升
   - 耗时 1-2 小时

2. **中等规模**（可选）
   - 10000 样本，30 epoch
   - 更准确的趋势评估
   - 耗时 8-12 小时

3. **全量训练**（最终）
   - 70000 样本，50 epoch
   - 固定配置，不轻易改动
   - 耗时 30-50 小时

## 黑名单配置（可选）

预扫描损坏图像，提升稳定性：

```bash
# 扫描生成黑名单
python tools/scan_corrupted_images.py \
  --ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --img-dir data/traffic_coco/bdd100k_det/images/train \
  --workers 16

# 训练时使用
python tools/train_detr_optimized.py \
  ... \
  --blacklist outputs/blacklist_instances_train.json
```

## 常见问题

**Q: grad_accum=2 会让训练慢一倍吗？**
A: 不会。虽然每 2 步才更新一次参数，但单步时间不变，总迭代次数不变。整体速度几乎无影响。

**Q: clip_max_norm 会影响收敛速度吗？**
A: 可能略微变慢，但换来的稳定性值得。没有裁剪可能导致训练崩溃。

**Q: BF16 和 FP16 哪个更稳？**
A: BF16 更稳（数值范围更大）。RTX 5090 在 `--amp` 下自动使用 BF16。

**Q: 如何判断是否需要增加 grad_accum？**
A: 如果 loss 震荡剧烈、mAP 不稳定上升，建议增加。有效 batch 建议 ≥32。
