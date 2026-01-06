# 训练基本功使用指南

本文档说明如何使用新增的训练功能：Checkpoint/Resume、AMP混合精度、子集采样、过拟合模式、Progressive Resizing 和预算化搜索。

---

## 📋 功能概览

### 1. Checkpoint / Resume（完整状态保存）

**功能**：保存并恢复训练的完整状态，包括：
- 模型参数
- Optimizer 状态
- LR Scheduler 状态
- AMP Scaler 状态（如启用AMP）
- 当前 epoch/iter
- 最佳指标值
- 随机数生成器状态（保证完全可复现）

**配置**：
```yaml
training:
  resume: null  # 设为checkpoint路径可恢复训练
  # 例如: resume: "outputs/baseline_run/checkpoint_epoch_10.pth"
```

**命令行**：
```bash
# 从头训练
python tools/train_detr.py --config configs/detr_baseline.yaml

# 恢复训练（方式1：配置文件）
# 修改配置: training.resume: "outputs/baseline_run/checkpoint_epoch_10.pth"
python tools/train_detr.py --config configs/detr_baseline.yaml

# 恢复训练（方式2：命令行）
python tools/train_detr.py --config configs/detr_baseline.yaml \
  --resume outputs/baseline_run/checkpoint_epoch_10.pth
```

**自动保存**：
- 每 N 个 epoch 保存：`checkpoint_epoch_{N}.pth`
- 最佳模型：`best.pth`（基于验证mAP或训练loss）
- 最终模型：`last.pth`

---

### 2. AMP 混合精度训练

**功能**：使用 PyTorch 的自动混合精度（Automatic Mixed Precision）加速训练并降低显存占用。

**优势**：
- 训练速度提升 1.5-2x
- 显存占用减少 30-50%
- 对精度影响很小

**配置**：
```yaml
training:
  amp: true  # 启用AMP
```

**命令行**：
```bash
# 启用AMP训练
python tools/train_detr.py --config configs/detr_baseline.yaml

# 冒烟测试（验证AMP稳定性）
# 修改 configs/detr_smoke.yaml: amp: true
python tools/train_detr.py --config configs/detr_smoke.yaml
```

**注意**：
- 仅在 GPU 训练时生效
- 若训练不稳定（loss=NaN），可临时关闭

---

### 3. 子集采样（快速验证）

**功能**：从完整数据集中随机采样子集进行训练，用于快速验证或预算搜索。

**使用场景**：
- 快速验证训练流程（100-500张图）
- 预算化超参数搜索（1000-5000张图）
- 调试数据加载器

**配置**：
```yaml
training:
  subset_size: 1000  # 使用1000张图
  subset_seed: 42    # 固定随机种子（可复现）
```

**命令行**：
```bash
# 使用100张图快速验证
python tools/train_detr.py --config configs/detr_baseline.yaml \
  --subset-size 100

# 结合冒烟测试配置
python tools/train_detr.py --config configs/detr_smoke.yaml
```

---

### 4. 小样本过拟合模式

**功能**：选择前N个样本，关闭数据增强和shuffle，验证模型能否过拟合小样本。

**目的**：
- 验证训练流程正确性
- 检查模型capacity
- 排查loss不下降问题

**配置**：
```yaml
training:
  subset_size: 10  # 使用10张图
  overfit: true    # 开启过拟合模式
```

**命令行**：
```bash
# 10张图过拟合测试
python tools/train_detr.py --config configs/detr_baseline.yaml \
  --subset-size 10 --overfit

# 或使用冒烟配置
python tools/train_detr.py --config configs/detr_smoke.yaml \
  --subset-size 10 --overfit
```

**预期结果**：
- Loss 明显下降（趋势为主，检测任务不必降到0）
- 训练集精度接近100%

**若过拟合失败，检查**：
- 类别映射是否正确
- bbox 坐标系与归一化
- 学习率是否过大/过小
- loss 计算逻辑

---

### 5. Progressive Resizing（渐进式分辨率）

**功能**：训练过程中逐步提高输入分辨率，先低分辨率快速收敛，再高分辨率冲击精度。

**优势**：
- 加快早期收敛
- 提升小目标AP
- 节省训练时间

**配置**：
```yaml
training:
  resize_schedule:
    - [1, 640]    # Epoch 1-19: 640x640
    - [20, 800]   # Epoch 20-39: 800x800
    - [40, 960]   # Epoch 40+: 960x960
```

**示例**：
```bash
# 编辑配置文件添加 resize_schedule
python tools/train_detr.py --config configs/detr_baseline.yaml
```

**注意**：
- 分辨率切换时学习率可能需要调整
- 建议配合checkpoint定期保存

---

### 6. 预算化超参数搜索

**功能**：批量运行小预算trial（少epoch/小子集/低分辨率），快速筛选超参数，淘汰差配置。

**使用场景**：
- 学习率调优
- Batch size 搜索
- 模型结构消融
- 数据增强策略对比

**步骤**：

#### 6.1 创建试验参数文件

编辑 `experiments/my_trials.json`：
```json
[
  {
    "training.optimizer.lr": 1e-4,
    "training.batch_size": 4
  },
  {
    "training.optimizer.lr": 5e-5,
    "training.batch_size": 4
  },
  {
    "training.optimizer.lr": 2e-4,
    "training.batch_size": 8
  }
]
```

#### 6.2 运行批量试验

```bash
python tools/run_trials.py \
  --base-config configs/detr_baseline.yaml \
  --trials-file experiments/my_trials.json \
  --output-dir outputs/trials \
  --budget-epochs 5 \
  --budget-subset 1000 \
  --budget-size 640 \
  --early-stop-threshold 0.1
```

**参数说明**：
- `--budget-epochs`：每个trial运行的epoch数
- `--budget-subset`：每个trial使用的样本数
- `--budget-size`：预算分辨率
- `--early-stop-threshold`：mAP低于此值标记为淘汰

#### 6.3 查看结果

生成 `outputs/trials/trials_{timestamp}.csv`：
```csv
trial_id,status,final_map,final_loss,param_training.optimizer.lr,param_training.batch_size
1,completed,0.1234,2.345,0.0001,4
2,completed,0.0987,2.678,0.00005,4
3,completed,0.1456,2.123,0.0002,8
```

#### 6.4 选择最佳配置

根据 `final_map` 排序，选择前几名配置进入完整训练。

---

## 🔄 完整训练流程示例

### 阶段1：冒烟测试（10分钟）

验证训练流程，确保没有bug。

```bash
# 1. 基础冒烟（100张图 × 2 epoch）
python tools/train_detr.py --config configs/detr_smoke.yaml

# 2. 小样本过拟合（10张图）
python tools/train_detr.py --config configs/detr_smoke.yaml \
  --subset-size 10 --overfit

# 3. 验证Resume
python tools/train_detr.py --config configs/detr_smoke.yaml \
  --resume outputs/smoke_test/checkpoint_epoch_1.pth

# 4. 验证AMP（修改配置: amp: true）
python tools/train_detr.py --config configs/detr_smoke.yaml
```

### 阶段2：预算化搜索（1-2小时）

快速筛选超参数。

```bash
# 运行多个trial（每个5 epoch × 1000张图）
python tools/run_trials.py \
  --base-config configs/detr_baseline.yaml \
  --trials-file experiments/trials_lr_bs.json \
  --output-dir outputs/trials_lr_bs \
  --budget-epochs 5 \
  --budget-subset 1000
```

### 阶段3：完整训练（数小时-数天）

使用最佳配置进行完整训练。

```bash
# 启用AMP，使用全量数据
# 编辑 configs/detr_baseline.yaml: amp: true
python tools/train_detr.py --config configs/detr_baseline.yaml

# 中断后恢复
python tools/train_detr.py --config configs/detr_baseline.yaml \
  --resume outputs/baseline_run/last.pth
```

### 阶段4：Progressive Resizing（可选）

先低分辨率训练，再高分辨率微调。

```bash
# 编辑配置添加:
# resize_schedule: [[1, 640], [20, 800], [40, 960]]
python tools/train_detr.py --config configs/detr_baseline.yaml
```

---

## 📊 配置文件对照表

| 功能 | 配置文件 | 关键参数 |
|------|---------|---------|
| 基础训练 | `detr_baseline.yaml` | 全量数据，50 epoch |
| 冒烟测试 | `detr_smoke.yaml` | 100张图，2 epoch |
| 预算搜索 | `run_trials.py` | 自定义 trial 参数 |

**配置继承**：
```bash
# 从baseline继承，覆盖部分参数
python tools/train_detr.py --config configs/detr_baseline.yaml \
  --subset-size 1000 \
  --max-iter 500 \
  --eval-interval 2
```

---

## ⚠️ 常见问题

### Q1: Resume后loss突然变大？
**A**: 检查学习率是否正确恢复。确保 `scheduler` 状态也被加载。

### Q2: AMP训练出现 NaN？
**A**: 尝试关闭AMP或降低学习率。某些模型对混合精度敏感。

### Q3: 过拟合测试loss不下降？
**A**: 检查：
1. 数据是否正确加载（bbox坐标、类别ID）
2. 学习率是否合适
3. 模型是否正确初始化

### Q4: 预算搜索很慢？
**A**: 减少 `--budget-epochs` 和 `--budget-subset`，或并行运行（手动分批）。

### Q5: Progressive Resizing何时切换？
**A**: 建议在loss plateau或学习率衰减后切换分辨率。

---

## 📝 配置模板

### 完整训练（Baseline）

```yaml
training:
  batch_size: 4
  max_epochs: 50
  amp: true
  resume: null
  subset_size: null  # 使用全量数据
  overfit: false
  resize_schedule: null
```

### 快速验证（Smoke Test）

```yaml
training:
  batch_size: 2
  max_epochs: 2
  amp: false
  subset_size: 100
  overfit: false
```

### 小样本过拟合

```yaml
training:
  batch_size: 2
  max_epochs: 10
  amp: false
  subset_size: 10
  overfit: true
```

### 预算搜索（Trial）

通过 `run_trials.py` 自动注入：
- `max_epochs: 5`
- `subset_size: 1000`
- `eval_interval: 1`

---

## ✅ 验收清单

按照 `develop.md` 中的"执行清单"验收：

**阶段 B：训练冒烟**
- [x] 冒烟测试通过（100-500张图，200 iter或1-2 epoch）
- [x] dataloader 正常迭代
- [x] loss 正常输出
- [x] eval 能跑通
- [x] checkpoint 保存正常（best/last）
- [x] 小样本过拟合（1-10张图，loss明显下降）
- [x] Resume 恢复训练成功
- [x] AMP 可用（或确认不稳定时关闭）

**阶段 C：Baseline 训练**
- [ ] 完整训练完成
- [ ] 指标记录（mAP/AP_small/Loss/时间/显存）
- [ ] 权重保存（best.pth/last.pth）

**阶段 D：预算搜索**
- [ ] 批量trial运行成功
- [ ] 结果CSV生成
- [ ] 最佳配置筛选

---

**🎯 现在所有训练基本功已就绪，可以开始实验了！**
