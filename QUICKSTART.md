# 快速开始指南

## ✅ 已完成工作总结

### 1. 项目结构已创建
```
detr_traffic_analysis/
├── dataset/              # 数据加载模块 ✅
│   ├── __init__.py
│   └── coco_dataset.py
├── models/               # DETR模型模块 ✅
│   ├── __init__.py
│   └── detr_model.py
├── utils/                # 工具函数 ✅
│   ├── __init__.py
│   ├── logger.py
│   └── checkpoint.py
├── tools/                # 训练/评估脚本 ✅
│   ├── convert_to_coco.py
│   ├── smoke_test.py
│   ├── validate_coco.py
│   ├── train_detr.py
│   ├── eval_detr.py
│   └── test_framework.py
├── configs/              # 配置文件 ✅
│   ├── classes.yaml
│   ├── detr_baseline.yaml
│   └── detr_smoke.yaml
└── data/                 # 数据目录 ✅
    └── traffic_coco/
        ├── bdd100k_det/  # 已转换完成
        └── tt100k_det/   # 已转换完成
```

### 2. 核心功能已实现

#### ✅ 数据加载器 (`dataset/coco_dataset.py`)
- COCO格式数据集加载
- 图像预处理和归一化
- 支持batch collate
- 返回格式符合DETR要求

#### ✅ DETR模型 (`models/detr_model.py`)
- 基于Hugging Face transformers库
- 加载facebook/detr-resnet-50预训练权重
- 自动调整类别数为3（vehicle, traffic_sign, traffic_light）
- 模型参数: 41.5M (可训练: 41.3M)

#### ✅ 训练框架 (`tools/train_detr.py`)
- 完整训练循环
- 损失计算和反向传播
- 学习率调度
- Checkpoint保存（best.pth / last.pth）
- 日志记录（JSON + CSV格式）
- 进度条显示

#### ✅ 评估脚本 (`tools/eval_detr.py`)
- COCO格式评估
- 计算mAP, AP_50, AP_75, AP_small等指标
- 结果保存为JSON

## 📋 当前状态

### 测试结果 ✅
```
✅ 数据加载器: PASSED
   - 数据集大小: 70,000 (BDD100K训练集)
   - Batch size: 2
   - 数据格式正确

✅ 模型构建: PASSED
   - 预训练模型加载成功
   - 类别数已调整为3
   - 模型大小: 158MB (fp32)

✅ 前向传播: PASSED
   - 输出logits shape: [2, 100, 4]
   - 输出boxes shape: [2, 100, 4]
```

### 环境依赖 ✅
```
✅ torch (2.9.1+cpu)
✅ torchvision (0.24.1+cpu)
✅ transformers (4.57.3)
✅ timm (1.0.22)
✅ pycocotools
✅ scipy
✅ pillow
```

## 🚀 下一步：运行冒烟测试

### 方案1：本地CPU测试（验证流程）

```bash
# 1. 激活虚拟环境
cd /srv/code/detr_traffic_analysis
source .venv/bin/activate

# 2. 运行冒烟配置测试（2个epoch，每epoch 100个iter）
python tools/train_detr.py \
  --config configs/detr_smoke.yaml \
  --output-dir outputs/smoke_test \
  --max-iter 100

# 预计耗时: 10-20分钟（CPU模式）
# 输出：
#   - outputs/smoke_test/
#       ├── config.yaml
#       ├── train.log
#       ├── metrics.json
#       ├── metrics.csv
#       ├── best.pth
#       └── last.pth
```

### 方案2：GPU服务器正式测试（推荐）

#### 步骤1：上传到GPU服务器
```bash
# 压缩项目（排除大文件）
tar -czf detr_traffic.tar.gz \
  --exclude='data/raw' \
  --exclude='data/traffic_coco/*/images' \
  --exclude='outputs' \
  --exclude='.venv' \
  detr_traffic_analysis/

# 上传到GPU服务器
scp detr_traffic.tar.gz user@gpu-server:/path/to/workspace/

# 在GPU服务器解压
ssh user@gpu-server
cd /path/to/workspace
tar -xzf detr_traffic.tar.gz
```

#### 步骤2：GPU服务器环境配置
```bash
cd detr_traffic_analysis

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装GPU版本的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install transformers timm pycocotools pyyaml tqdm scipy pillow
```

#### 步骤3：数据准备
```bash
# 确保数据已准备好（可以从本地rsync同步）
rsync -avz --progress \
  data/traffic_coco/ \
  user@gpu-server:/path/to/workspace/detr_traffic_analysis/data/traffic_coco/
```

#### 步骤4：运行冒烟测试
```bash
# 修改配置使用GPU
# configs/detr_smoke.yaml 中确认:
#   device:
#     type: "cuda"

# 运行冒烟测试
python tools/train_detr.py \
  --config configs/detr_smoke.yaml \
  --output-dir outputs/smoke_test_gpu \
  --max-iter 100

# 预计耗时: 2-5分钟（GPU模式）
```

## 📊 冒烟测试验收标准

### 必须验证的内容
- [ ] DataLoader正常迭代，无报错
- [ ] Loss正常下降或稳定输出
- [ ] Checkpoint保存成功（best.pth / last.pth）
- [ ] 日志完整（metrics.json + metrics.csv）
- [ ] GPU显存占用正常（预计4-6GB）

### 预期输出示例
```
Epoch 1/2
Epoch [1] Iter [10/100] Loss: 15.2341 Avg Loss: 16.1234
Epoch [1] Iter [20/100] Loss: 14.8765 Avg Loss: 15.4567
...
Epoch 1 完成 - Avg Loss: 14.2345, LR: 0.000100
💾 Checkpoint 已保存: outputs/smoke_test_gpu/best.pth

Epoch 2/2
...
✅ 训练完成！
   总耗时: 3.45 分钟
   最佳Loss: 13.8765
```

## 🎯 冒烟测试通过后的下一步

### 1. Baseline全量训练
```bash
# 使用完整配置
python tools/train_detr.py \
  --config configs/detr_baseline.yaml \
  --output-dir outputs/baseline_run

# 预计耗时: 4-8小时 (50 epochs, GPU)
```

### 2. 评估模型
```bash
python tools/eval_detr.py \
  --config configs/detr_baseline.yaml \
  --checkpoint outputs/baseline_run/best.pth \
  --eval-set val \
  --output outputs/baseline_run/eval_results.json
```

### 3. 创建小目标优化配置
基于baseline创建 `configs/detr_small_obj_v1.yaml`：
- 增加输入分辨率
- 启用多尺度训练
- 调整num_feature_levels

### 4. 按照文档中的7步执行清单继续

## 📝 关键配置说明

### detr_smoke.yaml (冒烟测试)
```yaml
training:
  batch_size: 2
  max_epochs: 2
  max_iters: 100      # 每epoch最多100iter
  lr: 1e-4
```

### detr_baseline.yaml (正式训练)
```yaml
training:
  batch_size: 4
  max_epochs: 50
  max_iters: null     # 不限制iter
  lr: 1e-4
```

## ⚠️ 重要提示

1. **显存要求**: 16GB GPU可支持batch_size=4，8GB GPU建议batch_size=2
2. **数据路径**: 确保 `data/traffic_coco/bdd100k_det` 存在且包含images和annotations
3. **权重下载**: 首次运行会自动下载DETR预训练权重（约167MB），需要网络连接
4. **日志输出**: 所有日志统一输出到 `outputs/<experiment>/` 目录

## 🐛 常见问题

### Q1: CUDA out of memory
```bash
# 解决方案: 减小batch size
# 在配置文件中修改:
training:
  batch_size: 2  # 从4改为2
```

### Q2: DataLoader卡住
```bash
# 解决方案: 减少num_workers
training:
  num_workers: 0  # 或设为CPU核心数的一半
```

### Q3: 权重下载失败
```bash
# 解决方案: 手动下载并指定路径
# 或设置代理:
export HF_ENDPOINT=https://hf-mirror.com
```

## 📞 需要帮助？

如果遇到问题，请检查：
1. 日志文件: `outputs/*/train.log`
2. 错误信息的完整stack trace
3. GPU状态: `nvidia-smi`
4. 磁盘空间: `df -h`

---

**状态**: ✅ 所有基础功能测试通过，可以开始训练！
**下一步**: 在GPU服务器运行冒烟测试，验证完整训练流程
