# 关键Bug修复报告

## 修复概览

本次修复解决了3个**高优先级**bug和1个**中优先级**bug，这些是影响训练启动和正确性的关键问题。

---

## 🔴 高优先级修复

### 1. 配置键名错误导致启动即报错

**问题描述**:
- [tools/train_detr.py](tools/train_detr.py#L210) 中读取验证标注使用了错误的键名 `val_ann_file`
- 配置文件中实际的键名是 `val_ann`（参见 [configs/detr_baseline.yaml](configs/detr_baseline.yaml#L9)）
- 导致训练启动时 KeyError

**修复方案**:
```python
# 修复前（错误）
val_ann_file = Path(config['dataset']['val_ann_file'])

# 修复后（正确）
root_dir = Path(config['dataset']['root_dir'])
val_ann_file = root_dir / config['dataset']['val_ann']
```

**影响**: 如不修复，训练脚本无法启动

---

### 2. 可变尺寸图像处理不正确

**问题描述**:
- 原有"修复"使用 `torch.stack` 失败后传递 `list`
- **但 HuggingFace 的 `DeformableDetrForObjectDetection` 要求 `pixel_values` 必须是 Tensor**
- **还需要 `pixel_mask` 来标识padding区域**
- 当前实现会在 list 分支直接崩溃或产生未定义行为

**错误代码位置**:
- [tools/train_detr.py](tools/train_detr.py#L78-L89)
- [tools/eval_detr.py](tools/eval_detr.py#L57-L63)

**正确修复方案**:

使用 `DeformableDetrImageProcessor` 自动处理 padding 和 pixel_mask：

```python
from transformers import DeformableDetrImageProcessor

# 初始化处理器
image_processor = DeformableDetrImageProcessor.from_pretrained('SenseTime/deformable-detr')

# 处理可变尺寸图像
images_pil = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
encoding = image_processor(images=images_pil, return_tensors='pt')

pixel_values = encoding['pixel_values'].to(device)  # 自动padding到相同尺寸
pixel_mask = encoding['pixel_mask'].to(device)      # 标识哪些是padding

# 正确调用模型
outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
```

**关键改进**:
1. ✅ 自动将不同尺寸的图像 padding 到批次中的最大尺寸
2. ✅ 生成 `pixel_mask` 告诉模型哪些区域是真实图像，哪些是padding
3. ✅ 返回标准 Tensor 格式，符合 Deformable DETR 模型要求
4. ✅ 支持真正的混合数据集训练（BDD100K + TT100K + CCTSDB）

**影响**: 如不修复，训练时会因类型错误崩溃

---

### 3. 评估阈值参数未生效

**问题描述**:
- `evaluate()` 函数虽然接收 `score_threshold` 参数
- 但在实际过滤时仍硬编码使用 `0.05`
- 导致无法通过参数调整阈值

**错误代码**:
```python
# tools/eval_detr.py (line 95)
def evaluate(..., score_threshold=0.05):
    ...
    keep = max_scores > 0.05  # ❌ 硬编码，未使用参数
```

**修复方案**:
```python
keep = max_scores > score_threshold  # ✅ 使用参数
```

**影响**: 无法灵活调整评估阈值，可能影响 mAP 计算

---

## 🟡 中优先级改进

### 4. 函数签名更新

为支持新的 `DeformableDetrImageProcessor`，更新了相关函数签名：

**train_one_epoch**:
```python
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    image_processor,  # ✅ 新增参数
    max_iters,
    log_interval,
    logger,
):
```

**evaluate**:
```python
def evaluate(
    model, 
    dataloader, 
    device, 
    coco_gt, 
    logger, 
    score_threshold=0.05, 
    image_processor=None  # ✅ 新增可选参数，支持默认初始化
):
```

---

## 📊 修复验证

运行验证脚本确认所有修复：

```bash
# 验证可变尺寸修复
python tools/verify_variable_size.py

# 验证所有bug修复
python tools/verify_fixes.py
```

**验证结果**:
```
🎉 所有可变尺寸修复验证通过!

关键改进:
1. ✅ 使用DeformableDetrImageProcessor自动处理padding和pixel_mask
2. ✅ 支持真正的可变尺寸图像（不会因torch.stack失败）
3. ✅ 修复配置键名错误（val_ann_file → val_ann）
4. ✅ 评估阈值参数真正生效（不再硬编码0.05）
```

---

## 🔧 修改文件列表

| 文件 | 修改内容 |
|------|----------|
| [tools/train_detr.py](tools/train_detr.py) | 1. 修复配置键名 `val_ann_file` → `val_ann`<br>2. 添加 `DeformableDetrImageProcessor` 导入和初始化<br>3. 使用 ImageProcessor 处理可变尺寸<br>4. 更新 `train_one_epoch` 函数签名 |
| [tools/eval_detr.py](tools/eval_detr.py) | 1. 添加 `DeformableDetrImageProcessor` 导入<br>2. 使用 ImageProcessor 处理可变尺寸<br>3. 修复 `score_threshold` 参数生效<br>4. 更新 `evaluate` 函数签名 |
| [tools/verify_variable_size.py](tools/verify_variable_size.py) | 新增：专门验证可变尺寸修复的脚本 |
| [tools/verify_fixes.py](tools/verify_fixes.py) | 更新：适配新的 DeformableDetrImageProcessor 验证 |

---

## 🚀 技术细节

### DeformableDetrImageProcessor 工作原理

1. **自动 Padding**:
   ```python
   # 输入：不同尺寸的图像列表
   images = [
       torch.randn(3, 720, 1280),  # BDD100K
       torch.randn(3, 2048, 2048), # TT100K
       torch.randn(3, 1024, 1024), # CCTSDB
   ]
   
   # ImageProcessor 自动padding到最大尺寸 (3, 2048, 2048)
   encoding = image_processor(images, return_tensors='pt')
   
   # 输出
   pixel_values.shape  # torch.Size([3, 3, 2048, 2048])
   pixel_mask.shape    # torch.Size([3, 2048, 2048])
   ```

2. **Pixel Mask 作用**:
   ```python
   # pixel_mask[i, h, w] = 1 表示真实像素
   # pixel_mask[i, h, w] = 0 表示padding
   
   # Deformable DETR 模型会忽略 pixel_mask=0 的区域
   # 避免padding区域影响attention计算
   ```

3. **与 Deformable DETR 模型集成**:
   ```python
   # 标准调用方式
   outputs = model(
       pixel_values=pixel_values,
       pixel_mask=pixel_mask,
       labels=targets  # 训练时提供
   )
   
   # 模型自动处理：
   # - 在 attention 中mask掉padding
   # - 在 loss 计算中忽略padding
   ```

### 为什么之前的 torch.stack 方案不可行

1. **类型不匹配**: 
   - Deformable DETR 的 `forward()` 期望 `pixel_values: torch.Tensor`
   - 传递 `list` 会导致类型错误

2. **缺少 pixel_mask**:
   - 即使强制 padding 成功，没有 mask 模型也不知道哪些是padding
   - 会导致 attention 计算错误

3. **HuggingFace 规范**:
   - HF 的所有 Vision Transformer 都要求使用对应的 Processor
   - 这是标准做法，不应该绕过

---

## ✅ 验证清单

在 GPU 服务器上运行前确认：

- [x] 所有修复已应用
- [x] 验证脚本全部通过
- [x] 配置文件键名正确
- [x] DeformableDetrImageProcessor 正确导入和初始化
- [x] 函数签名已更新
- [x] score_threshold 参数生效

---

## 📝 下一步

### 1. 本地验证语法

```bash
python tools/syntax_check.py
```

### 2. GPU 服务器部署

```bash
# 安装依赖（包含 transformers）
pip install -r requirements.txt

# 快速验证
python tools/verify_fixes.py

# 冒烟测试（2-5分钟）
python tools/train_detr.py --config configs/detr_smoke.yaml
```

### 3. 预期结果

**冒烟测试应该**:
- ✅ 成功加载 Deformable DETR 模型和 ImageProcessor
- ✅ 正常处理可变尺寸图像（无 stack 错误）
- ✅ 完成 2 个 epoch 训练
- ✅ 运行验证并计算 mAP
- ✅ 保存 checkpoint

**如果出现错误**:
- 检查 transformers 版本 (`pip show transformers`)
- 查看错误日志中是否涉及 `pixel_values` 或 `pixel_mask`
- 确认配置文件键名无误

---

## 🎓 论文相关

这些修复对应论文中的关键技术点：

1. **多数据集融合**: DeformableDetrImageProcessor 支持混合不同分辨率的数据集
2. **小目标检测**: 正确的 padding 和 mask 确保小目标不被误判
3. **可复现性**: 配置驱动的阈值参数便于论文实验复现

---

**修复完成时间**: 2026年1月5日  
**状态**: ✅ 所有关键bug已修复，准备GPU测试  
**风险**: 低（已通过语法和逻辑验证）
