# HuggingFace Deformable DETR 格式修复总结

**日期**: 2026-01-05  
**修复范围**: 训练/评测数据流的HuggingFace标准对齐

---

## 🎯 修复的问题

### 1. 高优先级：训练时annotations传参格式错误

**问题描述**:  
`train_detr.py` 中传给 `DeformableDetrImageProcessor` 的 `annotations` 参数格式不符合HF预期。

**错误代码**:
```python
# ❌ 错误：只传annotations列表，缺少image_id
annotations = [t['annotations'] for t in targets]
encoding = image_processor(images=images, annotations=annotations, return_tensors='pt')
```

**原因分析**:
- `DeformableDetrImageProcessor` 期望每张图一个完整的dict，包含 `image_id` 和 `annotations`
- 我们的Dataset已经返回了正确的格式：`{'image_id': int, 'annotations': [...]}`
- 但训练代码只拆出了 `annotations` 列表，丢失了 `image_id`
- 导致处理器无法正确匹配图像和标注，可能出现标签错位

**修复方案**:
```python
# ✅ 正确：直接传targets，包含完整的image_id和annotations
encoding = image_processor(
    images=images,
    annotations=targets,  # 直接传targets而非拆分
    return_tensors='pt'
)
```

**影响范围**: `tools/train_detr.py` (lines 80-88)

---

### 2. 中优先级：facebook/前缀重复拼接

**问题描述**:  
`detr_model.py` 中强制拼接 `facebook/` 前缀，若配置里已写 `SenseTime/deformable-detr` 会变成 `facebook/facebook/...`

**错误代码**:
```python
# ❌ 错误：无条件拼接前缀
self.model = DeformableDetrForObjectDetection.from_pretrained(
    f"facebook/{model_config['name']}",  # 如果config['name']已包含facebook/会重复
    num_labels=num_classes,
)
```

**修复方案**:
```python
# ✅ 正确：检查是否已有前缀
model_name = model_config['name']
if not model_name.startswith('facebook/'):
    model_name = f"facebook/{model_name}"
self.model = DeformableDetrForObjectDetection.from_pretrained(
    model_name,
    num_labels=num_classes,
)
```

**支持场景**:
- 配置写 `"deformable-detr"` → 自动补全为 `"SenseTime/deformable-detr"`
- 配置写 `"SenseTime/deformable-detr"` → 保持不变
- 未来支持其他组织的模型：`"hustvl/yolos-tiny"` → 保持不变

**影响范围**: `models/detr_model.py` (lines 33-36)

---

### 3. 中优先级：评测时处理器与模型不一致

**问题描述**:  
`eval_detr.py` 中硬编码处理器为 `SenseTime/deformable-detr`，若后续换模型会导致处理器与模型参数不一致。

**错误代码**:
```python
# ❌ 错误：硬编码模型名称
if image_processor is None:
    image_processor = DeformableDetrImageProcessor.from_pretrained('SenseTime/deformable-detr')
```

**风险**:
- 如果训练时用 `detr-resnet-101`，评测时处理器还是用 `resnet-50` 的参数
- 图像预处理方式不一致，导致评测结果不准确

**修复方案**:
```python
# ✅ 正确：从配置读取，保持一致
if image_processor is None:
    model_name = config['model']['name']
    if not model_name.startswith('facebook/'):
        model_name = f"facebook/{model_name}"
    logger.info(f"初始化DeformableDetrImageProcessor: {model_name}")
    image_processor = DeformableDetrImageProcessor.from_pretrained(model_name)
```

**影响范围**: `tools/eval_detr.py` (lines 57-60)

---

### 4. 低优先级：数据增强配置被忽略

**问题描述**:  
`make_transforms` 返回 `None`，配置中的数据增强参数（`random_horizontal_flip`, `color_jitter`）未被使用。

**现状**:
```python
def make_transforms(image_set: str, config: dict) -> Any:
    # DeformableDetrImageProcessor会自动处理resize/normalize
    # 这里不做任何变换，直接返回None
    return None
```

**说明**:
- 当前设计：`DeformableDetrImageProcessor` 统一处理 resize/pad/normalize
- 额外增强（flip/jitter）需要在Dataset中对PIL图像应用，然后再传给processor
- 不是bug，但需要文档说明如何添加增强

**解决方案**:
1. ✅ 已扩充 `make_transforms` 的文档说明
2. ✅ 提供了参考实现：`docs/data_augmentation_guide.py`
3. 推荐使用 `albumentations` 处理需要同步bbox的几何变换

**影响范围**: `dataset/coco_dataset.py` (lines 93-108)

---

## ✅ 验证方法

运行验证脚本：
```bash
python3 tools/verify_hf_format.py
```

**验证点**:
1. ✅ annotations传参：直接传targets（含image_id+annotations）
2. ✅ facebook/前缀：自动判断，避免重复
3. ✅ processor一致性：从配置读取，与模型保持一致
4. ✅ 数据增强文档：完整说明如何添加及与processor协作

---

## 📋 修改文件清单

| 文件 | 修改内容 | 影响 |
|------|---------|------|
| `tools/train_detr.py` | annotations传参改为直接传targets | 🔴 高 - 修复训练标签错位 |
| `models/detr_model.py` | 添加facebook/前缀判断 | 🟡 中 - 支持多种配置格式 |
| `tools/eval_detr.py` | 从配置读取模型名称 | 🟡 中 - 保证评测一致性 |
| `dataset/coco_dataset.py` | 扩充数据增强文档 | 🟢 低 - 改善可维护性 |
| `docs/data_augmentation_guide.py` | 新增增强参考实现 | 🟢 低 - 开发指南 |
| `tools/verify_hf_format.py` | 新增格式验证脚本 | 🟢 低 - 质量保证 |

---

## 🚀 后续步骤

### 必做
1. ✅ 语法检查通过
2. ✅ 格式验证通过
3. ⏳ GPU服务器部署
4. ⏳ 冒烟测试（2 epoch，验证数据流正确）

### 可选（根据实验需求）
- 如需数据增强：参考 `docs/data_augmentation_guide.py` 实现
- 如需细粒度类别：修改 `configs/classes.yaml` 和类别映射逻辑
- 如需换其他模型：配置中直接写完整模型名（如 `hustvl/yolos-tiny`）

---

## 🔍 关键设计决策

### 为什么直接传targets？
- HF的 `DeformableDetrImageProcessor` 需要 `image_id` 来正确关联图像和标注
- 我们的Dataset已经返回了符合HF预期的格式
- 训练代码只需透传，不需要拆分重组

### 为什么要前缀判断？
- 支持多种配置风格：简写（`deformable-detr`）或完整名（`SenseTime/deformable-detr`）
- 未来可能用其他组织的模型（如 `hustvl/yolos-tiny`），不能强制加 `facebook/`
- 统一处理逻辑，避免硬编码

### 为什么processor要从配置读？
- 训练和评测必须使用相同的预处理参数
- 不同Deformable DETR变体（resnet-50/101, DC5等）的预处理参数可能不同
- 硬编码会导致模型升级时忘记同步更新

### 为什么数据增强返回None？
- `DeformableDetrImageProcessor` 已经处理了resize/pad/normalize，这是标准化预处理
- 额外的增强（flip/jitter）是可选的，应该在processor之前对PIL图像应用
- 返回None简化了默认流程，需要增强时再启用

---

## 📚 参考资料

- [HuggingFace Deformable DETR文档](https://huggingface.co/docs/transformers/model_doc/deformable_detr)
- [DeformableDetrImageProcessor API](https://huggingface.co/docs/transformers/model_doc/deformable_detr#transformers.DeformableDetrImageProcessor)
- [数据增强指南](docs/data_augmentation_guide.py)
- [验证脚本](tools/verify_hf_format.py)
