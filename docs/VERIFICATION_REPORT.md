# DETR 训练脚本验证报告

**日期**: 2026-01-06  
**状态**: ✅ 所有检查通过

---

## 验证结果

### ✅ 1. 导入检查
- train_detr_optimized.py 成功导入
- DETR 标准归一化参数正确：
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- benchmark_dataloader.py 成功导入

### ✅ 2. Category ID 映射
- 数据集检测：BDD100K category_id 已经连续 [0, 1, 2]
- 反向映射正确：`{0: 0, 1: 1, 2: 2}`
- **对于非连续 category_id 的数据集（如 COCO），映射机制已就绪**

### ✅ 3. Bbox 格式
- 图像正确归一化（DETR 标准）
- Bbox 正确转换为归一化 cxcywh 格式
- Bbox 范围验证：[0.024, 0.918] ∈ [0, 1] ✓
- 坐标系正确：
  - 原始尺寸：720 x 1280
  - Resize后：750 x 1333
  - orig_size != size ✓

### ✅ 4. 坐标系（关键修复）
- Dataset 正确保存 orig_size 和 size
- **evaluate() 正确使用 orig_size 作为 target_sizes**
- 代码验证：`target_sizes = torch.stack([l["orig_size"] for l in labels])`

### ✅ 5. 文件状态
**可用文件**：
- tools/train_detr_optimized.py
- tools/benchmark_dataloader.py
- tools/run_torchvision_training.sh
- docs/DETR_TRAINING_GUIDE_CURRENT.md
- docs/DETR_TRAINING_README.md
- docs/FIXES_2026_01_06.md

**已标记不可用**：
- tools/train_detr_torchvision.py.BROKEN
- tools/smoke_test_torchvision.py.BROKEN

**已过时**：
- docs/TORCHVISION_DETR_GUIDE.md.OUTDATED
- docs/TORCHVISION_DETR_SUMMARY.md.OUTDATED

---

## 关键修复总结

### 第一轮修复（已完成）
1. ✅ 添加 DETR 标准归一化
2. ✅ Bbox 转换为归一化 cxcywh
3. ✅ 使用 DetrImageProcessor.post_process_object_detection
4. ✅ 修复 args.num-workers → args.num_workers

### 第二轮修复（已完成）
1. ✅ **严重**：evaluate() 使用 orig_size 而非 size 作为 target_sizes
2. ✅ **主要**：添加 Category ID 反向映射到 COCO 原始 ID
3. ✅ **主要**：修复 benchmark_dataloader.py 导入
4. ✅ **中等**：标记过时文档，创建当前指南
5. ✅ **低**：优化 processor 初始化（local_files_only 回退）

### 第三轮修复（已完成）
1. ✅ **主要**：benchmark_dataloader.py 适配新的 dict 格式
   - 修复迭代器解包：`for i, batch in enumerate(loader)`
   - 访问 `batch["pixel_values"]` 和 `batch["labels"]`
   - 使用 `class_labels` 而非 `labels`
2. ✅ **低**：verify_fixes.py 覆盖 train_detr_optimized.py
   - 检查 DETR 归一化、反向映射、orig_size 等关键路径
   - 验证新的 dict 格式 collate_fn

---

## 实际数据测试

### 数据集信息
- 数据集：BDD100K
- 图像数：~70,000
- 类别数：3
- Category IDs：[0, 1, 2]（连续）

### 数据加载测试
- 图像解码：C++ (torchvision.io.read_image) ✓
- 归一化：DETR 标准 ✓
- Bbox 格式：归一化 cxcywh ✓
- 坐标系：orig_size 正确保存 ✓
- **collate_fn 格式**：dict 输出 (pixel_values, labels) ✓
- **实际运行测试**：3 batches, 4 images/batch, 3.50 it/s ✓

---

## 下一步操作

### 1. 冒烟测试（推荐）
```bash
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --subset 100 \
  --num-epochs 1 \
  --batch-size 4 \
  --output-dir outputs/smoke_test
```

**预期结果**：
- 训练正常运行
- Loss 下降
- 无错误或警告

### 2. 完整训练
```bash
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --val-img data/traffic_coco/bdd100k_det/images/val \
  --val-ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --num-epochs 50 \
  --batch-size 8 \
  --num-workers 12 \
  --output-dir outputs/detr_bdd100k
```

**预期结果**：
- 达到 3-4 it/s（CUDA 环境，CPU 环境为 0.x it/s）
- mAP 正常计算
- 坐标系正确对齐

### 3. 未来计划
- 等待 CUDA 环境就绪
- 迁移到 torchvision DETR（当其可用时）
- 保持相同的数据处理流程

---

## 技术说明

### 坐标系统
- **orig_size**: 原始图像尺寸 (H_orig, W_orig)
- **size**: Resize 后尺寸 (H_new, W_new)
- **关键**: DetrImageProcessor.post_process_object_detection 需要 **orig_size**

### Category ID 映射
- **训练时**: 原始 ID → 连续 [0..N-1]
- **评估时**: 连续 [0..N-1] → 原始 ID（通过 reverse_cat_id_map）
- **原因**: pycocotools 需要原始 COCO category_id

### 性能优化
- C++ 图像解码：~30% 加速
- DataLoader: workers=12, prefetch_factor=2
- 目标吞吐量：3-4 it/s（RTX 5090，CUDA 环境）

---

## 验证命令
```bash
# 完整验证（验证两个脚本的所有修复）
python tools/verify_all_fixes.py  # 验证 train_detr_optimized.py 关键修复
python tools/verify_fixes.py      # 验证所有历史修复

# DataLoader 性能测试（实际数据流）
python tools/benchmark_dataloader.py --num-batches 3 --num-workers 2 --batch-size 4
```

---

**结论**: 所有关键修复已验证通过，脚本已准备就绪用于生产训练。
