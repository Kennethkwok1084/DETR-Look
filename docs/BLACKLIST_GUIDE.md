# 损坏图像黑名单使用指南

## 工作流程

### 1. 预扫描生成黑名单（一次性）

```bash
# 扫描训练集
python tools/scan_corrupted_images.py \
  --ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --img-dir data/traffic_coco/bdd100k_det/images/train \
  --output outputs/blacklist_train.json \
  --workers 16

# 扫描验证集
python tools/scan_corrupted_images.py \
  --ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --img-dir data/traffic_coco/bdd100k_det/images/val \
  --output outputs/blacklist_val.json \
  --workers 16
```

**输出示例**：
```
🔍 扫描数据集: data/traffic_coco/bdd100k_det/annotations/instances_train.json
   图像目录: data/traffic_coco/bdd100k_det/images/train
   总图像数: 70,000
   并发线程: 16

检查图像: 100%|████████████| 70000/70000 [02:15<00:00, 516.32it/s]

❌ 发现 12 张损坏图像:
   /path/to/image1.jpg: Image is incomplete or truncated
   /path/to/image2.jpg: RuntimeError: ...
   ...

📝 黑名单已保存: outputs/blacklist_train.json
```

### 2. 训练时使用黑名单

```bash
python tools/train_detr_optimized.py \
  --train-img data/traffic_coco/bdd100k_det/images/train \
  --train-ann data/traffic_coco/bdd100k_det/annotations/instances_train.json \
  --val-img data/traffic_coco/bdd100k_det/images/val \
  --val-ann data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --blacklist outputs/blacklist_train.json \
  --batch-size 16 \
  --num-epochs 50 \
  --device cuda
```

**启动输出**：
```
loading annotations into memory...
Done (t=3.10s)
creating index...
index created!
📋 黑名单过滤: 12 张损坏图像已跳过
```

## 黑名单文件格式

`outputs/blacklist_train.json`:
```json
{
  "annotation_file": "data/.../instances_train.json",
  "image_dir": "data/.../images/train",
  "total_images": 70000,
  "corrupted_count": 12,
  "corrupted_images": [
    {
      "path": "/path/to/corrupted_image1.jpg",
      "error": "RuntimeError: Image is incomplete or truncated"
    },
    {
      "path": "/path/to/corrupted_image2.jpg",
      "error": "RuntimeError: ..."
    }
  ]
}
```

## 优势对比

### 之前（运行时try/except）
```python
try:
    img = read_image(str(img_path), ...)
except Exception as e:
    print(f"⚠️  跳过损坏图像: {img_path}")
    return self.__getitem__((idx + 1) % len(self))
```
- ❌ 每次训练都要重新发现
- ❌ 日志充满警告信息
- ❌ 速度不稳定（随机遇到损坏图像）
- ❌ 递归调用可能导致栈溢出

### 现在（预扫描黑名单）
```python
# 加载时过滤
if blacklist_file and Path(blacklist_file).exists():
    corrupted_paths = load_blacklist(blacklist_file)
    self.ids = filter_out_corrupted(self.ids, corrupted_paths)
```
- ✅ 一次扫描，永久有效
- ✅ 日志清爽（仅显示过滤统计）
- ✅ 速度稳定（无异常处理开销）
- ✅ 安全可靠（无递归风险）

## 性能对比

| 方式 | 首次发现 | 后续训练 | 日志质量 |
|------|----------|----------|----------|
| try/except | 训练中随机 | 每次重复 | 大量警告 |
| 黑名单 | 预扫描2分钟 | 0开销 | 清爽 |

## 建议

1. **大数据集必用**：>10K图像的数据集建议预扫描
2. **定期更新**：数据集更新后重新扫描
3. **CI/CD集成**：数据准备阶段自动扫描
