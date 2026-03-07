# 训练基本功关键 Bug 修复 (2026-01-05)

## 📋 修复概览

本次修复解决了训练基本功实现中的**9个关键问题**，包括1个启动级错误（高优先级）、3个运行时异常（中优先级）和5个配置优化（低优先级）。

---

## 🔴 高优先级修复

### 1. resume_checkpoint UnboundLocalError

**问题描述**：
- `resume_checkpoint` 在定义前被引用，导致 `UnboundLocalError`
- 训练脚本无法启动

**问题代码**：
```python
# tools/train_detr.py (旧版)
is_resume = bool(resume_checkpoint)  # 第230行：使用
metrics_logger = MetricsLogger(output_dir, resume=is_resume)

# ... 50行之后 ...
resume_checkpoint = args.resume or config['training'].get('resume')  # 第278行：定义
```

**修复方案**：
```python
# tools/train_detr.py (新版)
# Resume 检查（在使用前定义）
resume_checkpoint = args.resume or config['training'].get('resume')  # 第229行：定义
is_resume = bool(resume_checkpoint)  # 第230行：使用

# 设置日志（Resume 模式续写）
logger = setup_logger('train', output_dir / 'train.log')
metrics_logger = MetricsLogger(output_dir, resume=is_resume)
```

**验证方法**：
```bash
python tools/train_detr.py --config configs/detr_smoke.yaml --resume outputs/checkpoints/checkpoint_epoch_001.pth
```

---

## 🟡 中优先级修复

### 2. MetricsLogger.get_best() 引用已删除变量

**问题描述**：
- `get_best()` 方法引用 `self.metrics_history`，但该变量已在前期重构中删除
- 调用 `get_best()` 会抛出 `AttributeError`

**问题代码**：
```python
# utils/logger.py (旧版)
def get_best(self, metric_name: str, mode: str = 'max') -> Dict[str, Any]:
    if not self.metrics_history:  # ❌ metrics_history 已删除
        return {}
    
    if mode == 'max':
        best_record = max(self.metrics_history, key=lambda x: x.get(metric_name, float('-inf')))
    else:
        best_record = min(self.metrics_history, key=lambda x: x.get(metric_name, float('inf')))
    
    return best_record
```

**修复方案**：
```python
# utils/logger.py (新版)
def get_best(self, metric_name: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
    if not self.metrics:  # ✅ 使用统一的 self.metrics
        return None
    
    # 过滤出包含该指标的记录
    valid_records = [r for r in self.metrics if metric_name in r]
    if not valid_records:
        return None
    
    if mode == 'max':
        return max(valid_records, key=lambda x: x[metric_name])
    else:
        return min(valid_records, key=lambda x: x[metric_name])
```

**关键变更**：
- `self.metrics_history` → `self.metrics`
- 返回 `None` 而非空字典 `{}`（更符合 Python 惯例）
- 添加 `valid_records` 过滤，避免 `KeyError`

---

### 3. Resume 模式覆盖 CSV 历史记录

**问题描述**：
- Resume 时 `csv_header_written` 初始化为 `False`
- 第一条记录使用 `'w'` 模式写入，覆盖历史 CSV

**问题代码**：
```python
# utils/logger.py (旧版)
def __init__(self, output_dir: Path, resume: bool = False):
    # ... 加载 JSON ...
    
    self.csv_header_written = False  # ❌ Resume 时也是 False
```

**修复方案**：
```python
# utils/logger.py (新版)
def __init__(self, output_dir: Path, resume: bool = False):
    # Resume模式：加载已有指标
    self.metrics = []
    if resume and self.json_path.exists():
        try:
            with open(self.json_path, 'r') as f:
                self.metrics = json.load(f)
            print(f"📂 Resume: 已加载 {len(self.metrics)} 条历史指标")
        except Exception as e:
            print(f"⚠️  无法加载历史指标: {e}，从空列表开始")
            self.metrics = []
    
    # CSV 状态：Resume 时检查是否已有 CSV
    self.csv_header_written = False
    if resume and self.csv_path.exists():
        # 已有 CSV，设置为已写入 header（后续用 append 模式）
        self.csv_header_written = True
        print(f"📂 Resume: 将续写 CSV 文件")
```

**关键逻辑**：
- Resume 且 CSV 存在 → `csv_header_written = True`
- 后续 `log()` 使用 `'a'` 模式追加，不覆盖

---

### 4. Resume 时 JSON/CSV 一致性检测

**问题描述**：
- Resume 时如果 JSON 丢失/损坏但 CSV 仍存在
- JSON 从空列表开始，CSV 继续追加 → 两者不一致

**修复方案**：
```python
# utils/logger.py (新版)
def __init__(self, output_dir: Path, resume: bool = False):
    # Resume模式：加载已有指标
    self.metrics = []
    json_loaded = False
    if resume and self.json_path.exists():
        try:
            with open(self.json_path, 'r') as f:
                self.metrics = json.load(f)
            json_loaded = True
            print(f"📂 Resume: 已加载 {len(self.metrics)} 条历史指标")
        except Exception as e:
            print(f"⚠️  无法加载历史指标: {e}，从空列表开始")
            self.metrics = []
    
    # CSV 状态：Resume 时检查是否已有 CSV
    self.csv_header_written = False
    csv_exists = resume and self.csv_path.exists()
    if csv_exists:
        self.csv_header_written = True
        print(f"📂 Resume: 将续写 CSV 文件")
    
    # 一致性检查：Resume 时 CSV 存在但 JSON 不存在（或加载失败）
    if resume and csv_exists and not json_loaded:
        print(f"⚠️  警告: CSV 存在但 JSON 缺失/损坏")
        print(f"    → CSV 将继续追加，但历史指标无法在 JSON 中体现")
        print(f"    → 建议检查 {self.json_path} 或手动恢复")
```

**关键改进**：
- 增加 `json_loaded` 标志追踪 JSON 加载状态
- 检测 `csv_exists and not json_loaded` 情况
- 输出警告提示用户数据不一致风险

---

## 🟢 低优先级优化

### 5. Progressive Resizing longest_edge 配置化

**问题描述**：
- 原实现强制 `longest_edge = shortest_edge`，可能导致过度压缩
- 无法保留 Deformable DETR 默认的长边上限 1333

**优化前**：
```python
# tools/train_detr.py (旧版)
image_processor.size = {
    "shortest_edge": current_size,
    "longest_edge": current_size  # ❌ 强制等边
}
```

**优化后**：
```python
# tools/train_detr.py (新版)
if isinstance(current_size, dict):
    # 字典格式：{"shortest": 640, "longest": 1333}
    shortest = current_size.get('shortest', 800)
    longest = current_size.get('longest', 1333)
else:
    # 整数格式：短边为该值，长边使用默认上限
    shortest = current_size
    longest = 1333  # Deformable DETR 默认上限

image_processor.size = {"shortest_edge": shortest, "longest_edge": longest}
```

**配置示例**：
```yaml
# configs/detr_baseline.yaml
training:
  resize_schedule:
    # 整数格式（推荐）
    - [0, 640]   # 短边640，长边1333
    - [10, 800]  # 短边800，长边1333
    
    # 或字典格式（精细控制）
    - [0, {"shortest": 640, "longest": 1024}]
    - [10, {"shortest": 800, "longest": 1333}]
```

---

### 5. subset_filter_empty 可配置

**问题描述**：
- 原实现一律过滤空标注样本
- 导致子集分布偏离全量数据（空标注被移除）

**优化前**：
```python
# dataset/coco_dataset.py (旧版)
if subset_size:
    # 筛选有标注的样本（过拟合测试必须有标注）
    valid_indices = [...]  # ❌ 总是过滤
    indices = random.sample(valid_indices, subset_size)
```

**优化后**：
```python
# dataset/coco_dataset.py (新版)
if subset_size and image_set == 'train':
    # 是否过滤空标注样本（默认仅过拟合模式下过滤）
    filter_empty = config['training'].get('subset_filter_empty', overfit_mode)
    
    if filter_empty:
        # 筛选有标注的样本
        pool_indices = [idx for idx in range(len(dataset)) 
                       if dataset[idx][1].get('annotations')]
        print(f"🔍 已过滤空标注样本：{len(dataset)} → {len(pool_indices)} 个有效样本")
    else:
        # 不过滤，保持原始分布
        pool_indices = list(range(len(dataset)))
        print(f"📊 使用全量样本池（包含空标注）：{len(pool_indices)} 个样本")
```

**配置说明**：
- `subset_filter_empty` 未设置 → 默认 `overfit_mode`（过拟合过滤，常规不过滤）
- `subset_filter_empty: true` → 强制过滤空标注
- `subset_filter_empty: false` → 强制保留空标注

---

### 7. subset_filter_empty 配置暴露

**问题描述**：
- 代码中已实现 `subset_filter_empty` 逻辑
- 但 YAML 配置文件未暴露该选项

**修复方案**：
```yaml
# configs/detr_baseline.yaml 和 detr_smoke.yaml
training:
  subset_size: null
  subset_seed: 42
  subset_filter_empty: null  # 新增：是否过滤空标注样本
                             # null=自动，true=强制过滤，false=保持原始分布
                             # 默认：overfit 模式过滤，常规模式不过滤
  overfit: false
```

**使用示例**：
```yaml
# 强制所有样本有标注（即使非 overfit 模式）
training:
  subset_size: 100
  subset_filter_empty: true  # 强制过滤
  overfit: false
```

---

### 8. overfit 显式禁用 transforms

**问题描述**：
- 当前 `make_transforms` 返回 `None`，但未来启用时需要确保过拟合模式下禁用
- 缺少显式检查可能导致过拟合结果不稳定

**修复方案**：
```python
# dataset/coco_dataset.py (新版)
if overfit_mode:
    shuffle = False
    # 显式禁用 transforms（当前 make_transforms 返回 None，但作为防御性检查）
    if transforms is not None:
        print("⚠️  过拟合模式：强制禁用 transforms 以确保可复现性")
        transforms = None
    print(f"📌 过拟合模式：关闭数据增强和打乱")
```

**关键改进**：
- 即使未来启用 transforms，过拟合模式也会强制禁用
- 保证过拟合测试的可复现性

---

### 9. early_stopped 未写入 CSV

**问题描述**：
- `run_trials.py` 记录了 `early_stopped` 标记
- 但 `save_results()` 未将该字段写入 CSV

**优化前**：
```python
# tools/run_trials.py (旧版)
fieldnames = ['trial_id', 'status', 'final_map', 'final_loss', 'output_dir']
# ❌ 缺少 early_stopped
```

**优化后**：
```python
# tools/run_trials.py (新版)
fieldnames = ['trial_id', 'status', 'final_map', 'final_loss', 'early_stopped', 'output_dir']

for result in results:
    row = {
        'trial_id': result['trial_id'],
        'status': result['status'],
        'final_map': result['final_map'],
        'final_loss': result['final_loss'],
        'early_stopped': result.get('early_stopped', False),  # ✅ 写入字段
        'output_dir': result['output_dir'],
    }
```

---

## ✅ 验证清单

### 语法检查
```bash
python tools/syntax_check.py
# ✓ 所有文件通过
```

### 功能验证

1. **Resume 启动**：
```bash
python tools/train_detr.py --config configs/detr_smoke.yaml --resume outputs/checkpoints/checkpoint.pth
# 预期：不抛 UnboundLocalError
```

2. **MetricsLogger.get_best()**：
```bash
python -c "
from utils.logger import MetricsLogger
from pathlib import Path
logger = MetricsLogger(Path('outputs'))
logger.log({'epoch': 1, 'mAP': 0.5})
best = logger.get_best('mAP', mode='max')
print(best)  # 预期：{'epoch': 1, 'mAP': 0.5}
"
```

3. **Resume CSV 续写**：
```bash
# 第一次运行
python tools/train_detr.py --config configs/detr_smoke.yaml --max-epochs 2

# Resume 运行
python tools/train_detr.py --config configs/detr_smoke.yaml --resume outputs/checkpoints/checkpoint_epoch_002.pth --max-epochs 4

# 检查 CSV
head -20 outputs/metrics.csv  # 预期：epoch 1-4 连续，无重复 header
```

4. **Progressive Resizing 配置**：
```bash
# 测试字典格式
python -c "
import yaml
config = yaml.safe_load('''
training:
  resize_schedule:
    - [0, {\"shortest\": 640, \"longest\": 1024}]
    - [5, {\"shortest\": 800, \"longest\": 1333}]
''')
print(config['training']['resize_schedule'])
"
```

5. **subset_filter_empty**：
```bash
# 过拟合模式（默认过滤）
python tools/train_detr.py --config configs/detr_smoke.yaml --subset-size 10 --overfit

# 常规模式（默认不过滤）
python tools/train_detr.py --config configs/detr_smoke.yaml --subset-size 100
```

6. **early_stopped CSV**：
```bash
python tools/run_trials.py --trials-file experiments/trials.json
cat outputs/trial_results.csv | head -1
# 预期：trial_id,status,final_map,final_loss,early_stopped,output_dir,...
```

7. **JSON/CSV 一致性警告**：
```bash
# 模拟场景：删除 JSON 但保留 CSV
rm outputs/metrics.json
python tools/train_detr.py --config configs/detr_smoke.yaml --resume outputs/checkpoints/checkpoint.pth

# 预期输出：
# ⚠️  警告: CSV 存在但 JSON 缺失/损坏
# → CSV 将继续追加，但历史指标无法在 JSON 中体现
```

8. **subset_filter_empty 配置**：
```bash
# 测试配置可读取
python -c "
import yaml
config = yaml.safe_load(open('configs/detr_smoke.yaml'))
print(config['training'].get('subset_filter_empty'))  # 预期：None
"
```

9. **overfit transforms 检查**：
```bash
# 当前 transforms=None，不会触发警告
# 未来启用 transforms 后，overfit 模式会强制禁用
python tools/train_detr.py --config configs/detr_smoke.yaml --subset-size 10 --overfit
# 预期：📌 过拟合模式：关闭数据增强和打乱
```

---

## 📝 修复总结

| 优先级 | Bug | 文件 | 状态 |
|-------|-----|------|------|
| 🔴 高 | resume_checkpoint UnboundLocalError | train_detr.py | ✅ 已修复 |
| 🟡 中 | get_best() 引用 metrics_history | logger.py | ✅ 已修复 |
| 🟡 中 | Resume 覆盖 CSV | logger.py | ✅ 已修复 |
| � 中 | Resume JSON/CSV 一致性检测 | logger.py | ✅ 已修复 |
| 🟢 低 | Progressive Resizing 配置化 | train_detr.py | ✅ 已优化 |
| 🟢 低 | subset_filter_empty 可配置 | coco_dataset.py | ✅ 已优化 |
| 🟢 低 | subset_filter_empty 配置暴露 | detr_*.yaml | ✅ 已优化 |
| 🟢 低 | overfit 显式禁用 transforms | coco_dataset.py | ✅ 已优化 |
| 🟢 低 | early_stopped CSV 字段 | run_trials.py | ✅ 已优化 |

**修复统计**：
- 高优先级（启动级）：1 个
- 中优先级（运行时）：3 个
- 低优先级（配置优化）：5 个
- **总计**：9 个

---

## 🎯 下一步行动

1. ✅ **语法验证** - 已通过
2. ⏭️ **冒烟测试** - `python tools/train_detr.py --config configs/detr_smoke.yaml`
3. ⏭️ **Resume 测试** - 中断后添加 `--resume` 验证续写
4. ⏭️ **过拟合测试** - `--subset-size 10 --overfit` 验证 RNG 和样本过滤
5. ⏭️ **Progressive Resizing 测试** - 配置不同尺寸验证 processor 更新
6. ⏭️ **Trial 搜索测试** - 验证 early_stopped 标记写入 CSV

---

## 📚 相关文档

- [训练基本功指南](TRAINING_GUIDE.md)
- [关键修复 (2026-01-05)](CRITICAL_FIXES_20260105.md)
- [开发指南](develop.md)
