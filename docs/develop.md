# 开发与实现说明（develop.md）

说明对象：开发者 / 维护者 / 答辩技术细节展示
目的：完整对齐毕业论文中的系统设计、实现与实验指标，指导如何复现论文中的所有实验结果。

---

## 0. 快速开始指南

### 0.1 10分钟快速体验

```bash
# 1. 克隆/进入项目目录
cd /srv/code/detr_traffic_analysis

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装基础依赖
pip install pycocotools pyyaml tqdm

# 4. 创建项目结构
mkdir -p tools configs data/{raw,traffic_coco} outputs

# 5. 查看已创建的配置文件
cat configs/classes.yaml
cat configs/detr_baseline.yaml

# 6. 准备测试数据（假设已有BDD100K数据）
# 运行转换脚本（详见第3节）
python tools/convert_to_coco.py --help

# 7. 运行冒烟测试
python tools/smoke_test.py
```

### 0.2 项目实施路线图

```
第1阶段：数据准备（1-2天）
  ├─ 创建项目目录结构 ✅
  ├─ 配置类别映射 ✅
  ├─ BDD100K → COCO转换 ✅
  └─ 数据验证与冒烟测试 ✅

第2阶段：检测模型训练（3-5天）
  ├─ 实现数据加载器
  ├─ 实现DETR模型
  ├─ Baseline训练
  ├─ 小目标优化训练
  └─ 检测指标评测

第3阶段：跟踪模块集成（2-3天）
  ├─ 实现ByteTrack封装
  ├─ 实现OC-SORT封装
  ├─ 生成跟踪结果
  └─ MOT指标评测

第4阶段：系统集成与可视化（3-4天）
  ├─ Streamlit界面开发
  ├─ 书签与回放功能
  ├─ 结果导出
  └─ 性能基准测试

第5阶段：实验与论文撰写（5-7天）
  ├─ 完整实验运行
  ├─ 数据整理与分析
  ├─ 论文图表制作
  └─ 演示视频录制
```

### 0.3 已完成工作概览

#### ✅ 项目架构
- 目录结构：`tools/`, `configs/`, `data/`, `outputs/`
- 配置文件：类别映射、训练基础配置

#### ✅ 数据准备工具
- `tools/convert_to_coco.py`：BDD100K转COCO格式，支持类别映射
- `tools/validate_coco.py`：COCO数据集完整性验证
- `tools/smoke_test.py`：快速冒烟测试

#### ✅ 配置系统
- `configs/classes.yaml`：3类粗粒度映射（vehicle/traffic_sign/traffic_light）
- `configs/detr_baseline.yaml`：完整训练配置模板

#### ✅ 训练框架
- `tools/train_detr.py`：训练脚本框架（待完善模型实现）

#### 📋 待开发模块
- 数据加载器（Dataset/DataLoader）
- DETR模型实现（backbone/transformer/heads）
- 训练循环与评估逻辑
- 跟踪器封装
- Streamlit前端
- 完整评测流程

### 0.4 执行节奏与闭环（含冒烟）

节奏原则：单变量、可复现、每步有产出（日志/配置/权重）。

执行约束：
- 训练框架基于现有 DETR 实现，保持论文题目一致性。
- 日志输出优先 `metrics.json` 或 `metrics.csv`，字段保持统一。
- 执行前确认 `data/traffic_coco` 中已有可用数据集（当前仓库检测到 `bdd100k_det` 与 `tt100k_det`）。

1) **数据/映射唯一真源**：
   - `configs/classes.yaml` 作为唯一类别表与映射规则来源。
   - `convert_to_coco.py` 启动时打印 `original_name -> coarse_name -> class_id` 映射，并在 `mapping.json` 落盘。
2) **数据转换 + 冒烟**：
   - 完成 BDD100K → COCO 转换后，立刻用 `pycocotools` 做加载+评测冒烟（见 3.6.3）。
   - 同步运行 `tools/smoke_test.py`，确认 COCO 结构与类别分布正常。
3) **10 分钟级训练冒烟**：
   - 用少量 iter/1-2 epoch 验证 dataloader、loss、保存、eval 全链路。
4) **Baseline 完整训练**：
   - 固化 `configs/detr_baseline.yaml`，记录 mAP/AP_small、耗时、显存与推理 FPS。
5) **small_obj 单变量消融**：
   - 顺序：输入尺度/多尺度 → `num_feature_levels` → 结构性改动；每次只改一个变量。
6) **跟踪闭环（阈值固化）**：
   - 用最佳 detector 权重跑 ByteTrack/OC-SORT，记录展示阈值与关联阈值。
7) **Streamlit 体验闭环**：
   - 默认优先“缓存回放”，实时推理才启用抽帧/降分辨率/限刷新。
   - 书签与导出流程必须在同一次回放中验证通过。
8) **性能基准**：
   - 完成 baseline vs small_obj 推理配置对比，给出推荐部署参数。

当前进度：第 1-2 步已完成。

---

### 0.5 执行清单（详细步骤）

本清单用于研发执行与验收，按阶段逐条勾选。

**阶段 A：数据确认与环境准备**
1) 确认 `data/traffic_coco/bdd100k_det` 与 `data/traffic_coco/tt100k_det` 存在。
2) 确认 `annotations/instances_train.json` 与 `annotations/instances_val.json` 存在。
3) 冒烟通过：`pycocotools` 能加载，类别数与 `configs/classes.yaml` 一致。
4) `tools/smoke_test.py` 通过，输出图像数、标注数、类别分布。

**阶段 B：训练冒烟（10 分钟级）**
1) 选用 smoke 配置（基于 baseline 缩小规模）。
2) 控制训练规模：
   - `num_epochs=1~2` 或固定 `max_iters`；
   - 可选：子集训练（100-500 张）或降低分辨率/batch。
3) 验证链路：
   - dataloader 正常迭代；
   - loss 正常输出；
   - eval 能跑通；
   - `best.pth` / `last.pth` 保存正常。
4) 日志落盘：`outputs/smoke_run/metrics.json` 或 `metrics.csv`。

**阶段 C：Baseline 全量训练闭环**
1) 固化 `configs/detr_baseline.yaml` 作为对照基线。
2) 完整训练 + 评测：
   - 训练日志（loss/lr/耗时）；
   - 验证指标（mAP/AP_small）；
   - 权重保存（best/last）。
3) 输出目录：`outputs/baseline_run/`。

**阶段 D：small_obj 单变量消融**
1) 严格顺序：输入尺度/多尺度 → `num_feature_levels` → 结构性改动。
2) 每轮只改一个变量，其余保持 baseline 不变。
3) 每轮输出：
   - 配置：`configs/detr_small_obj_v*.yaml`；
   - 结果：`outputs/small_obj_run_v*/`；
   - 记录：AP/AP_small/耗时/显存/FPS。
4) 汇总：`experiments/ablation_small_obj.csv`。

**阶段 E：跟踪闭环（ByteTrack / OC-SORT）**
1) 选用最佳 detector 权重作为输入。
2) 生成 MOT 结果：`tools/inference_tracks.py`。
3) 指标评测：`tools/eval_mot.py` 输出 HOTA/IDF1/MOTA。
4) 阈值固化并落盘：
   - 展示阈值：`detector_score_thresh`；
   - 关联阈值：`tracker_match_thresh` / `tracker_low_score`。

**阶段 F：Streamlit 体验闭环**
1) 默认“缓存回放”，优先播放已有结果。
2) 实时推理仅在必要时开启，并配合抽帧/降分辨率/限刷新。
3) 书签、回放、导出功能在缓存与实时模式各验证一次。

**阶段 G：性能基准与推荐配置**
1) baseline vs small_obj 做性能对比（FPS/Latency/VRAM + mAP/AP_small）。
2) 输出推荐推理配置（分辨率/阈值/batch），区分演示/实时/离线场景。

---

## 1. 开发环境与依赖

### 1.1 硬件环境

* GPU：NVIDIA RTX 3060 或以上，显存 ≥ 8 GB（建议）
* CPU：4 核及以上
* 内存：16 GB 及以上
* 磁盘：≥ 50 GB（数据集 + 权重 + 日志）

> 提示：若显存为 6 GB，可适当减小 batch size 或分辨率，但可能影响训练稳定性与 AP 指标。

### 1.2 软件环境

* 操作系统：Windows 10/11 或 Ubuntu 20.04+
* Python：3.10.x
* 虚拟环境：推荐 Conda / venv

关键依赖（实际版本以 `requirements.txt` 为准）：

* 深度学习与数据处理：

  * `torch`
  * `torchvision`
  * `numpy`
  * `pycocotools`
  * `tqdm`
  * `pyyaml`
* 可视化与前端：

  * `streamlit`
  * `opencv-python`
  * `matplotlib`
* 跟踪与评测：

  * `trackeval`（多目标跟踪指标计算）
* 日志与工具：

  * `loguru` 或 Python 内置 `logging`

安装示例：

```bash
conda create -n detr_traffic python=3.10
conda activate detr_traffic
pip install -r requirements.txt
```

使用 uv（可选）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## 2. 代码结构与模块说明

本节对主要目录与模块职责进行说明，便于维护与扩展。

### 2.1 顶层结构

* `app/`：Streamlit 前端入口与交互逻辑；
* `models/`：Deformable DETR 模型定义与构建；
* `tracker/`：多目标跟踪算法封装；
* `viz/`：可视化绘制函数；
* `video_io/`：视频读取与控制；
* `tools/`：训练、评测、推理与基准测试脚本；
* `configs/`：模型与训练配置文件；
* `data/`：数据集（原始与转换后）；
* `outputs/`：权重、日志与结果；
* `experiments/`：实验记录与论文相关数据（可选）。

### 2.2 `app/`：前端与交互

文件：`app/app_streamlit.py`

主要职责：

* 构建 Streamlit 页面布局：侧边栏参数面板 + 主视频展示区域；
* 管理 `st.session_state`：

  * `current_frame`：当前帧索引；
  * `bookmarks`：书签列表（包含 frame_id、备注）；
  * `filters`：类别筛选、显示开关等；
* 对接后端推理接口：

  * 初始化模型与跟踪器实例；
  * 循环处理视频，每帧调用推理与可视化函数；
* 提供书签添加、跳转回放与结果导出按钮。

### 2.3 `models/`：Deformable DETR 模型

典型文件：

* `detr_backbone.py`：骨干网络 + 特征金字塔；
* `detr_heads.py`：分类与回归头；
* `build_model.py`：根据配置构建完整模型；
* `__init__.py`：对外暴露 `build_detr_model(cfg)` 接口。

核心设计要点：

* 支持多尺度特征输入（如 4 层：P2–P5）；
* 支持通过配置切换：

  * 是否加载 COCO 预训练权重；
  * 编解码器层数、注意力头数；
  * 小目标优化相关参数（见第 4 节）。

### 2.4 `tracker/`：多目标跟踪

典型文件：

* `base_tracker.py`：定义跟踪器抽象类，约定 `update(detections)` 接口；
* `bytetrack_wrapper.py`：对 ByteTrack 进行封装；
* `ocsort_wrapper.py`：对 OC-SORT 进行封装；
* `utils.py`：轨迹缓存结构、ID 管理工具等。

输入输出约定：

* 输入：当前帧检测结果 `detections`，格式如：

  * `[ [x1, y1, x2, y2, score, class_id], ... ]`；
* 输出：附带 `track_id` 的结果列表，用于绘制轨迹与导出。

### 2.5 `viz/`：可视化绘制

文件：`viz/drawer.py`

主要职能：

* 根据当前帧图像与检测/跟踪结果，绘制：

  * 边界框（bbox）；
  * 类别与置信度文本；
  * 轨迹尾迹（根据历史中心点）；
* 根据用户设置（显示开关、类别过滤）选择性绘制不同元素；
* 输出适合在 Streamlit 中展示的图像（`numpy` 数组或 `PIL.Image`）。

### 2.6 `video_io/`：视频读取

文件：`video_io/video_reader.py`

职责：

* 使用 OpenCV 打开视频文件；
* 提供按帧读取接口，如：`read_frame(idx)`；
* 支持根据帧索引或时间戳进行跳转（`seek(frame_id)`）。

### 2.7 `tools/`：训练、评测与工具脚本

主要脚本示例：

* `convert_to_coco.py`：原始数据 → COCO JSON；
* `train_detr.py`：模型训练；
* `eval_detr.py`：检测指标评测（mAP / AP_small 等）；
* `inference_tracks.py`：生成带 track_id 的跟踪结果文件；
* `eval_mot.py`：基于 TrackEval 计算 HOTA / IDF1 / MOTA；
* `benchmark_system.py`：系统性能测试（FPS / Latency / VRAM）。

---

## 3. 数据准备与 COCO 转换

### 3.1 项目目录架构创建

在开始数据准备前，首先需要创建完整的项目目录结构：

```bash
# 创建核心目录
mkdir -p tools configs data/{raw,traffic_coco} outputs
```

完整目录结构：

```text
detr_traffic_analysis/
├── tools/              # 工具脚本目录
│   ├── convert_to_coco.py      # BDD100K转COCO格式
│   ├── validate_coco.py        # COCO数据集验证
│   ├── smoke_test.py          # 快速冒烟测试
│   └── train_detr.py          # 训练脚本框架
├── configs/            # 配置文件目录
│   ├── classes.yaml           # 类别映射配置
│   └── detr_baseline.yaml     # 基础训练配置
├── data/               # 数据目录
│   ├── raw/                   # 原始数据集
│   └── traffic_coco/          # 转换后的COCO格式数据
├── outputs/            # 输出目录（权重、日志）
├── app/                # Streamlit前端（后续创建）
├── models/             # 模型定义（后续创建）
├── tracker/            # 跟踪器封装（后续创建）
└── viz/                # 可视化模块（后续创建）
```

### 3.2 原始数据集

* 推荐使用 BDD100K / TT100K 等交通场景数据集；
* 要求数据集中包含：车辆与交通标志类标注；
* 将原始数据存放于 `data/raw/` 目录下。

BDD100K数据集结构示例：

```text
data/raw/bdd100k/
├── images/
│   └── 100k/
│       ├── train/    # 训练集图像
│       ├── val/      # 验证集图像
│       └── test/     # 测试集图像
└── labels/
    ├── det_20/            # 新版子目录
    │   ├── det_train.json
    │   ├── det_val.json
    │   └── det_test.json
    ├── det_train.json     # 旧版扁平命名
    ├── det_val.json
    ├── det_test.json
    ├── det_20_train.json  # 带版本号扁平命名
    ├── det_20_val.json
    └── det_20_test.json
```

BDD100K逐图标注结构（可选，逐图 JSON）：

```text
data/raw/bdd100k/
└── labels/
    ├── bdd100k/
    │   ├── train/*.json
    │   ├── val/*.json
    │   └── test/*.json
    └── bd100k/             # 历史命名兼容
        ├── train/*.json
        ├── val/*.json
        └── test/*.json
```

CCTSDB数据集结构示例（VOC XML）：

```text
data/raw/cctsdb/
├── images/
│   ├── train/
│   ├── val/        # 或 test/
│   └── test/
└── labels/
    └── xml/        # VOC标注
        ├── train/*.xml
        ├── val/*.xml
        └── test/*.xml
```

TT100K数据集结构示例：

```text
data/raw/tt100k/
├── annotations_all.json
├── train/
├── test/
└── other/
```

> 说明：实际路径以 `--src` 参数为准，推荐统一放在 `data/raw/` 目录下。

### 3.3 类别映射配置

#### 3.3.1 粗粒度类别定义

在 `configs/classes.yaml` 中定义了3个粗粒度类别（配置驱动）：

```yaml
# 粗粒度类别定义（ID -> 名称）
COARSE_CLASSES:
  0: vehicle        # 包含 car, bus, truck 等交通工具
  1: traffic_sign   # 路牌/交通标志
  2: traffic_light  # 红绿灯

# BDD100K原始类别到粗粒度类别的映射
BDD100K_MAPPING:
  car: vehicle
  bus: vehicle
  truck: vehicle
  traffic sign: traffic_sign
  traffic light: traffic_light

# CCTSDB类别到粗粒度类别的映射
CCTSDB_MAPPING:
  prohibitory: traffic_sign
  warning: traffic_sign
  mandatory: traffic_sign

# TT100K目标类别（所有标志统一映射到该类）
TT100K_TARGET: traffic_sign

# 映射选项
MAPPING_OPTIONS:
  include_bike: false      # 是否将自行车并入指定类别
  bike_target: vehicle     # 自行车映射到的目标类别
  include_motor: false     # 是否将摩托车并入指定类别
  motor_target: vehicle    # 摩托车映射到的目标类别
  min_area: 0              # 最小bbox面积过滤阈值（像素）
```

**说明**：配置文件是类别映射的唯一真源，`convert_to_coco.py` 从配置读取映射规则。修改映射只需修改配置文件。

#### 3.3.2 类别映射设计说明

1. **粗粒度分类原因**：
   - 聚焦交通场景核心对象，减少类别数提高检测精度
   - 便于统一处理不同数据集的细粒度类别差异
   - 符合实际应用需求（车辆检测、交通设施识别）

2. **可配置映射选项**：
   - 通过 `MAPPING_OPTIONS` 控制两轮车是否并入指定目标类别
   - `bike_target` / `motor_target` 必须存在于 `COARSE_CLASSES`
   - 支持最小面积过滤，排除过小的噪声标注
   - 便于论文中进行消融实验

3. **小目标定义**：
   - 目标面积 < 32×32 像素视为小目标
   - 在COCO评测中对应 AP_small 指标
   - 交通标志通常属于小目标范畴

### 3.4 转换为 COCO 格式

#### 3.4.1 转换脚本功能

脚本：`tools/convert_to_coco.py`

**职责说明（配置驱动设计）**：
- 类别映射从 `configs/classes.yaml` 读取（配置是唯一真源）
- 启动时验证配置合法性（ID连续性、映射完整性）
- 修改类别映射只需修改配置文件，无需改代码
 - `mapping.json` 记录完整配置快照（路径 + 内容），便于复现

核心功能：
1. **类别映射**：BDD100K原始类别 → 粗粒度类别 → class_id
2. **格式转换**：BDD100K JSON → COCO格式JSON
3. **统计输出**：总图片数/标注数/类别计数
4. **映射记录**：生成 `mapping.json` 用于论文复现

#### 3.4.2 使用示例

基础转换命令（以 BDD100K 为例）：

```bash
python tools/convert_to_coco.py \
  --dataset bdd100k \
  --src data/raw/bdd100k \
  --dst data/traffic_coco/bdd100k_det \
  --splits train val
```

完整参数说明：

```bash
python tools/convert_to_coco.py \
  --dataset bdd100k \                   # 数据集类型（bdd100k/cctsdb/tt100k）
  --src data/raw/bdd100k \              # 数据集根目录
  --dst data/traffic_coco/bdd100k_det \ # COCO格式输出目录
  --config configs/classes.yaml \       # 类别配置文件
  --min-area 0.0 \                      # 最小bbox面积阈值
  --splits train val                    # 要转换的数据集划分
```

其他数据集示例：

```bash
# CCTSDB（VOC XML）
python tools/convert_to_coco.py \
  --dataset cctsdb \
  --src data/raw/cctsdb \
  --dst data/traffic_coco/cctsdb_det \
  --config configs/classes.yaml \
  --splits train test

# TT100K（annotations_all.json）
python tools/convert_to_coco.py \
  --dataset tt100k \
  --src data/raw/tt100k \
  --dst data/traffic_coco/tt100k_det \
  --config configs/classes.yaml \
  --splits train test
```

#### 3.4.3 预期输出结构

转换完成后的目录结构：

```text
data/traffic_coco/bdd100k_det/
├── images/
│   ├── train/          # 训练集图像（复制或软链接）
│   ├── val/            # 验证集图像
│   └── test/           # 测试集图像
├── annotations/
│   ├── instances_train.json  # COCO格式训练集标注
│   ├── instances_val.json    # COCO格式验证集标注
│   └── instances_test.json   # COCO格式测试集标注
└── mapping.json        # 类别映射记录
```

#### 3.4.4 转换输出示例

控制台输出示例：

```text
================================================================
📋 配置摘要
================================================================

粗粒度类别:
  [0] vehicle
  [1] traffic_sign
  [2] traffic_light

BDD100K映射规则 (5 个):
  'car' -> 'vehicle' (ID: 0)
  'bus' -> 'vehicle' (ID: 0)
  ...
================================================================

🚀 BDD100K → COCO 转换工具
================================================================
源目录: data/raw/bdd100k
目标目录: data/traffic_coco/bdd100k_det
最小面积: 0.0 像素²
转换划分: train, val
================================================================

📂 加载 train 集标注: data/raw/bdd100k/labels/det_20_train.json
🔄 转换 train 集...
Processing train: 100%|████████| 69863/69863 [02:34<00:00]
💾 保存标注文件: data/traffic_coco/bdd100k_det/annotations/instances_train.json

================================================================
📊 转换摘要
================================================================

【TRAIN 集】
  总图像数: 69,863
  总标注数: 456,234
  过滤标注数: 0
  类别分布:
    [0] vehicle: 398,567
    [1] traffic_sign: 45,123
    [2] traffic_light: 12,544

【VAL 集】
  总图像数: 10,000
  总标注数: 65,432
  过滤标注数: 0
  类别分布:
    [0] vehicle: 57,890
    [1] traffic_sign: 6,234
    [2] traffic_light: 1,308
    
✅ 转换完成！输出目录: data/traffic_coco/bdd100k_det
```

#### 3.4.5 mapping.json 内容

生成的 `mapping.json` 记录完整映射关系和配置快照：

```json
{
 "class_mapping": {
    "bdd100k_to_coarse": {
      "car": "vehicle",
      "bus": "vehicle",
      "truck": "vehicle",
      "traffic sign": "traffic_sign",
      "traffic light": "traffic_light"
    },
    "coarse_to_id": {
      "vehicle": 0,
      "traffic_sign": 1,
      "traffic_light": 2
    }
  },
  "config_snapshot": {
    "path": "configs/classes.yaml",
    "content_yaml": "COARSE_CLASSES:\\n  0: vehicle\\n  1: traffic_sign\\n  2: traffic_light\\n...",
    "content_dict": {
      "COARSE_CLASSES": {
        "0": "vehicle",
        "1": "traffic_sign",
        "2": "traffic_light"
      },
      "BDD100K_MAPPING": {
        "car": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "traffic sign": "traffic_sign",
        "traffic light": "traffic_light"
      },
      "CCTSDB_MAPPING": {
        "prohibitory": "traffic_sign",
        "warning": "traffic_sign",
        "mandatory": "traffic_sign"
      },
      "TT100K_TARGET": "traffic_sign",
      "MAPPING_OPTIONS": {
        "include_bike": false,
        "bike_target": "vehicle",
        "include_motor": false,
        "motor_target": "vehicle",
        "min_area": 0
      }
    }
  },
  "statistics": {
    "train": {
      "total_images": 69863,
      "total_annotations": 456234,
      "filtered_annotations": 0,
      "class_counts": {
        "vehicle": 398567,
        "traffic_sign": 45123,
        "traffic_light": 12544
      },
      "original_class_counts": {
        "car": 352341,
        "bus": 23456,
        "truck": 22770,
        "traffic sign": 45123,
        "traffic light": 12544
      },
      "unmapped_classes": []
    }
  }
}
```

### 3.5 第1-2步技术总结（论文复现依据）

本节用于论文撰写与复现实验的“可追溯依据”，总结第1-2步的数据准备与转换实现。

#### 3.5.1 设计目标与边界

- **统一类别口径**：以 `COARSE_CLASSES` 作为跨数据集统一类别表。
- **配置驱动**：类别映射与可选开关由 `configs/classes.yaml` 驱动，代码仅负责实现逻辑与校验。
- **可复现**：每次转换输出 `mapping.json`，包含配置快照与统计信息。

#### 3.5.2 配置结构与约束

```yaml
COARSE_CLASSES:            # 约束：ID 从 0 连续
  0: vehicle
  1: traffic_sign
  2: traffic_light

BDD100K_MAPPING:           # 约束：value 必须存在于 COARSE_CLASSES
  car: vehicle
  bus: vehicle
  truck: vehicle
  traffic sign: traffic_sign
  traffic light: traffic_light

CCTSDB_MAPPING:            # CCTSDB XML 类别映射
  prohibitory: traffic_sign
  warning: traffic_sign
  mandatory: traffic_sign

TT100K_TARGET: traffic_sign  # TT100K 全量映射目标类

MAPPING_OPTIONS:
  include_bike: false
  bike_target: vehicle     # 目标类别必须存在于 COARSE_CLASSES
  include_motor: false
  motor_target: vehicle    # 目标类别必须存在于 COARSE_CLASSES
  min_area: 0              # 过滤过小标注的面积阈值
```

#### 3.5.3 数据集适配与解析策略

- **BDD100K**：
  - 支持官方 JSON 与逐图 JSON 两种标注形式。
  - 标注路径兼容：`labels/det_{split}.json` → `labels/det_20/det_{split}.json` → `labels/det_20_{split}.json`。
  - 逐图标注兼容：`labels/bdd100k/{split}/*.json` 或 `labels/bd100k/{split}/*.json`。
  - 图像目录兼容：`images/100k/{split}` → `images/{split}` → `{split}`。
  - 图像尺寸使用 BDD100K 固定分辨率（1280×720）。

- **CCTSDB**：
  - 解析 VOC XML (`labels/xml` 或 `xml` 目录)。
  - 从 XML 读取图像尺寸，缺失时使用 OpenCV 读取真实尺寸。
  - 类别映射由 `CCTSDB_MAPPING` 驱动，全部归入 `traffic_sign`。

- **TT100K**：
  - 解析 `annotations_all.json`（字段：`imgs`）。
  - 按 `path` 前缀筛选 split（`train/`、`test/`）。
  - 使用数值排序生成稳定 `image_id`，并提供异常兜底排序，确保可复现。
  - 所有类别统一映射到 `TT100K_TARGET`。

#### 3.5.4 转换产出与复现凭证

每个数据集的输出目录结构一致：

```text
data/traffic_coco/<dataset>_det/
├── images/{train,val,test}
├── annotations/instances_{split}.json
└── mapping.json
```

`mapping.json` 记录三类关键信息：

1. **class_mapping**：`bdd100k_to_coarse` / `cctsdb_to_coarse` / `tt100k_to_coarse` + `coarse_to_id`，形成 `original_name -> coarse_name -> class_id` 的可复现链路。
2. **statistics**：每个 split 的图像数、标注数、类别分布与未映射类别。
3. **config_snapshot**：配置文件路径 + YAML 原文 + 解析后的字典快照。

#### 3.5.5 转换与验证流程（第2步闭环）

1) **转换（COCO 生成）**：

```bash
python tools/convert_to_coco.py \
  --dataset bdd100k \
  --src data/raw/bdd100k \
  --dst data/traffic_coco/bdd100k_det \
  --config configs/classes.yaml \
  --splits train val
```

2) **冒烟测试（10 秒级）**：

```bash
python tools/smoke_test.py data/traffic_coco/bdd100k_det/annotations/instances_val.json
```

3) **pycocotools 加载+评测冒烟**：见 3.6.3-3.6.4 的命令片段，验证 COCO 加载与评测链路。

4) **完整验证（COCO 统计）**：

```bash
python tools/validate_coco.py \
  --ann-file data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --check-images
```

### 3.6 COCO数据集冒烟测试

#### 3.6.1 快速验证脚本

使用 `tools/smoke_test.py` 快速验证转换结果：

```bash
# 验证默认路径（验证集）
python tools/smoke_test.py

# 验证指定文件
python tools/smoke_test.py data/traffic_coco/bdd100k_det/annotations/instances_train.json
```

预期输出：

```text
🔥 冒烟测试: data/traffic_coco/bdd100k_det/annotations/instances_val.json

loading annotations into memory...
Done (t=0.52s)
creating index...
index created!

✅ 加载成功!
   图像数: 10,000
   标注数: 65,432
   类别数: 3
   类别映射: {0: 'vehicle', 1: 'traffic_sign', 2: 'traffic_light'}

   类别分布:
     [0] vehicle: 57,890
     [1] traffic_sign: 6,234
     [2] traffic_light: 1,308
```

#### 3.6.2 详细验证

使用 `tools/validate_coco.py` 进行完整验证：

```bash
python tools/validate_coco.py \
  --ann-file data/traffic_coco/bdd100k_det/annotations/instances_val.json \
  --check-images
```

验证内容：
- ✓ JSON格式正确性
- ✓ 类别ID连续性（从0开始）
- ✓ 标注数据完整性
- ✓ 图像文件存在性（可选）
- ✓ 类别分布统计

#### 3.6.3 使用pycocotools验证

命令行直接测试：

```bash
python - <<'PY'
from pycocotools.coco import COCO
import os

ann = "data/traffic_coco/bdd100k_det/annotations/instances_val.json"
coco = COCO(ann)
cats = coco.loadCats(coco.getCatIds())

print("images:", len(coco.imgs), "annotations:", len(coco.anns), "categories:", len(cats))
print({c['id']: c['name'] for c in cats})
PY
```

#### 3.6.4 pycocotools eval 冒烟（仅用于链路验证）

使用 GT 生成伪检测结果，验证 COCOeval 链路是否正常（不要用于报告指标）：

```bash
python - <<'PY'
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ann = "data/traffic_coco/bdd100k_det/annotations/instances_val.json"
coco = COCO(ann)

img_ids = coco.getImgIds()[:100]  # 冒烟只取少量图片
ann_ids = coco.getAnnIds(imgIds=img_ids)

dets = []
for ann_obj in coco.loadAnns(ann_ids):
    dets.append({
        "image_id": ann_obj["image_id"],
        "category_id": ann_obj["category_id"],
        "bbox": ann_obj["bbox"],
        "score": 1.0,
    })

coco_dt = coco.loadRes(dets)
e = COCOeval(coco, coco_dt, iouType="bbox")
e.params.imgIds = img_ids
e.evaluate()
e.accumulate()
e.summarize()
PY
```

### 3.7 数据准备检查清单

在进入训练阶段前，确认以下事项：

- [ ] ✅ 目录结构创建完成
- [ ] ✅ BDD100K原始数据已下载并解压到 `data/raw/`
- [ ] ✅ 类别映射配置 `configs/classes.yaml` 已创建
- [ ] ✅ 转换脚本成功运行，生成COCO格式数据
- [ ] ✅ mapping.json 已生成，记录映射关系
- [ ] ✅ 冒烟测试通过，pycocotools可正常加载
- [ ] ✅ 类别数量、标注数量符合预期
- [ ] ✅ 图像文件完整（可选验证）

> **重要提示**：确保 `mapping.json` 保存完好，用于论文中的方法复现说明。

### 3.8 完整执行示例

以下是从零开始完成数据准备的完整命令序列：

```bash
# 步骤1: 激活Python环境
source .venv/bin/activate

# 步骤2: 安装必要依赖（如果还没安装）
pip install pycocotools pyyaml tqdm opencv-python

# 步骤3: 创建项目目录结构
cd /srv/code/detr_traffic_analysis
mkdir -p tools configs data/{raw,traffic_coco} outputs

# 步骤4: 准备BDD100K数据集
# （假设已下载到Downloads目录）
# ln -s ~/Downloads/bdd100k data/raw/bdd100k

# 步骤5: 执行COCO转换
python tools/convert_to_coco.py \
  --dataset bdd100k \
  --src data/raw/bdd100k \
  --dst data/traffic_coco/bdd100k_det \
  --config configs/classes.yaml \
  --splits train val

# 步骤6: 验证转换结果
python tools/smoke_test.py \
  data/traffic_coco/bdd100k_det/annotations/instances_val.json

# 步骤7: 详细验证（可选）
python tools/validate_coco.py \
  --ann-file data/traffic_coco/bdd100k_det/annotations/instances_train.json

# 步骤8: 查看映射信息
cat data/traffic_coco/bdd100k_det/mapping.json | python -m json.tool
```

### 3.9 常见问题与解决方案

#### 3.9.1 文件路径错误

**问题**：`FileNotFoundError: 标注文件不存在`

**解决**：
```bash
# 检查BDD100K目录结构
ls -la data/raw/bdd100k/labels/
ls -la data/raw/bdd100k/images/100k/

# 确保标注文件存在
# det_20_train.json, det_20_val.json, det_20_test.json
```

#### 3.9.2 类别映射冲突

**问题**：转换时出现"未映射类别"警告

**解决**：检查 `configs/classes.yaml` 中的映射表，确保覆盖所有BDD100K类别，或在 `BDD100K_MAPPING` 中添加缺失类别。

#### 3.9.3 内存不足

**问题**：处理大数据集时内存溢出

**解决**：
```python
# 修改 convert_to_coco.py，分批处理图像
# 或者先只转换小数据集进行测试
python tools/convert_to_coco.py \
  --src data/raw/bdd100k \
  --dst data/traffic_coco/bdd100k_det \
  --splits val  # 先只转换验证集
```

#### 3.9.4 图像复制耗时长

**问题**：转换脚本在复制图像时非常慢

**优化方案**：
1. 使用软链接代替复制：
```python
# 在 convert_to_coco.py 中修改
# shutil.copy2(src_img, dst_img) → os.symlink(src_img, dst_img)
```

2. 或者跳过图像复制，仅生成标注文件：
```python
# 注释掉转换脚本中的图像复制部分
# 在训练时直接从原始位置读取图像
```

### 3.10 性能优化建议

1. **转换加速**：
   - 使用多进程并行处理多个split
   - 对大数据集启用进度条显示（tqdm）

2. **存储优化**：
   - 训练时使用软链接避免重复存储图像
   - 仅保留必要的split（如train和val）

3. **验证优化**：
   - 冒烟测试仅检查前100张图像
   - 使用 `--check-images` 仅在怀疑数据问题时启用

---

## 4. 检测模型与小目标优化

本节对应论文中“基于 Deformable DETR 的检测模型设计与小目标优化”部分。

### 4.1 配置文件说明

核心配置文件示例：`configs/detr_small_obj.yaml`

建议包含以下关键信息：

* 数据集配置：

  * `dataset.root`: `data/traffic_coco`
  * `dataset.num_classes`: 类别数
* 训练超参数：

  * `train.batch_size`
  * `train.num_epochs`
  * `train.base_lr`
  * 学习率调度、warmup 等
* 模型结构：

  * 编解码器层数、注意力头数；
  * `num_feature_levels`（多尺度特征层数，如 4：P2–P5）；
* 小目标优化开关：

  * 是否启用多尺度训练；
  * 是否在损失函数中使用 SIoU 等增强定位能力的变体（若有实现）。

### 4.2 小目标检测专项优化

为支撑论文中“解决交通标志尺寸小问题”的论述，在配置与实现中进行了以下专项优化：

1. **多尺度特征融合**：

   * 在 `build_model.py` 中，构建 FPN 或等价结构，输出 P2–P5 多层特征；
   * 在 Transformer Encoder 中为不同尺度特征分配独立的位置编码与注意力权重；
   * 通过 `num_feature_levels` 及相关字段控制使用的特征层数。

2. **多尺度训练（Multi-scale Training）**：

   * 在数据加载/增强阶段，随机改变输入图像的短边尺寸；
   * 典型设置：在 `[640, 720, 800, 960]` 中随机采样；
   * 通过配置文件中的 `train.scales` 或类似键进行控制。

3. **数据增强策略**：

   * 随机尺度裁剪（Random Resize & Crop）；
   * 随机翻转、颜色扰动等基础增强；
   * 目标：提高模型对不同尺寸交通标志的鲁棒性。

> 若后续实现了 Copy-Paste 等更激进的小目标增强，可在此处补充说明并在配置中增加开关。

### 4.3 训练流程

#### 4.3.1 10 分钟级训练冒烟（先跑）

目标：用短时间确认 dataloader、loss、保存与评测链路全通。

建议任一方式控制规模：

* 使用小规模子集（如训练集抽取 100-500 张）；
* 临时将 `train.num_epochs` 设为 1-2；
* 降低输入分辨率或 batch size，减少单步耗时。

产出：`outputs/smoke_run/`（日志、权重、评测结果）。

脚本：`tools/train_detr.py`

示例命令：

```bash
python tools/train_detr.py \
  --config configs/detr_small_obj.yaml \
  --output-dir outputs/small_obj_run
```

建议在训练脚本中实现：

* 自动保存最佳 mAP/最佳 AP_small 的模型权重；
* 记录训练日志（loss 曲线、学习率变化等），保存在 `outputs/logs/`；
* 可选：记录验证集 mAP 曲线，用于论文绘图。

#### 4.3.2 日志格式与字段（统一）

日志输出允许 JSON 或 CSV（二选一或同时输出），建议字段包括：

* `epoch` / `iter`
* `loss_total` 与关键分项
* `lr`
* `mAP` / `AP_small`
* `time_per_iter`
* `gpu_mem_mb`（如可采集）

日志文件路径建议固定为 `outputs/<run>/metrics.json` 或 `outputs/<run>/metrics.csv`。

### 4.4 检测指标评测

脚本：`tools/eval_detr.py`

示例命令：

```bash
python tools/eval_detr.py \
  --config configs/detr_small_obj.yaml \
  --checkpoint outputs/small_obj_run/best.pth \
  --eval-set val
```

应输出：

* COCO Summary 表格（含 mAP@0.5:0.95、mAP@0.5、AP_small 等）；
* 每类 AP，便于分析特定标志/车辆类别的检测难度。

### 4.5 小目标消融顺序（控变量）

严格单变量推进，避免结论混淆：

1) **输入尺度/多尺度策略**（固定模型结构与训练超参）；
2) **`num_feature_levels`**（固定输入尺度与其他超参）；
3) **结构性改动**（backbone/heads/attention 等）。

每次改动后记录 AP/AP_small、训练耗时、显存与推理 FPS，形成对比表。

---

## 5. 多目标跟踪实现与评测

### 5.1 跟踪管线

脚本：`tools/inference_tracks.py`

基本流程：

1. 读取视频帧或图像序列；
2. 调用检测模型获取当前帧目标框与类别；
3. 调用跟踪器的 `update(detections)` 接口获得带 `track_id` 的结果；
4. 将结果以 MOT 格式写入文件，供 TrackEval 使用。

输出格式示例（逐帧 txt 或 csv）：

```text
frame, id, x, y, w, h, score, class_id
1, 7, 100, 120, 80, 60, 0.95, 0
1, 8, 300, 200, 40, 40, 0.87, 1
...
```

### 5.2 MOT 指标评测

脚本：`tools/eval_mot.py`

示例命令：

```bash
python tools/eval_mot.py \
  --gt data/mot_gt \
  --res outputs/results/tracks \
  --seqmap configs/mot_seqmap.txt
```

评测要求：

* 调用 TrackEval 库，至少启用：

  * HOTA 模块；
  * CLEAR 模块（MOTA 等）；
  * Identity 模块（IDF1 等）。
* 输出：

  * 控制台摘要；
  * 详细 CSV 报表，用于论文中绘制对比表格。

> 若对比不同跟踪算法（ByteTrack vs OC-SORT）或不同检测模型作为输入，可通过配置或命名区分输出目录。

### 5.3 阈值固化与日志记录

为保证可复现与论文可引用，需固定并输出两套阈值：

* **展示阈值**：`detector_score_thresh`（仅影响可视化展示）。
* **关联阈值**：`tracker_match_thresh` / `tracker_low_score`（影响轨迹关联）。

阈值写入实验日志或结果目录（如 `outputs/<run>/metrics.json`），并随评测结果一并保存。

---

## 6. Streamlit 界面与交互逻辑

本节对应论文中“可视化与交互功能设计”部分。

### 6.1 Session State 设计

在 `app/app_streamlit.py` 中建议维护以下状态：

* `current_frame`：当前帧索引；
* `bookmarks`：书签列表，元素格式如 `{ "frame_id": int, "note": str }`；
* `filters`：过滤条件（显示的类别、置信度阈值等）；
* `play_mode`：播放/暂停状态；
* `selected_tracker`：当前使用的跟踪算法（ByteTrack / OC-SORT）。

### 6.2 书签与回放逻辑

1. 用户在特定帧点击“添加书签”；
2. 系统读取当前 `frame_id`，弹出对话框收集备注信息；
3. 将 `{frame_id, note}` 写入 `st.session_state.bookmarks`；
4. 在侧边栏显示书签列表，点击某一项：

   * 调用 `video_reader.seek(frame_id)` 跳转；
   * 重新触发该帧的检测、跟踪与绘制；
   * 支持播放该帧前后若干秒场景进行局部回放。

### 6.3 结果导出

前端提供“导出结果”按钮：

* 导出 `result.csv`：采用在 README 中定义的 Schema；
* 可选：导出带框视频：

  * 将每帧可视化结果写回为新视频文件（使用 OpenCV VideoWriter）。

### 6.4 缓存回放优先策略

默认使用“缓存回放”（优先播放已计算结果），实时推理仅在必要时开启，并配合：

* 抽帧（如每 2-4 帧取 1 帧）；
* 降分辨率（小目标可接受范围内）；
* 限制刷新频率（降低 UI 卡顿）。

书签与导出功能需在缓存回放与实时模式各验证一次，确保闭环稳定。

---

## 7. 系统性能基准测试

脚本：`tools/benchmark_system.py`

### 7.1 设计目标

* 测量在不考虑前端渲染开销的情况下：

  * 单帧检测 + 跟踪耗时；
  * 整体 FPS；
  * 显存占用峰值。

### 7.2 使用示例

```bash
python tools/benchmark_system.py \
  --config configs/detr_small_obj.yaml \
  --checkpoint outputs/small_obj_run/best.pth \
  --input data/test_video.mp4 \
  --warmup 20 \
  --iters 200
```

### 7.3 输出指标

* `Inference Latency (ms/frame)`：平均模型推理时间；
* `Tracking Latency (ms/frame)`：平均跟踪关联时间；
* `System FPS`：整体吞吐率（1 / 平均总耗时）；
* `Peak VRAM (MB)`：通过 `torch.cuda.max_memory_allocated()` 采集的显存峰值。

> 建议将不同配置（baseline 与 small_obj）测得的结果整理为表格，用于论文中“实时性与资源占用分析”章节。

### 7.4 baseline vs small_obj 对比与推荐配置

完成两组对比后，输出推荐推理配置（分辨率、阈值、batch size 等），并标注：

* 目标场景（离线评测/实时推理/演示）；
* 速度-精度权衡结论；
* 最终推荐的默认参数（用于后续 Streamlit 与部署）。

---

## 8. 数据导出 Schema 细节

系统导出的 `result.csv` 与 JSON 结构需与论文中的“数据管线”描述保持一致，例如：

| 字段名        | 类型     | 说明                          |
| ---------- | ------ | --------------------------- |
| frame_id   | int    | 视频帧号（从 1 开始）                |
| timestamp  | float  | 相对时间戳（单位：秒）                 |
| track_id   | int    | 全局唯一跟踪 ID，-1 表示未参与跟踪        |
| class_id   | int    | 类别索引（0: vehicle, 1: sign 等） |
| class_name | str    | 类别名称                        |
| conf       | float  | 检测置信度                       |
| bbox_xywh  | string | "[cx, cy, w, h]" 格式的像素坐标字符串 |

JSON 结构可使用类似：

```json
{
  "frame_id": 120,
  "timestamp": 4.0,
  "objects": [
    {
      "track_id": 7,
      "class_id": 1,
      "class_name": "traffic_sign",
      "conf": 0.83,
      "bbox_xywh": [512.3, 240.5, 34.2, 36.8]
    }
  ]
}
```

---

## 9. 实验命名规范与复现流程

### 9.1 输出目录命名

* `baseline_run/`：未启用小目标专项优化的 Deformable DETR；
* `small_obj_run/`：启用多尺度特征与小目标增强后训练得到的模型；
* `tracker_byte_run/`：以 ByteTrack 为主跟踪算法的实验；
* `tracker_ocsort_run/`：以 OC-SORT 为主跟踪算法的实验。

### 9.2 典型复现步骤

1. 数据集：

   * 使用 `convert_to_coco.py` 完成 COCO 转换；
2. 训练检测模型：

   * 先运行 baseline；
   * 再运行 small_obj 配置，对比 AP_small 提升幅度（按 4.5 的单变量顺序推进）；
3. 生成跟踪结果：

   * 使用最佳检测模型 + ByteTrack / OC-SORT 分别生成结果；
4. 评测指标：

   * `eval_detr.py` → 检测指标（mAP / AP_small）；
   * `eval_mot.py` → 跟踪指标（HOTA / IDF1 / MOTA）；
5. 性能测试：

   * 使用 `benchmark_system.py` 采集 FPS / Latency / VRAM；
6. 整理结果：

   * 将关键结果保存至 `experiments/`，对应论文中的表格与图。

---

## 10. 调试建议与常见问题

* **检测效果不佳（尤其是小目标）：**

  * 检查输入分辨率与 multi-scale 配置是否生效；
  * 检查训练数据中交通标志样本是否充足，类别映射是否正确；
* **跟踪 ID 频繁切换：**

  * 调整跟踪器的匹配阈值（如 IoU 阈值、最小置信度等）；
  * 检查检测置信度阈值是否过高导致漏检；
* **Streamlit 页面卡顿：**

  * 降低实时刷新频率，避免每一帧都触发完整重绘；
  * 在长视频上可选择抽帧或仅对关键片段进行可视化；
* **显存不足：**

  * 降低输入分辨率或 batch size；
  * 关闭不必要的中间变量保存，使用 `torch.no_grad()` 包裹推理阶段。

以上内容可根据后续实际实现与调试过程进一步补充与修正。
