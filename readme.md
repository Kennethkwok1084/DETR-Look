# 基于 Deformable DETR 的交通标志与车辆多目标检测系统

2026 届本科毕业论文工程源码
题目：基于 Deformable DETR 的交通标志与车辆多目标检测系统的设计与实现
关键词：Deformable DETR，小目标检测，多目标跟踪，Streamlit 可视化

---

## 文档入口

- 项目主文档保留在根目录：`readme.md`
- 其余说明文档已整理到 [`docs/README.md`](docs/README.md)

---

## 📋 项目当前状态（2026-01-03更新）

### ✅ 已完成（第1-2步：数据准备阶段）

#### 1️⃣ 项目架构搭建
- ✅ 完整目录结构：`tools/`, `configs/`, `data/`, `outputs/`
- ✅ 配置管理系统：YAML格式配置文件
- ✅ 文档体系：`docs/develop.md` 开发指南

#### 2️⃣ 类别映射与配置
- ✅ **configs/classes.yaml**：定义3类粗粒度映射
  - `0: vehicle` (car, bus, truck)
  - `1: traffic_sign` (交通标志)
  - `2: traffic_light` (红绿灯)
- ✅ 可配置选项：bike/motor并入、最小面积过滤

#### 3️⃣ 数据集转换工具链
- ✅ **tools/convert_to_coco.py**：完整转换脚本
  - 支持数据集：BDD100K, CCTSDB, TT100K
  - 类别映射：原始类别 → 粗粒度类别 → class_id
  - 统计输出：总图片数/标注数/类别计数
  - 映射记录：生成 `mapping.json` 用于论文复现
- ✅ **BDD100K**：支持 det_20 聚合 JSON 或单图 JSON
- ✅ **CCTSDB**：支持 VOC XML 标注格式
- ✅ **TT100K**：已验证转换（6,034 训练图像，16,749 标注）
- ✅ **tools/validate_coco.py**：数据集完整性验证
- ✅ **tools/smoke_test.py**：快速冒烟测试

#### 4️⃣ 训练配置模板
- ✅ **configs/detr_baseline.yaml**：基础训练配置
  - 数据集路径、类别数
  - 模型结构参数（编解码器层数、注意力头等）
  - 训练超参数（学习率、batch size、优化器等）
  - 冒烟测试配置（200 iter快速验证）
- ✅ **tools/train_detr.py**：训练脚本框架

### 🚧 进行中

- ⏸️ 数据集下载与转换（等待BDD100K数据）
- ⏸️ Deformable DETR模型实现
- ⏸️ 训练循环实现

### 📅 待开发

- ⬜ 数据加载器（Dataset/DataLoader）
- ⬜ Deformable DETR模型核心组件（backbone/transformer/heads）
- ⬜ 训练与评估流程
- ⬜ 跟踪器封装（ByteTrack/OC-SORT）
- ⬜ Streamlit可视化界面
- ⬜ 完整评测流程

---

### 🔁 Deformable DETR 迁移补充说明（新增）

本段为新增补充说明，原有 DETR 相关描述不做删除或替换。接下来将以 Deformable DETR 作为核心路线，迁移步骤按“模型初始化 → 预处理/后处理 → 评估对齐”的顺序推进。为保证现有调用链稳定，脚本与配置文件命名仍保留 `train_detr.py` 与 `configs/detr_*.yaml`，迁移稳定后再统一整理与清理。

## 🚀 快速开始

### 环境准备

```bash
# 1. 进入项目目录
cd /srv/code/detr_traffic_analysis

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 安装依赖
pip install pycocotools pyyaml tqdm numpy opencv-python
```

### 数据准备（第1-2步）

```bash
# 查看类别映射配置
cat configs/classes.yaml

# ============================================================
# 1️⃣ BDD100K 数据集转换
# ============================================================
# 执行COCO转换（需要先准备BDD100K数据）
python tools/convert_to_coco.py \
  --dataset bdd100k \
  --src data/raw/bdd100k \
  --dst data/traffic_coco/bdd100k_det \
  --config configs/classes.yaml \
  --splits train val

# 验证转换结果
python tools/smoke_test.py \
  data/traffic_coco/bdd100k_det/annotations/instances_val.json

# 详细验证（可选）
python tools/validate_coco.py \
  --ann-file data/traffic_coco/bdd100k_det/annotations/instances_val.json

# ============================================================
# 2️⃣ CCTSDB 数据集转换（中国交通标志小目标）
# ============================================================
# 数据集结构：
#   data/raw/cctsdb/
#     images/train/
#     images/test/
#     labels/xml/
python tools/convert_to_coco.py \
  --dataset cctsdb \
  --src data/raw/cctsdb \
  --dst data/traffic_coco/cctsdb_det \
  --config configs/classes.yaml \
  --splits train test

# 验证转换结果
python tools/smoke_test.py \
  data/traffic_coco/cctsdb_det/annotations/instances_train.json

# ============================================================
# 3️⃣ TT100K 数据集转换（大规模交通标志）
# ============================================================
# 数据集结构：
#   data/tt100k/
#     annotations_all.json
#     train/
#     test/
python tools/convert_to_coco.py \
  --dataset tt100k \
  --src data/tt100k \
  --dst data/traffic_coco/tt100k_det \
  --config configs/classes.yaml \
  --splits train test

# 验证转换结果
python tools/smoke_test.py \
  data/traffic_coco/tt100k_det/annotations/instances_train.json
```

### 预期输出

转换成功后会生成：

```
data/traffic_coco/
├── bdd100k_det/
│   ├── images/
│   │   ├── train/        # 训练集图像
│   │   └── val/          # 验证集图像
│   ├── annotations/
│   │   ├── instances_train.json  # COCO格式标注
│   │   └── instances_val.json
│   └── mapping.json      # 类别映射记录
│
├── cctsdb_det/
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   ├── annotations/
│   │   ├── instances_train.json
│   │   └── instances_test.json
│   └── mapping.json
│
└── tt100k_det/
    ├── images/
    │   ├── train/
    │   └── test/
    ├── annotations/
    │   ├── instances_train.json
    │   └── instances_test.json
    └── mapping.json
```

冒烟测试输出示例：

**BDD100K**:
```
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

**TT100K** (已验证):
```
✅ 加载成功!
   图像数: 6,034
   标注数: 16,749
   类别数: 3
   类别映射: {0: 'vehicle', 1: 'traffic_sign', 2: 'traffic_light'}
   
   类别分布:
     [0] vehicle: 0
     [1] traffic_sign: 16,749
     [2] traffic_light: 0
```

---

## 1. 项目概览

### 1.1 项目简介

本项目围绕自动驾驶和高级驾驶辅助系统（ADAS）中的交通场景感知任务，设计并实现了一套基于 **Deformable DETR** 的交通标志与车辆多目标检测系统。系统以车载第一视角视频为主要输入，实现：

* 交通标志与车辆的多类别目标检测；
* 对检测结果的跨帧多目标跟踪与轨迹生成；
* 基于 Streamlit 的交互式可视化与 Failure Case 分析。

该系统既是毕业论文的工程实现载体，也是后续算法调试与实验复现的平台。

### 1.2 研究背景与痛点

* **小目标问题**：远距离交通标志在图像中仅占少量像素，易被背景淹没；
* **遮挡与拥堵**：城市道路中车辆密集、遮挡频繁，易出现漏检与 ID 频繁切换；
* **可解释性需求**：在自动驾驶研发流程中，研发人员需要直观地观察检测与跟踪行为，分析失败样本。

本项目通过多尺度可变形注意力、数据增强、在线多目标跟踪与可视化分析等手段，对上述问题进行系统性探索与工程实现。

### 1.3 项目目标

* **检测层面**：实现对车辆与交通标志的高精度检测，尤其关注小目标 AP 提升；
* **跟踪层面**：在视频序列中保持目标 ID 的时序一致性，减轻轨迹断裂与 ID 切换；
* **系统层面**：构建可交互的可视化分析工具，支持书签、回放、数据导出与性能评测；
* **论文支撑**：为毕业论文第 4 章（系统实现）与第 5 章（实验验证）提供完整的工程基础与实验闭环。

---

## 2. 系统功能

### 2.1 检测与小目标优化

* 支持对以下类别进行检测（可根据数据集配置）：

  * 车辆类：Car、Truck、Bus 等；
  * 交通标志类：Traffic Sign（可进一步细分或合并类别）。
* 基于 Deformable DETR 实现端到端目标检测：

  * 编解码器结构承担全局建模与注意力聚合；
  * 使用多尺度特征（如 P2–P5）增强小目标感知能力。
* 针对小目标的专项优化：

  * 启用多尺度训练（multi-scale training）；
  * 在配置中显式增加高分辨率特征层参与注意力计算；
  * 配合数据增强（随机缩放与裁剪）提升模型对不同尺度的鲁棒性。

### 2.2 多目标跟踪

* 集成在线多目标跟踪算法（如 ByteTrack / OC-SORT 封装）：

  * 接收每帧检测结果（bbox + score + class）；
  * 输出带有 track_id 的稳定轨迹；
* 支持在漏检、遮挡场景下维持 ID 稳定性；
* 输出格式兼容 TrackEval 工具，可用于计算 HOTA、IDF1、MOTA 等指标。

### 2.3 交互式可视化（Streamlit）

* 基于 Streamlit 实现 Web 端可视化界面：

  * 上传/选择测试视频；
  * 设置检测阈值、跟踪算法、是否显示轨迹等；
  * 实时叠加显示边界框、类别、置信度与历史轨迹线。
* 书签与回放功能：

  * 在关键帧添加书签（frame_id + 备注信息）；
  * 通过侧边栏书签列表快速跳转至对应帧并回放上下文片段。
* 结果导出：

  * 导出包含检测 + 跟踪结果的 CSV / JSON；
  * 导出叠加可视化结果的视频文件（带框回放）。

### 2.4 性能评估

项目支持如下三类指标的评测和导出：

* **检测指标**：

  * mAP@0.5:0.95（综合精度）；
  * AP_small（小目标精度）；
* **跟踪指标**：

  * HOTA、IDF1、MOTA；
* **系统指标**：

  * FPS（吞吐量）、端到端 Latency（单帧时延）、峰值 VRAM（显存占用）。

---

## 3. 系统架构与目录结构

本系统采用「前端–后端合一」架构：

* 后端：PyTorch 实现的 Deformable DETR 推理引擎与多目标跟踪模块；
* 前端：Streamlit 实现的 Web 界面与交互逻辑；
* 工具层：数据集转换、离线评测与性能基准测试脚本。

### 3.1 模块划分

* `app/`：Streamlit 前端应用与交互逻辑（Session State 管理、书签列表等）；
* `models/`：Deformable DETR 模型构建与推理封装；
* `tracker/`：多目标跟踪器封装（ByteTrack / OC-SORT）；
* `viz/`：可视化绘制模块（基于 OpenCV，对图片绘制 bbox/轨迹等）；
* `video_io/`：视频解码与逐帧读取、帧号跳转；
* `tools/`：数据转换、训练、评测、推理与性能测试脚本；
* `configs/`：模型与训练配置文件（含小目标优化版本）；
* `data/`：数据集存放目录（原始 + COCO 转换后）；
* `outputs/`：训练日志、模型权重、评测结果、推理导出等；
* `experiments/`（可选）：实验记录、配置快照与论文图表。

### 3.2 目录结构示例

```text
├── app/
│   └── app_streamlit.py       # Streamlit 主入口
├── models/
│   ├── detr_backbone.py
│   ├── detr_heads.py
│   ├── build_model.py
│   └── __init__.py
├── tracker/
│   ├── base_tracker.py
│   ├── bytetrack_wrapper.py
│   ├── ocsort_wrapper.py
│   └── utils.py
├── viz/
│   └── drawer.py
├── video_io/
│   └── video_reader.py
├── tools/
│   ├── convert_to_coco.py
│   ├── train_detr.py
│   ├── eval_detr.py
│   ├── inference_tracks.py
│   ├── eval_mot.py
│   └── benchmark_system.py
├── configs/
│   ├── detr_baseline.yaml
│   └── detr_small_obj.yaml
├── data/
│   └── traffic_coco/          # COCO 格式数据集
├── outputs/
│   ├── logs/
│   ├── weights/
│   └── results/
├── experiments/               # 实验与论文图表（可选）
├── README.md
└── develop.md
```

---

## 4. 数据集策略与目录规划

本项目以 BDD100K 为主战场，CCTSDB / TT100K 作为小目标强化与跨数据集泛化评估，LISA Traffic Light 视时间资源决定是否加入。所有数据统一转换为 COCO JSON。

### 4.1 数据集分工

- **BDD100K**：主检测与小目标实验（baseline + small_obj），多目标跟踪指标（HOTA/MOTA/IDF1），Streamlit demo 视频来源。
- **CCTSDB**：中国交通标志小目标专项，基于 BDD100K 预训练权重微调或独立训练，关注 AP_small；可做跨数据集迁移故事。
- **TT100K**：大规模小目标路牌，验证小目标优化的泛化；可直接测试或短暂微调。
- **LISA Traffic Light（可选）**：红绿灯细粒度/小目标加分实验。

### 4.2 目录与 COCO 转换约定

```text
data/
  raw/
    bdd100k/
    cctsdb/
    tt100k/
    lisa_traffic_light/         # 可选
  traffic_coco/
    bdd100k_det/
      images/{train,val,test}
      annotations/{instances_train.json, instances_val.json, instances_test.json}
    cctsdb_det/
      images/{train,val}
      annotations/{instances_train.json, instances_val.json}
    tt100k_det/
      images/{train,val}
      annotations/{instances_train.json, instances_val.json}
    lisa_light_det/             # 可选
```

建议在 `tools/convert_to_coco.py` 下实现独立转换函数（`convert_bdd100k_to_coco` / `convert_cctsdb_to_coco` / `convert_tt100k_to_coco`），统一输出到上述目录。

### 4.3 类别映射方案（示例）

在 `configs/classes.yaml` 维护全局 coarse 类别，训练/评测时通过 DataLoader 映射：

```yaml
COARSE_CLASSES:
  0: vehicle        # car, bus, truck 等
  1: traffic_sign   # 各种路牌/标志
  2: traffic_light  # 红绿灯
```

- **BDD100K**：`car/bus/truck` → vehicle；`traffic sign` → traffic_sign；`traffic light` → traffic_light（可选 bike/motor 是否并入 vehicle）。
- **CCTSDB**：三大类（指示/禁止/警告）全部映射为 traffic_sign；需要细粒度时保留原类。
- **TT100K**：200+ 路牌全部映射为 traffic_sign；可选长尾分析时保留原类。
- **LISA TL（可选）**：14 类信号状态全部映射为 traffic_light；细粒度实验保留原类。

### 4.4 训练与实验顺序（执行建议）

1) **BDD100K 主干**：先跑 baseline，再跑 small_obj，多尺度与高分辨率特征开关，关注 AP_small；在 BDD100K MOT 上评测 HOTA/MOTA/IDF1。
2) **小目标专项**：基于 BDD100K 预训练权重，对 CCTSDB / TT100K 微调或直接测试，报告 AP_small 提升；时间充足可独立训练一版作对照。
3) **可选扩展**：LISA traffic light 细粒度或小目标实验。
4) **可视化与答辩**：Streamlit 使用 BDD100K 或自采视频，支持书签/回放/导出；引用上述数据集产出的指标与样例。

---

## 5. 快速开始

### 4.1 环境准备

推荐使用 Conda：

```bash
conda create -n detr_traffic python=3.10
conda activate detr_traffic

pip install -r requirements.txt
```

确保 `requirements.txt` 中包含：

* `torch`, `torchvision`
* `streamlit`
* `opencv-python`
* `pycocotools`
* `trackeval`
* 以及 `numpy`, `tqdm`, `pyyaml` 等基础依赖。

### 4.2 数据集准备

1. 将原始交通场景数据集（如 BDD100K / TT100K）放入 `data/raw/`；
2. 使用脚本转换为 COCO 格式：

```bash
python tools/convert_to_coco.py \
  --src data/raw \
  --dst data/traffic_coco
```

3. 转换完成后，确认 `data/traffic_coco/` 下存在：

* `images/train`, `images/val`；
* `annotations/instances_train.json`, `annotations/instances_val.json`。

### 4.3 模型权重

* 方式一：使用官方 Deformable DETR 预训练权重作为初始化；
* 方式二：在本项目数据集上从头训练或微调，权重保存在 `outputs/weights/`。

配置文件中通过字段（如 `MODEL.WEIGHTS` 或自定义 `weights_path`）指定权重路径。

### 4.4 运行可视化系统

```bash
streamlit run app/app_streamlit.py
```

启动后在浏览器访问：

* 默认：`http://localhost:8501`

即可：

* 上传视频文件进行检测与跟踪展示；
* 调整置信度阈值、跟踪算法类型、可视化选项；
* 为误检/漏检帧添加书签并回放分析。

### 4.5 离线推理与评测

* 检测评测（mAP / AP_small）：

```bash
python tools/eval_detr.py \
  --config configs/detr_small_obj.yaml \
  --eval-set val
```

* 生成跟踪结果并评测（HOTA / IDF1 / MOTA）：

```bash
# 生成跟踪结果
python tools/inference_tracks.py \
  --config configs/detr_small_obj.yaml \
  --output outputs/results/tracks/

# 计算 MOT 指标
python tools/eval_mot.py \
  --gt data/mot_gt \
  --res outputs/results/tracks
```

* 系统性能基准测试（FPS / Latency / VRAM）：

```bash
python tools/benchmark_system.py \
  --config configs/detr_small_obj.yaml \
  --input data/test_video.mp4
```

---

## 6. 数据导出规范（Schema）

系统导出的 `result.csv` 默认采用以下字段结构：

| 字段名        | 类型     | 说明                          |
| ---------- | ------ | --------------------------- |
| frame_id   | int    | 视频帧号（从 1 开始）                |
| timestamp  | float  | 相对时间戳（单位：秒）                 |
| track_id   | int    | 全局唯一跟踪 ID，-1 表示未参与跟踪        |
| class_id   | int    | 类别索引（0: vehicle, 1: sign 等） |
| class_name | str    | 类别名称                        |
| conf       | float  | 检测置信度（0.0–1.0）              |
| bbox_xywh  | string | "[cx, cy, w, h]" 格式像素坐标     |

示例：

```csv
frame_id,timestamp,track_id,class_id,class_name,conf,bbox_xywh
120,4.00,7,1,traffic_sign,0.83,"[512.3, 240.5, 34.2, 36.8]"
```

该 Schema 与论文中“数据管线与落地字段”章节保持一致，可直接用于实验数据分析与图表绘制。

---

## 7. 实验命名规范

为方便论文插图与对比实验，建议对输出结果采用统一命名约定：

* `baseline_run/`：原始 Deformable DETR，无小目标专项优化；
* `small_obj_run/`：启用多尺度特征与小目标增强配置；
* `tracker_byte_run/`：以 ByteTrack 作为主跟踪器；
* `tracker_ocsort_run/`：以 OC-SORT 作为主跟踪器。

典型目录结构：

```text
outputs/
  baseline_run/
  small_obj_run/
  tracker_byte_run/
  tracker_ocsort_run/
```

在论文撰写时，可直接引用上述命名对应的实验结果。

---

## 8. 与论文章节的对应关系

* **第 3 章 系统设计**：

  * 对应 README 中的项目概览、系统功能与架构设计部分；
* **第 4 章 系统实现**：

  * 对应 `develop.md` 中的模块实现与代码结构说明；
* **第 5 章 实验验证**：

  * 对应 `tools/` 中的评测脚本与 `experiments/` 中保存的实验记录与图表。

通过本仓库可以完整复现论文中的检测性能、小目标指标、跟踪指标以及系统性能评估结果。
