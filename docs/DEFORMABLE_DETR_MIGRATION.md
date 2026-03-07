# Deformable DETR 迁移开发文档（高复用分支方案）

## 目标与范围

- 在当前代码基础上新增 Deformable DETR 分支能力，保持最大复用。
- 训练/评估/推理/可视化/导出全部迁移，并保留 DETR 旧路径可继续使用（向后兼容）。
- 数据集使用 COCO 格式，评估指标以 COCO mAP 为主（含 AP50/75、小中大目标等）。
- 配置与脚本入口尽量不变，仅新增少量字段即可切换模型类型。

## 现状概览（当前代码）

- 训练主链路：`tools/train_detr.py` + `dataset/coco_dataset.py` + `models/detr_model.py`
- 评估链路：`tools/eval_detr.py`（COCOeval）
- 优化训练链路：`tools/train_detr_optimized.py`
- 处理器依赖：`DetrImageProcessor`（HuggingFace）
- 模型依赖：`DetrForObjectDetection`（HuggingFace）

注意：当前文档中已有“Deformable”描述，但代码实际仍为 DETR，需要统一。

## 解释：什么是“高复用分支”

高复用分支的核心是“不复制整套训练/评估脚本”，而是在现有入口中新增“模型类型开关”，将差异集中在模型构建与后处理层。这样能做到：

- 训练/评估脚本逻辑基本不动；
- 数据加载与 COCO 评估完全复用；
- 差异集中在模型类 + 图像处理器 + 少量配置字段。

## 迁移设计（最小改动点）

### 1) 模型封装层

- 新增 `models/deformable_detr_model.py`，接口与 `models/detr_model.py` 一致：
  - `DeformableDetrForObjectDetection`
  - `DeformableDetrConfig`
- 在 `models/__init__.py` 中新增统一入口，例如 `build_model(config)`：
  - 根据 `config['model']['type']` 选择 DETR/Deformable DETR

关键注意  
当前 `models/detr_model.py` 会自动补 `facebook/` 前缀，若 `config['model']['name'] = "SenseTime/deformable-detr"` 会被误改为 `facebook/SenseTime/...`。  
迁移时需要把模型名称标准化逻辑改为：

- 若已包含 `/`，保持不变
- 若不包含 `/`，默认加 `facebook/`

### 2) ImageProcessor 统一入口

在训练/评估脚本中抽出统一方法：

- DETR 使用 `DetrImageProcessor`
- Deformable DETR 使用 `DeformableDetrImageProcessor`
- 由 `config['model']['type']` 选择对应处理器

这样 `dataset/coco_dataset.py` 的 `build_dataloader(..., image_processor=...)` 可以直接复用。

### 3) 训练脚本改造

首选方案：保持 `tools/train_detr.py` 入口不变，仅增加模型类型开关。  
主要修改点：

- 由 `build_detr_model` 切换为 `build_model`
- 由 `DetrImageProcessor` 切换为 `build_image_processor`
- 日志输出显示模型类型与模型 ID

可选方案：新增 `tools/train_deformable_detr.py` 作为快捷入口，内部调用通用训练函数。

### 4) 评估脚本改造

保持 `tools/eval_detr.py` 入口不变，关键改动：

- 使用通用 `build_image_processor`（与训练一致）
- 预测后处理使用对应 processor 的 `post_process_object_detection`
- 确保 COCO category_id 与训练时一致（必要时保留映射表）

### 5) 优化训练脚本改造

`tools/train_detr_optimized.py` 目前基于 DETR。  
建议将其改为支持 `--model-type` 或拆分出 `train_deformable_detr_optimized.py`，核心变化：

- 模型构建：`DeformableDetrForObjectDetection`
- 评估后处理：`DeformableDetrImageProcessor.post_process_object_detection`
- 预处理与归一化保持一致（ImageNet mean/std）

### 6) 推理/可视化/导出

若已有推理或可视化脚本，迁移原则与评估一致：

- 统一使用 `build_model` + `build_image_processor`
- 统一采用 processor 的 `post_process_object_detection`

如需导出（TorchScript/ONNX），需确认 Deformable DETR 的动态 shape 兼容性并添加导出说明。

## 配置改动（保持现有习惯）

新增字段建议：

- `model.type`: `detr` 或 `deformable_detr`
- `model.name`: 允许 HuggingFace 完整 ID（含 `/`）
- 其它字段保持不变（如 `training`, `dataset`）

示例（新增配置文件 `configs/deformable_detr_baseline.yaml`）：

```yaml
model:
  type: "deformable_detr"
  name: "SenseTime/deformable-detr"
  pretrained: true
  num_queries: 300
  loss_weights:
    class_loss_coef: 1.0
    bbox_loss_coef: 5.0
    giou_loss_coef: 2.0
    eos_coef: 0.1
```

## COCO 数据与 mAP 指标解析

### COCO 数据

- 标注格式：`bbox` 为 `[x, y, w, h]`（像素坐标）
- `category_id` 不一定连续，需要映射到 `[0..N-1]` 训练
- 评估时必须反向映射回原始 `category_id`，否则 COCOeval 结果会错位

### mAP 指标

- **mAP (AP@[.5:.95])**：IoU 从 0.5 到 0.95，步长 0.05 的平均 AP
- **AP50 / AP75**：IoU=0.5 和 IoU=0.75 的 AP
- **AP_small / AP_medium / AP_large**：按 COCO 目标面积分段统计

这些指标需要 `COCOeval` 的标准流程，不能手工计算替代。

## 整体项目修改方案（基于现有结构的最小侵入）

### 1) 目录与依赖引入（vendoring）

- 新增目录：`third_party/deformable_detr/`
- 拷贝官方 Deformable DETR repo 的核心源码：
  - `models/`, `util/`, `ops/`（至少包含 `MultiScaleDeformableAttention`）
- 新增 `third_party/deformable_detr/UPSTREAM.md` 记录上游 commit 与来源
- 新增编译说明：CUDA 扩展编译步骤与版本矩阵

### 2) 模型封装与统一入口

- 新增 `models/deformable_detr_model.py`
  - 对接官方 `build_model(...)`，获得 `model/criterion/postprocessors`
  - 封装为与现有训练脚本兼容的接口（输出 `loss` 或 `loss_dict`）
- 改造 `models/__init__.py`
  - 新增 `build_model(config)` 作为统一入口
  - 根据 `config['model']['type']` 路由到 DETR/Deformable DETR

### 3) 数据流与标注适配

- 保持 COCO 输入与现有 `dataset/` 逻辑
- 增加 Deformable DETR 专用的 target 生成：
  - `labels`: 连续映射 `[0..N-1]`
  - `boxes`: 归一化 `cxcywh`（官方训练格式）
  - `orig_size`/`size` 用于后处理
- 复用或抽取 transforms：
  - 保持当前 resize/pad/normalize 的行为一致
  - 若采用官方 `datasets/transforms.py`，做最小适配

### 4) 训练脚本改造（保持入口）

- `tools/train_detr.py`
  - 由 `build_detr_model` 切到 `build_model`
  - 根据 `model.type` 选择数据预处理路径
  - 保持 `yaml` 配置结构不变，仅新增必要字段

### 5) 评估脚本改造

- `tools/eval_detr.py`
  - 使用 `postprocessors['bbox']` 生成 bbox（官方方式）
  - 统一 COCO category_id 映射（训练/评估一致）
  - 输出仍使用 COCOeval（mAP/AP50/AP75/AP_small 等）

### 6) 优化训练脚本（可选）

- `tools/train_detr_optimized.py`
  - 增加 `--model-type` 与 Deformable DETR 路由
  - 保持高性能数据读取，但输出需符合官方模型格式

### 7) 配置与实验对比

- 新增配置：
  - `configs/deformable_detr_baseline.yaml`（单阶段）
  - `configs/deformable_detr_two_stage.yaml`（两阶段对比）
- 关键参数：
  - `model.type = deformable_detr`
  - `model.name`（可留作说明，不强依赖 HF）
  - `two_stage`、`num_queries`、`num_feature_levels`

### 8) 预检与构建说明（不改代码也要写清楚）

- 预检项：
  - `nvcc` 是否可用
  - PyTorch CUDA 版本与编译器版本是否匹配
- 失败回退：
  - 没有 CUDA 时无法编译扩展，需换环境或容器

### 9) 验收与回归

- 冒烟训练：小子集 1 个 epoch，loss 下降
- 评估输出：COCOeval 可正常输出 mAP
- 回归验证：旧 DETR 配置仍可训练/评估

## 实施步骤（建议里程碑）

### 引入方式（已选：vendoring）

说明：将官方 Deformable DETR 源码拷入仓库内，优点是自包含、部署简单；缺点是后续升级需要手动合并。

落地步骤：

1) 新建目录：`third_party/deformable_detr`（或 `models/third_party/deformable_detr`）
2) 从官方 repo 拷贝 `models/`、`util/`、`ops/` 相关源码
3) 保留上游版本信息（在目录内新增 `UPSTREAM.md` 记录 commit/版本）
4) 在项目构建流程中加入 `ops/` 的编译指引（CUDA 扩展）

### 迁移里程碑

1) 新建分支：`feature/deformable-detr`（仅做增量改动）  
2) 引入官方源码（vendoring）：`third_party/deformable_detr/`  
3) 新增模型封装与统一入口：`models/deformable_detr_model.py` + `models/__init__.py`  
4) 训练脚本改造：`tools/train_detr.py` 支持 `model.type`  
5) 评估脚本改造：`tools/eval_detr.py` 支持 `model.type`  
6) 优化脚本改造：`tools/train_detr_optimized.py` 或新增 Deformable 版本  
7) 新增配置样例与文档更新

## 第 4-5 章补充：跨帧 warm-start + ByteTrack + 最小化数据产出

目标：在不训练自有模型的前提下，补齐跨帧信息利用与在线关联的实现闭环，并给出最小可交付的定量/定性结果。

### 4.x 跨帧 warm-start 的最小实现

核心思路：推理时用上一帧的高置信检测结果，替换 Deformable DETR 的初始 reference points 前 K 个。

最小缓存字段：

- `ref_points_prev`: 上一帧 top-K 的 `(cx, cy)`，归一化到 [0, 1]
- `scores_prev`: 置信度
- 可选 `wh_prev`: `(w, h)`，如需更强引导

选择规则：

- score 排序 + `score > tau` 过滤
- 最多取 K（建议 K=30 或 50，tau=0.5）

替换逻辑：

- 在模型 forward 里获取 `init_reference`（形状 `[B, num_queries, 2/4]`）
- 若有缓存，则替换前 K 个：
  - `init_reference[:, :K, :2] = ref_points_prev`
- 数值安全：`clamp` 到 `[eps, 1-eps]`
- 若无有效缓存，则退化为 baseline

推理主循环结构（用于流程图/方法描述）：

```text
for frame in video:
  warm_ref = prev_ref_points if valid else None
  outputs = model(frame, warm_ref_points=warm_ref)
  dets = post_process(outputs)
  prev_ref_points = topk_ref_points(dets)
  tracks = bytetrack.update(dets)
```

### 4.y ByteTrack 在线关联接入

输入输出约定（论文描述用）：

- 输入 detections：`(x1, y1, x2, y2, score, cls)`
- 输出 tracks：`(x1, y1, x2, y2, track_id, cls, score)`

工程建议：

- 先按 `cls` 分组做 tracking，减少跨类误匹配导致的 IDSW
- 每类独立 tracker，最后合并输出

### 5.x 最小、最快速、最稳定的数据产出路径（无自训）

原则：不训练，只做推理 + 跟踪 + 轻量评测，优先使用已有公开的 tracking 标注集。

推荐路径（最小成本）：

1) 模型：直接用预训练 Deformable DETR 权重（例如 `SenseTime/deformable-detr`）
2) 数据：BDD100K tracking 的 val 子集（带 GT track id）
3) 子集规模：3-5 段短序列（每段 200-300 帧），控制时长与稳定性
4) 评测：用 ByteTrack 输出轨迹，计算 IDSW/Frag（可选 IDF1/HOTA）
5) 对比：同一子集跑 baseline vs warm-start

最小交付物清单（第 5 章闭环）：

- 表 1：消融表（Baseline vs Warm-start）
  - IDSW, Frag, FPS（可选 IDF1/HOTA）
- 图 1：系统流程图（检测 + warm-start + ByteTrack + 输出）
- 图 2：定性对比拼图（2 组场景，baseline vs ours）

如果无法引入 tracking 子集，可用 2-3 段人工标注的短视频做定性+弱量化，但该方案仅作备选。

## 验收与验证

- 冒烟测试：小子集训练 1 个 epoch，loss 正常下降
- 评估测试：mAP 可输出且合理
- 兼容性测试：旧 DETR 配置仍可训练与评估

## 风险与回退

- 预训练权重下载失败：需提前缓存或提供离线说明
- `model.name` 前缀处理错误：会导致模型加载失败
- CPU 环境训练非常慢：建议增加清晰的 GPU 要求说明
- 评估 category_id 映射不一致：导致 mAP 假高/假低

## 交付物清单（预期新增或修改）

- `models/deformable_detr_model.py`
- `models/__init__.py`
- `tools/train_detr.py`
- `tools/eval_detr.py`
- `tools/train_detr_optimized.py`（或新增 Deformable 版本）
- `configs/deformable_detr_baseline.yaml`
- `docs/DEFORMABLE_DETR_MIGRATION.md`
