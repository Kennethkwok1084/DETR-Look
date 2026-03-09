# 基于 Deformable DETR 的交通标志与车辆多目标检测系统

2026 届本科毕业论文工程源码  
**题目**：基于 Deformable DETR 的交通标志与车辆多目标检测系统的设计与实现  
**关键词**：Deformable DETR，小目标检测，多目标跟踪，Streamlit 可视化

---

## 📖 项目介绍

本项目针对自动驾驶场景中的交通标志与车辆检测任务，特别是远距离小目标检测难题，设计并实现了一套完整的检测与跟踪系统。系统基于 **Deformable DETR** 架构，集成了多尺度特征优化、数据增强策略以及 **ByteTrack** 多目标跟踪算法，并提供了基于 **Streamlit** 的交互式可视化界面，用于模型评估与 Failure Case 分析。

### 核心特性

- **多类别目标检测**：支持车辆（Car, Bus, Truck）、交通标志（Traffic Sign）和红绿灯（Traffic Light）的高精度检测。
- **小目标专项优化**：利用 Deformable DETR 的多尺度注意力机制与数据增强策略，提升小目标（如远距离路牌）的检测性能。
- **多目标跟踪**：集成 ByteTrack 算法，实现稳定的跨帧目标跟踪与轨迹生成（开发中）。
- **交互式可视化**：提供 Web 端界面，支持图片/视频推理、GT 可视化、预测对比与批量导出。
- **完整工具链**：包含从 BDD100K/CCTSDB/TT100K 数据集转换、训练、评测到部署的全套工具。

---

## 🚀 快速开始 (Quick Start)

### 1. 获取代码

部署的第一步是获取最新的代码更新：

```bash
# 获取最新代码
git pull https://github.com/Kennethkwok1084/DETR-Look
```

### 2. 环境准备

推荐使用 Python 3.10+ 环境：

```bash
# 进入项目目录
cd /srv/code/detr_traffic_analysis

# 创建并激活虚拟环境 (可选)
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
# 如果没有 requirements.txt，可以安装基础依赖：
pip install torch torchvision pycocotools pyyaml tqdm opencv-python streamlit
```

### 3. 数据准备

本项目统一使用 COCO 格式。提供了针对 BDD100K、CCTSDB 和 TT100K 的转换工具。

```bash
# 查看类别映射配置
cat configs/classes.yaml

# 示例：转换 BDD100K 数据集
python tools/convert_to_coco.py \
  --dataset bdd100k \
  --src data/raw/bdd100k \
  --dst data/traffic_coco/bdd100k_det \
  --splits train val

# 验证数据转换
python tools/smoke_test.py data/traffic_coco/bdd100k_det/annotations/instances_val.json
```

更多数据准备细节请参考 [开发文档](docs/develop.md)。

### 4. 运行可视化演示

启动 Streamlit 界面进行模型推理与数据浏览：

```bash
streamlit run tools/app.py
```

启动后访问终端显示的 URL（通常是 `http://localhost:8501`）。

---

## 📂 项目结构

```text
detr_traffic_analysis/
├── configs/            # 配置文件 (模型参数, 类别映射)
├── data/               # 数据集存放目录
├── docs/               # 文档 (开发指南, 实验记录)
├── models/             # 模型定义 (Deformable DETR)
├── outputs/            # 输出目录 (日志, 权重, 结果)
├── tools/              # 工具脚本 (训练, 评测, 可视化)
│   ├── app.py          # Streamlit 可视化应用
│   ├── convert_to_coco.py # 数据集转换工具
│   ├── train_detr.py   # 训练脚本
│   └── ...
├── utils/              # 通用工具函数
└── readme.md           # 项目主文档
```

---

## 📋 项目状态

| 模块 | 状态 | 说明 |
| :--- | :---: | :--- |
| **项目架构** | ✅ 完成 | 目录结构、配置系统、文档体系 |
| **数据工具链** | ✅ 完成 | 支持 BDD100K/CCTSDB/TT100K 转 COCO |
| **检测模型** | ✅ 完成 | Deformable DETR 模型实现与训练脚本 |
| **可视化系统** | ✅ 完成 | Streamlit 演示系统 (推理/GT对比/批量导出) |
| **多目标跟踪** | �� 进行中 | ByteTrack 封装已就绪，待集成测试 |
| **评测流程** | 🚧 进行中 | 基础评测可用，完整指标体系完善中 |

---

## 📚 文档索引

- **[开发与实现说明 (develop.md)](docs/develop.md)**：详细的开发指南、模块说明与实验复现步骤。
- **[配置文件说明](configs/classes.yaml)**：类别映射与数据配置详情。

---

## 🔗 引用

本项目基于以下开源项目进行二次开发：
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
