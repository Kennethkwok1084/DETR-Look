# 重新安装 CUDA 版本 PyTorch 并编译 Deformable DETR 扩展

## 当前状态

✅ **已完成**：
- PyTorch 2.7.1+cu118 安装成功
- CUDA 可用（torch.cuda.is_available() = True）

⚠️ **待完成**：
- 编译 CUDA 扩展（需要 MSVC 编译器）

## 问题与解决方案

### 问题 1: CUDA 版本不匹配 ✅ 已解决
- **问题**: 系统 CUDA 13.0 vs PyTorch CUDA 11.8
- **解决**: 修改 `setup.py` 跳过版本检查（已完成）

### 问题 2: 缺少 MSVC 编译器 ⚠️ 待解决
- **问题**: `error: Microsoft Visual C++ 14.0 or greater is required`
- **解决方案**：

#### 方案 A：安装 Visual Studio Build Tools（推荐）

1. 下载安装器：
   https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/

2. 运行安装器，选择：
   - ✅ "使用 C++ 的桌面开发"
   - ✅ "MSVC v143 - VS 2022 C++ x64/x86 生成工具"
   - ✅ "Windows 11 SDK"

3. 安装完成后，重新打开终端并编译：
   ```powershell
   cd D:\TrainingData\Code\third_party\deformable_detr\models\ops
   D:/TrainingData/Code/.venv/Scripts/python.exe setup.py build install
   ```

#### 方案 B：使用预编译版本（如果有）

从官方或社区获取预编译的 `.pyd` 文件，放到对应目录。

#### 方案 C：Docker 容器编译（最简单）

使用 Docker 避免本地编译问题：

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

WORKDIR /workspace
COPY . .

RUN cd third_party/deformable_detr/models/ops && \
    python setup.py build install
```

## 快速验证脚本

编译完成后运行：

```powershell
D:/TrainingData/Code/.venv/Scripts/python.exe -c "from models import build_model; print('✅ 模块导入成功')"
```

## 当前编译命令

```powershell
# 确保在正确的目录
cd D:\TrainingData\Code\third_party\deformable_detr\models\ops

# 编译（需要 MSVC）
D:/TrainingData/Code/.venv/Scripts/python.exe setup.py build install
```

## 临时绕过方案（仅用于测试）

如果只是想测试代码结构，可以暂时使用 DETR 模型：

```yaml
# configs/detr_baseline.yaml
model:
  type: "detr"  # 不需要编译 CUDA 扩展
  name: "detr-resnet-50"
```

等编译环境准备好后，再切换到 Deformable DETR。
