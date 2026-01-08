@echo off
echo ============================================================
echo 重新安装 CUDA 版本 PyTorch
echo ============================================================
echo.

echo [1/4] 卸载 CPU 版本 PyTorch...
D:\TrainingData\Code\.venv\Scripts\pip.exe uninstall torch torchvision -y
echo.

echo [2/4] 安装 CUDA 12.4 版本 PyTorch...
D:\TrainingData\Code\.venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo.

echo [3/4] 验证安装...
D:\TrainingData\Code\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo.

echo [4/4] 编译 CUDA 扩展...
cd /d D:\TrainingData\Code\third_party\deformable_detr\models\ops
D:\TrainingData\Code\.venv\Scripts\python.exe setup.py build install
echo.

echo ============================================================
echo 完成！
echo ============================================================
pause
