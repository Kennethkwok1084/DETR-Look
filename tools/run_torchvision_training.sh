#!/bin/bash
# Torchvision DETR 快速启动脚本

set -e

echo "================================================================================"
echo "🚀 Torchvision DETR 训练快速启动"
echo "================================================================================"

# 检查虚拟环境
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  未检测到虚拟环境，尝试激活..."
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
        echo "✅ 虚拟环境已激活: $VIRTUAL_ENV"
    else
        echo "❌ 未找到虚拟环境，请先运行: source .venv/bin/activate"
        exit 1
    fi
fi

# 检查数据集
TRAIN_IMG="data/traffic_coco/bdd100k_det/images/train"
TRAIN_ANN="data/traffic_coco/bdd100k_det/annotations/instances_train.json"
VAL_IMG="data/traffic_coco/bdd100k_det/images/val"
VAL_ANN="data/traffic_coco/bdd100k_det/annotations/instances_val.json"

if [[ ! -f "$TRAIN_ANN" ]]; then
    echo "❌ 训练标注文件不存在: $TRAIN_ANN"
    echo "请先运行数据转换："
    echo "  python tools/convert_to_coco.py --dataset bdd100k --src data/raw/bdd100k --dst data/traffic_coco/bdd100k_det"
    exit 1
fi

echo "✅ 数据集检查通过"
echo ""

# 选择模式
echo "请选择运行模式："
echo "  1) 冒烟测试（100张图，1 epoch）"
echo "  2) 快速训练（子集2000张，10 epochs）"
echo "  3) 完整训练（全部数据，50 epochs）"
echo ""
read -p "输入选项 [1-3]: " MODE

case $MODE in
    1)
        echo ""
        echo "================================================================================"
        echo "🔥 模式 1: 冒烟测试"
        echo "================================================================================"
        python tools/train_detr_optimized.py \
            --train-img "$TRAIN_IMG" \
            --train-ann "$TRAIN_ANN" \
            --batch-size 4 \
            --num-workers 4 \
            --subset 100 \
            --num-epochs 1 \
            --output-dir outputs/smoke_test
        ;;
    
    2)
        echo ""
        echo "================================================================================"
        echo "⚡ 模式 2: 快速训练（子集验证）"
        echo "================================================================================"
        python tools/train_detr_optimized.py \
            --train-img "$TRAIN_IMG" \
            --train-ann "$TRAIN_ANN" \
            --val-img "$VAL_IMG" \
            --val-ann "$VAL_ANN" \
            --num-classes 3 \
            --batch-size 16 \
            --num-workers 12 \
            --prefetch-factor 2 \
            --min-size 800 \
            --max-size 1333 \
            --subset 2000 \
            --num-epochs 10 \
            --eval-interval 2 \
            --amp \
            --pretrained \
            --output-dir outputs/detr_fast
        ;;
    
    3)
        echo ""
        echo "================================================================================"
        echo "🎯 模式 3: 完整训练"
        echo "================================================================================"
        python tools/train_detr_optimized.py \
            --train-img "$TRAIN_IMG" \
            --train-ann "$TRAIN_ANN" \
            --val-img "$VAL_IMG" \
            --val-ann "$VAL_ANN" \
            --num-classes 3 \
            --batch-size 16 \
            --num-workers 12 \
            --prefetch-factor 2 \
            --min-size 800 \
            --max-size 1333 \
            --num-epochs 50 \
            --eval-interval 5 \
            --amp \
            --pretrained \
            --output-dir outputs/detr_full
        ;;
    
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "✅ 完成！"
echo "================================================================================"
