#!/usr/bin/env python3
"""
DETR模型评估脚本
使用pycocotools计算COCO格式的检测指标
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from transformers import DetrImageProcessor
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset import build_dataloader
from models import build_detr_model
from utils import load_checkpoint, setup_logger


def load_config(config_path: str) -> dict:
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, dataloader, device, coco_gt, logger, score_threshold=0.05, image_processor=None):
    """
    评估模型
    
    Args:
        model: DETR模型
        dataloader: 数据加载器
        device: 设备
        coco_gt: COCO ground truth对象
        logger: 日志器
        score_threshold: 置信度阈值
        image_processor: DETR图像处理器（可选）
    
    Returns:
        评估结果字典
    """
    model.eval()
    
    results = []
    
    logger.info("开始评估...")
    
    # 初始化图像处理器（如果未提供）
    if image_processor is None:
        image_processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    
    for images, targets in tqdm(dataloader, desc="Evaluating"):
        # 使用DetrImageProcessor处理可变尺寸图像
        images_pil = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
        encoding = image_processor(images=images_pil, return_tensors='pt')
        
        pixel_values = encoding['pixel_values'].to(device)
        pixel_mask = encoding['pixel_mask'].to(device)
        
        # 推理
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        
        # 解析输出
        logits = outputs.logits  # [B, num_queries, num_classes+1]
        boxes = outputs.pred_boxes  # [B, num_queries, 4] (cxcywh格式，归一化)
        
        # 转换为COCO格式
        for i, target in enumerate(targets):
            image_id = target['image_id'].item()
            img_h, img_w = target['size'].tolist()
            
            # 获取预测
            scores = logits[i].softmax(-1)  # [num_queries, num_classes+1]
            max_scores, labels = scores[:, :-1].max(-1)  # 排除background类
            
            # 转换boxes格式: cxcywh -> xyxy -> xywh (COCO格式)
            pred_boxes = boxes[i]
            
            # cxcywh -> xyxy (像素坐标)
            boxes_xyxy = torch.zeros_like(pred_boxes)
            boxes_xyxy[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * img_w  # x1
            boxes_xyxy[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * img_h  # y1
            boxes_xyxy[:, 2] = (pred_boxes[:, 0] + pred_boxes[:, 2] / 2) * img_w  # x2
            boxes_xyxy[:, 3] = (pred_boxes[:, 1] + pred_boxes[:, 3] / 2) * img_h  # y2
            
            # xyxy -> xywh (COCO格式)
            boxes_xywh = torch.zeros_like(pred_boxes)
            boxes_xywh[:, 0] = boxes_xyxy[:, 0]  # x
            boxes_xywh[:, 1] = boxes_xyxy[:, 1]  # y
            boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # w
            boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # h
            
            # 过滤低置信度预测（使用传入的score_threshold参数）
            keep = max_scores > score_threshold
            
            for score, label, box in zip(max_scores[keep], labels[keep], boxes_xywh[keep]):
                results.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': box.cpu().tolist(),
                    'score': score.item(),
                })
    
    logger.info(f"生成了 {len(results)} 个检测结果")
    
    if len(results) == 0:
        logger.warning("没有检测结果！")
        return {}
    
    # 使用COCO API评估
    logger.info("开始COCO评估...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取关键指标
    metrics = {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="评估DETR模型")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型checkpoint路径",
    )
    parser.add_argument(
        "--eval-set",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="评估数据集",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="结果输出路径",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="检测置信度阈值（默认0.05）",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"📖 加载配置: {args.config}")
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device(config['device']['type'] if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    
    # 设置日志
    logger = setup_logger('eval')
    
    # 构建数据加载器
    print("\n📦 构建数据加载器")
    dataloader = build_dataloader(config, args.eval_set)
    
    # 加载COCO ground truth
    root = Path(config['dataset']['root_dir'])
    ann_file = root / config['dataset'][f'{args.eval_set}_ann']
    coco_gt = COCO(str(ann_file))
    
    # 构建模型
    print("\n🏗️  构建模型")
    model = build_detr_model(config)
    model = model.to(device)
    
    # 加载checkpoint
    print(f"\n📂 加载checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=str(device))
    
    # 评估
    print("\n🎯 开始评估")
    print(f"置信度阈值: {args.score_threshold}")
    print("="*60)
    metrics = evaluate(model, dataloader, device, coco_gt, logger, args.score_threshold)
    
    # 打印结果
    print("\n" + "="*60)
    print("📊 评估结果")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("="*60)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 结果已保存: {output_path}")


if __name__ == '__main__':
    main()
