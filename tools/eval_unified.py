#!/usr/bin/env python3
"""
统一评估脚本
支持 DETR (HF) 和 Deformable DETR (官方) 双数据流
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_model, build_image_processor
from utils import load_checkpoint, setup_logger


def load_config(config_path: str) -> dict:
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_eval_dataloader(config, eval_set='val'):
    """根据模型类型构建评估数据加载器"""
    model_type = config.get('model', {}).get('type', 'detr').lower()
    
    if model_type == 'deformable_detr' or model_type == 'deformable-detr':
        # Deformable DETR: 官方数据流
        from dataset.deformable_dataset import build_deformable_dataloader
        return build_deformable_dataloader(config, eval_set)
    else:
        # DETR: HF 数据流
        from dataset import build_dataloader
        image_processor = build_image_processor(config)
        dataloader = build_dataloader(
            config=config,
            image_set=eval_set,
            image_processor=image_processor
        )
        # 返回 (dataloader, dataset) 元组
        return dataloader, dataloader.dataset


@torch.no_grad()
def evaluate_detr(model, dataloader, device, coco_gt, logger, config, score_threshold=0.05):
    """
    评估 DETR 模型（HF 数据流）
    
    Args:
        model: DETR 模型
        dataloader: HF 格式数据加载器
        device: 设备
        coco_gt: COCO ground truth
        logger: 日志器
        config: 配置字典（用于构建 image_processor）
        score_threshold: 置信度阈值
    
    Returns:
        metrics dict
    """
    from transformers import DetrImageProcessor
    
    model.eval()
    results = []
    
    logger.info("评估 DETR (HF 数据流)...")
    
    # 从 config 构建 image processor
    image_processor = build_image_processor(config)
    if image_processor is None:
        raise ValueError("DETR 评估需要 image_processor")
    
    for batch in tqdm(dataloader, desc="Evaluating DETR"):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device) if 'pixel_mask' in batch else None
        
        # 保留原始 targets（包含 image_id）
        # 注意：batch['labels'] 是 HF processor 处理后的，可能不含 image_id
        # 优先使用原始 batch['targets']，fallback 到 batch['labels']
        original_targets = batch.get('targets', batch.get('labels', []))
        
        # 原图 PIL images（用于获取尺寸）
        images = batch.get('images', None)
        if images is None:
            # 如果没有原图，从 targets 获取尺寸
            target_sizes = torch.tensor([[t.get('orig_size', t.get('size', [800, 800]))[0], 
                                         t.get('orig_size', t.get('size', [800, 800]))[1]] 
                                        for t in original_targets]).to(device)
        else:
            target_sizes = torch.tensor([img.size[::-1] for img in images]).to(device)
        
        # 推理
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        
        # 后处理
        processed_outputs = image_processor.post_process_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=target_sizes
        )
        
        # 转换为 COCO 格式（使用原始 targets 确保有 image_id）
        for output, target in zip(processed_outputs, original_targets):
            # 确保 image_id 存在
            if 'image_id' not in target:
                raise ValueError(f"Target 缺少 image_id，请检查数据加载器是否保留原始 targets")
            image_id = target['image_id'].item() if torch.is_tensor(target['image_id']) else target['image_id']
            
            scores = output['scores']
            labels = output['labels']
            boxes = output['boxes']  # xyxy 格式
            
            for score, label, box in zip(scores, labels, boxes):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'image_id': image_id,
                    'category_id': label.item(),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # xywh
                    'score': score.item(),
                })
    
    return compute_coco_metrics(results, coco_gt, logger)


@torch.no_grad()
def evaluate_deformable(model, dataloader, device, coco_gt, logger, score_threshold=0.05):
    """
    评估 Deformable DETR 模型（官方数据流）
    
    Args:
        model: Deformable DETR 模型封装
        dataloader: 官方格式数据加载器（返回 NestedTensor, targets）
        device: 设备
        coco_gt: COCO ground truth
        logger: 日志器
        score_threshold: 置信度阈值
    
    Returns:
        metrics dict
    """
    import sys
    from pathlib import Path
    
    # 添加 third_party 路径
    _third_party_path = Path(__file__).parent.parent / "third_party" / "deformable_detr"
    if str(_third_party_path) not in sys.path:
        sys.path.insert(0, str(_third_party_path))
    
    from util.misc import NestedTensor
    
    model.eval()
    results = []
    
    logger.info("评估 Deformable DETR (官方数据流)...")
    
    for samples, targets in tqdm(dataloader, desc="Evaluating Deformable"):
        # 移到设备
        if isinstance(samples, NestedTensor):
            samples = NestedTensor(samples.tensors.to(device), samples.mask.to(device))
        else:
            samples = samples.to(device)
        
        # 推理（不需要 targets）
        outputs = model(samples, targets=None)
        
        # 获取原图尺寸（从 targets）
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)
        
        # 后处理（使用官方 PostProcess）
        processed_outputs = model.postprocess(outputs, orig_target_sizes)
        
        # 转换为 COCO 格式
        for output, target in zip(processed_outputs, targets):
            image_id = target['image_id'].item() if torch.is_tensor(target['image_id']) else target['image_id'][0]
            
            scores = output['scores']
            labels = output['labels']
            boxes = output['boxes']  # xyxy 格式（已还原到原图）
            
            # 过滤低置信度
            keep = scores > score_threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            
            for score, label, box in zip(scores, labels, boxes):
                x1, y1, x2, y2 = box.tolist()
                
                # 关键：将连续索引反映射回原始 COCO category_id
                # Deformable 数据集将 category_id 映射为 [0, num_classes-1]
                # 需要反映射回原始 ID 以匹配 COCO 标注
                continuous_id = label.item()
                if hasattr(dataloader.dataset, 'dataset'):  # Subset 包装
                    dataset = dataloader.dataset.dataset
                else:
                    dataset = dataloader.dataset
                
                if hasattr(dataset, 'continuous_to_cat_id'):
                    category_id = dataset.continuous_to_cat_id[continuous_id]
                else:
                    # DETR 或无映射的数据集，直接使用
                    category_id = continuous_id
                
                results.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],  # xywh
                    'score': score.item(),
                })
    
    return compute_coco_metrics(results, coco_gt, logger)


def compute_coco_metrics(results, coco_gt, logger):
    """计算 COCO 指标"""
    logger.info(f"生成了 {len(results)} 个检测结果")
    
    if len(results) == 0:
        logger.warning("没有检测结果！")
        return {}
    
    # 使用 COCO API 评估
    logger.info("开始 COCO 评估...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取指标
    metrics = {
        'mAP': coco_eval.stats[0],
        'mAP_50': coco_eval.stats[1],
        'mAP_75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
    }
    
    return metrics


def evaluate(model, dataloader, device, coco_gt, logger, config, score_threshold=0.05):
    """
    统一评估入口（自动选择数据流）
    """
    model_type = config.get('model', {}).get('type', 'detr').lower()
    
    if model_type == 'deformable_detr' or model_type == 'deformable-detr':
        return evaluate_deformable(model, dataloader, device, coco_gt, logger, score_threshold)
    else:
        return evaluate_detr(model, dataloader, device, coco_gt, logger, config, score_threshold)


def main():
    parser = argparse.ArgumentParser(description="DETR/Deformable DETR 统一评估脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--eval-set", type=str, default="val", choices=["train", "val"], help="评估集")
    parser.add_argument("--output", type=str, help="结果输出路径")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="置信度阈值")
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"📖 加载配置: {args.config}")
    config = load_config(args.config)
    
    model_type = config.get('model', {}).get('type', 'detr')
    print(f"🔧 模型类型: {model_type}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    
    # 日志
    logger = setup_logger('eval')
    
    # 构建数据加载器
    print("\n📦 构建数据加载器...")
    dataloader, dataset = build_eval_dataloader(config, args.eval_set)
    
    # 加载 COCO ground truth
    if args.eval_set == 'train':
        ann_file = config['dataset']['train_ann']
    else:
        ann_file = config['dataset']['val_ann']
    
    # 如果是相对路径，拼接 root_dir
    from pathlib import Path
    ann_file = Path(ann_file)
    if not ann_file.is_absolute():
        root_dir = config['dataset'].get('root_dir', '')
        if root_dir:
            ann_file = Path(root_dir) / ann_file
    
    coco_gt = COCO(str(ann_file))
    
    # 构建模型
    print("\n🏗️  构建模型...")
    model = build_model(config)
    model = model.to(device)
    
    # 加载 checkpoint
    print(f"\n📂 加载 checkpoint: {args.checkpoint}")
    load_checkpoint(args.checkpoint, model, device=str(device))
    
    # 评估
    print("\n🎯 开始评估")
    print(f"置信度阈值: {args.score_threshold}")
    print("=" * 60)
    
    metrics = evaluate(model, dataloader, device, coco_gt, logger, config, args.score_threshold)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 60)
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 结果已保存: {output_path}")


if __name__ == '__main__':
    main()
