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


def resolve_eval_output_dir(config: dict, checkpoint_path: str, eval_set: str, override: str | None = None) -> Path:
    """解析评估输出目录，默认紧邻 checkpoint 保存评估资产。"""
    if override:
        override_path = Path(override)
        if override_path.suffix:
            return override_path.parent
        return override_path

    checkpoint_stem = Path(checkpoint_path).stem
    checkpoint_parent = Path(checkpoint_path).resolve().parent
    return checkpoint_parent / 'eval' / f'{eval_set}_{checkpoint_stem}'


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


def _iter_leaf_datasets(dataset):
    """递归展开 Subset / ConcatDataset，返回最底层数据集。"""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        yield from _iter_leaf_datasets(dataset.dataset)
    elif isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            yield from _iter_leaf_datasets(sub_dataset)
    else:
        yield dataset


def _iter_leaf_samples_at_index(dataset, idx):
    """按索引递归展开 Subset / ConcatDataset，返回最底层样本位置。"""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        yield from _iter_leaf_samples_at_index(dataset.dataset, int(dataset.indices[int(idx)]))
    elif isinstance(dataset, ConcatDataset):
        running = 0
        for sub_dataset in dataset.datasets:
            sub_len = len(sub_dataset)
            if idx < running + sub_len:
                yield from _iter_leaf_samples_at_index(sub_dataset, idx - running)
                return
            running += sub_len
        raise IndexError(f"样本索引越界: {idx}")
    else:
        yield dataset, int(idx)


def _iter_eval_samples(dataset):
    """按评估顺序遍历所有底层样本。"""
    from torch.utils.data import ConcatDataset, Subset

    if isinstance(dataset, Subset):
        for idx in dataset.indices:
            yield from _iter_leaf_samples_at_index(dataset.dataset, int(idx))
    elif isinstance(dataset, ConcatDataset):
        for sub_dataset in dataset.datasets:
            yield from _iter_eval_samples(sub_dataset)
    else:
        for idx in range(len(dataset)):
            yield dataset, idx


def _categories_signature(categories):
    ordered = sorted(categories, key=lambda cat: int(cat.get('id', -1)))
    return tuple(
        (
            int(cat.get('id', -1)),
            cat.get('name'),
            cat.get('supercategory'),
        )
        for cat in ordered
    )


def _build_eval_coco_gt(dataset, output_dir: Path, eval_set: str):
    """
    从实际评估数据集构建一个顺序重编号后的 COCO GT。
    这样可以兼容 Subset / ConcatDataset，并避免 image_id 冲突。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        'images': [],
        'annotations': [],
        'categories': [],
    }
    reference_signature = None
    next_image_id = 0
    next_ann_id = 0

    for leaf_dataset, local_idx in _iter_eval_samples(dataset):
        if not hasattr(leaf_dataset, 'coco') or not hasattr(leaf_dataset, 'ids'):
            raise ValueError("当前评估数据集不支持 COCO GT 构建，请检查数据集封装")

        coco = leaf_dataset.coco
        img_id = leaf_dataset.ids[int(local_idx)]
        img_info = dict(coco.loadImgs(img_id)[0])
        img_info['id'] = next_image_id
        merged['images'].append(img_info)

        categories = list(coco.dataset.get('categories', []))
        signature = _categories_signature(categories)
        if reference_signature is None:
            reference_signature = signature
            merged['categories'] = list(sorted(categories, key=lambda cat: int(cat.get('id', -1))))
            if 'info' in coco.dataset:
                merged['info'] = coco.dataset['info']
            if 'licenses' in coco.dataset:
                merged['licenses'] = coco.dataset['licenses']
        elif signature != reference_signature:
            raise ValueError("多数据集评估时检测到不一致的 categories 定义，请先统一类别映射")

        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann in coco.loadAnns(ann_ids):
            new_ann = dict(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = next_image_id
            merged['annotations'].append(new_ann)
            next_ann_id += 1

        next_image_id += 1

    merged_path = output_dir / f'{eval_set}_merged_coco.json'
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    return COCO(str(merged_path)), merged_path


def build_eval_coco_gt(dataset, output_dir: Path, eval_set: str):
    """公开评估 GT 构建接口，供训练阶段验证复用。"""
    return _build_eval_coco_gt(dataset, output_dir, eval_set)


def _resolve_continuous_category_map(dataset):
    """在 Subset / ConcatDataset 中稳健查找连续类别映射。"""
    mappings = []
    for leaf_dataset in _iter_leaf_datasets(dataset):
        mapping = getattr(leaf_dataset, 'continuous_to_cat_id', None)
        if mapping is not None:
            mappings.append(mapping)

    if not mappings:
        return None

    reference = mappings[0]
    for mapping in mappings[1:]:
        if mapping != reference:
            raise ValueError("多数据集评估时检测到不一致的 continuous_to_cat_id 映射")
    return reference


def _resolve_eval_image_id(target, coco_gt, sequential_image_id):
    """优先使用 target 中的 image_id，若不命中当前 coco_gt，则回退到顺序编号。"""
    target_image_id = target.get('image_id') if isinstance(target, dict) else None
    if torch.is_tensor(target_image_id):
        target_image_id = int(target_image_id.item())
    elif isinstance(target_image_id, (list, tuple)) and target_image_id:
        target_image_id = int(target_image_id[0])

    if target_image_id is not None and target_image_id in coco_gt.imgs:
        return target_image_id
    return sequential_image_id


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
    sample_index = 0
    
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
        
        # 转换为 COCO 格式，使用评估顺序重编号后的 image_id 对齐 merged GT
        for output, target in zip(processed_outputs, original_targets):
            image_id = _resolve_eval_image_id(target, coco_gt, sample_index)
            sample_index += 1
            
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
    sample_index = 0
    
    logger.info("评估 Deformable DETR (官方数据流)...")
    category_id_map = _resolve_continuous_category_map(dataloader.dataset)
    
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
            image_id = _resolve_eval_image_id(target, coco_gt, sample_index)
            sample_index += 1
            
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
                continuous_id = label.item()
                if category_id_map:
                    category_id = category_id_map[continuous_id]
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

    # 追加按类别 AP 指标，便于重点跟踪 traffic_sign AP
    # precision: [TxRxKxAxM], area index: 0=all, 1=small, 2=medium, 3=large
    precision = coco_eval.eval.get('precision')
    params = coco_eval.params
    if precision is not None:
        cat_ids = params.catIds
        area_idx_all = params.areaRngLbl.index('all') if 'all' in params.areaRngLbl else 0
        area_idx_small = params.areaRngLbl.index('small') if 'small' in params.areaRngLbl else 1
        max_det_idx = len(params.maxDets) - 1

        for cat_pos, cat_id in enumerate(cat_ids):
            cat_name = coco_gt.cats.get(cat_id, {}).get('name', str(cat_id))
            metric_key = f"AP_{cat_name}"
            metric_key_small = f"AP_small_{cat_name}"

            p_all = precision[:, :, cat_pos, area_idx_all, max_det_idx]
            p_small = precision[:, :, cat_pos, area_idx_small, max_det_idx]

            valid_all = p_all[p_all > -1]
            valid_small = p_small[p_small > -1]

            metrics[metric_key] = float(valid_all.mean()) if valid_all.size > 0 else float('nan')
            metrics[metric_key_small] = float(valid_small.mean()) if valid_small.size > 0 else float('nan')
    
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


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="DETR/Deformable DETR 统一评估脚本")
        parser.add_argument("--config", type=str, required=True, help="配置文件路径")
        parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
        parser.add_argument("--eval-set", type=str, default="val", choices=["train", "val"], help="评估集")
        parser.add_argument("--output", type=str, help="结果输出路径（可选，默认会自动保存到 outputs/ 下）")
        parser.add_argument("--score-threshold", type=float, default=0.05, help="置信度阈值")
        args = parser.parse_args()
    
    # 加载配置
    print(f"📖 加载配置: {args.config}")
    config = load_config(args.config)
    
    model_type = config.get('model', {}).get('type', 'detr')
    print(f"🔧 模型类型: {model_type}")
    
    output_dir = resolve_eval_output_dir(config, args.checkpoint, args.eval_set, args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    preferred_device = config.get('device', {}).get('type', 'cuda')
    device = torch.device('cuda' if preferred_device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"🔧 设备: {device}")
    
    # 日志
    logger = setup_logger(f"eval_{args.eval_set}_{Path(args.checkpoint).stem}", output_dir / 'eval.log')
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"checkpoint: {args.checkpoint}")
    logger.info(f"评估集: {args.eval_set}")
    
    # 构建数据加载器
    print("\n📦 构建数据加载器...")
    dataloader, dataset = build_eval_dataloader(config, args.eval_set)

    # 基于实际评估数据集构建 COCO GT，避免多数据集 / Subset / ConcatDataset 的 image_id 冲突
    coco_gt, merged_gt_path = _build_eval_coco_gt(dataset, output_dir, args.eval_set)
    logger.info(f"COCO GT: {merged_gt_path}")
    
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
    
    # 保存结果：默认落盘，用户不需要额外参数
    if args.output:
        output_path = Path(args.output)
        if not output_path.suffix:
            output_path = output_path / 'metrics.json'
    else:
        output_path = output_dir / 'metrics.json'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存: {output_path}")


if __name__ == '__main__':
    main()
