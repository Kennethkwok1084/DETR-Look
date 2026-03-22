#!/usr/bin/env python3
"""
统一训练脚本
支持 DETR (HF) 和 Deformable DETR (官方) 双数据流
根据配置文件中的 model.type 自动选择
"""

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from pycocotools.coco import COCO
from torch.amp import GradScaler

try:
    import numpy as np
except ImportError:  # pragma: no cover - 训练环境通常会安装 numpy
    np = None

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_model, build_image_processor
from tools.eval_unified import evaluate as evaluate_model
from utils import MetricsLogger, load_checkpoint, save_checkpoint, setup_logger, train_one_epoch


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_output_dir(config: dict, override: str | None) -> Path:
    """解析输出目录。"""
    if override:
        return Path(override)
    if config['training'].get('output_dir'):
        return Path(config['training']['output_dir'])
    return Path(config.get('output', {}).get('base_dir', 'outputs')) / config.get('output', {}).get('experiment_name', 'experiment')


def setup_seed(config: dict, logger):
    """设置随机种子，保证训练可复现。"""
    seed = config.get('seed')
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"随机种子: {seed}")


def build_dataloader_for_model(config, image_set='train'):
    """
    根据模型类型构建对应的数据加载器

    Returns:
        (dataloader, dataset) 元组
    """
    model_type = config.get('model', {}).get('type', 'detr').lower()

    if model_type in ('deformable_detr', 'deformable-detr'):
        from dataset.deformable_dataset import build_deformable_dataloader
        return build_deformable_dataloader(config, image_set)

    from dataset import build_dataloader

    image_processor = build_image_processor(config)
    dataloader = build_dataloader(
        config=config,
        image_set=image_set,
        image_processor=image_processor,
    )
    return dataloader, dataloader.dataset


def build_optimizer(model, config, logger):
    """构建优化器，Deformable DETR 支持 backbone / linear proj 分组学习率。"""
    train_config = config['training']
    opt_config = train_config['optimizer']
    opt_type = opt_config['type']
    base_lr = float(opt_config['lr'])
    weight_decay = float(opt_config.get('weight_decay', 0.0001))
    betas = tuple(opt_config.get('betas', [0.9, 0.999]))
    model_type = config.get('model', {}).get('type', 'detr').lower()

    param_groups = None
    if model_type in ('deformable_detr', 'deformable-detr'):
        lr_backbone = float(opt_config.get('lr_backbone', train_config.get('lr_backbone', base_lr * 0.1)))
        lr_linear_proj_mult = float(opt_config.get('lr_linear_proj_mult', 0.1))
        linear_proj_names = tuple(opt_config.get('linear_proj_names', ['sampling_offsets', 'reference_points']))

        backbone_params = []
        linear_proj_params = []
        main_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in linear_proj_names):
                linear_proj_params.append(param)
            elif '.backbone.' in name or name.startswith('model.backbone'):
                backbone_params.append(param)
            else:
                main_params.append(param)

        param_groups = []
        if main_params:
            param_groups.append({'params': main_params, 'lr': base_lr})
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr_backbone})
        if linear_proj_params:
            param_groups.append({'params': linear_proj_params, 'lr': base_lr * lr_linear_proj_mult})

        logger.info(
            "Deformable 优化器分组: "
            f"main={len(main_params)} tensors @ {base_lr:.2e}, "
            f"backbone={len(backbone_params)} tensors @ {lr_backbone:.2e}, "
            f"linear_proj={len(linear_proj_params)} tensors @ {base_lr * lr_linear_proj_mult:.2e}"
        )

    if param_groups is None:
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr}]

    if opt_type == 'AdamW':
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay, betas=betas)
    if opt_type == 'Adam':
        return torch.optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
    raise ValueError(f"不支持的优化器: {opt_type}")


def build_scheduler(optimizer, config):
    """构建学习率调度器，支持 warmup。"""
    train_config = config['training']
    num_epochs = train_config.get('num_epochs') or train_config.get('max_epochs', 50)
    sch_config = train_config.get('lr_scheduler')
    if not sch_config:
        return None

    sch_type = sch_config['type']
    if sch_type == 'StepLR':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_config['step_size']),
            gamma=float(sch_config.get('gamma', 0.1)),
        )
    elif sch_type == 'MultiStepLR':
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sch_config['milestones'],
            gamma=float(sch_config.get('gamma', 0.1)),
        )
    elif sch_type == 'CosineAnnealingLR':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sch_config.get('t_max', num_epochs)),
            eta_min=float(sch_config.get('eta_min', 0.0)),
        )
    else:
        raise ValueError(f"不支持的调度器: {sch_type}")

    warmup_epochs = int(sch_config.get('warmup_epochs', train_config.get('warmup_epochs', 0)) or 0)
    warmup_start_factor = float(sch_config.get('warmup_start_factor', 0.1))
    if warmup_epochs <= 0:
        return main_scheduler

    if warmup_epochs >= num_epochs:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=num_epochs,
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs],
    )


def resolve_annotation_path(config, image_set='val') -> Path:
    """解析 COCO 标注路径。"""
    ann_key = 'train_ann' if image_set == 'train' else 'val_ann'
    ann_file = Path(config['dataset'][ann_key])
    if ann_file.is_absolute():
        return ann_file
    root_dir = config['dataset'].get('root_dir')
    if root_dir:
        return Path(root_dir) / ann_file
    return ann_file


def resolve_dataset_sources(config: dict, image_set: str) -> list[dict]:
    """解析单/多数据集配置为统一的 split 源列表。"""
    dataset_config = config['dataset']
    ann_key = f'{image_set}_ann'
    img_key = f'{image_set}_img'
    sources = []

    if dataset_config.get('datasets'):
        for idx, ds_conf in enumerate(dataset_config['datasets']):
            if ann_key not in ds_conf:
                continue

            root_dir = Path(ds_conf['root_dir'])
            ann_file = Path(ds_conf[ann_key])
            if not ann_file.is_absolute():
                ann_file = root_dir / ann_file

            img_folder = ds_conf.get(img_key)
            if img_folder:
                img_folder = Path(img_folder)
                if not img_folder.is_absolute():
                    img_folder = root_dir / img_folder
            else:
                img_folder = root_dir / 'images' / image_set

            sources.append({
                'name': ds_conf.get('name', f'{image_set}_{idx}'),
                'ann_file': ann_file,
                'img_folder': img_folder,
            })
        return sources

    if ann_key not in dataset_config:
        return sources

    ann_file = Path(dataset_config[ann_key])
    root_dir = Path(dataset_config.get('root_dir', ''))
    if not ann_file.is_absolute() and str(root_dir):
        ann_file = root_dir / ann_file

    img_folder = dataset_config.get(img_key)
    if img_folder:
        img_folder = Path(img_folder)
        if not img_folder.is_absolute() and str(root_dir):
            img_folder = root_dir / img_folder
    else:
        img_folder = root_dir / 'images' / image_set if str(root_dir) else Path('images') / image_set

    sources.append({
        'name': dataset_config.get('name', image_set),
        'ann_file': ann_file,
        'img_folder': img_folder,
    })
    return sources


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def collect_split_summary(sources: list[dict], image_set: str) -> dict:
    """收集 split 统计，并为多数据集分配唯一 image_id offset。"""
    summary = {
        'split': image_set,
        'num_sources': len(sources),
        'total_images': 0,
        'total_annotations': 0,
        'categories': [],
        'sources': [],
    }
    normalized_categories = None
    image_id_offset = 0

    for source in sources:
        with open(source['ann_file'], 'r', encoding='utf-8') as f:
            data = json.load(f)

        categories = data.get('categories', [])
        current_categories = sorted(
            ((cat.get('id'), cat.get('name')) for cat in categories),
            key=lambda item: item[0],
        )
        if normalized_categories is None:
            normalized_categories = current_categories
            summary['categories'] = [
                {'id': cat_id, 'name': cat_name}
                for cat_id, cat_name in current_categories
            ]
        elif current_categories != normalized_categories:
            raise ValueError(
                f"{image_set} split 的类别定义不一致: {source['ann_file']} 与前序数据集不匹配"
            )

        images = data.get('images', [])
        annotations = data.get('annotations', [])
        image_ids = [int(img['id']) for img in images]
        max_image_id = max(image_ids) if image_ids else -1

        source_summary = {
            'name': source['name'],
            'ann_file': str(source['ann_file']),
            'img_folder': str(source['img_folder']),
            'num_images': len(images),
            'num_annotations': len(annotations),
            'image_id_offset': image_id_offset,
            'max_source_image_id': max_image_id,
        }
        summary['sources'].append(source_summary)
        summary['total_images'] += len(images)
        summary['total_annotations'] += len(annotations)

        image_id_offset += max_image_id + 1 if max_image_id >= 0 else 0

    return summary


def prepare_coco_ground_truth(output_dir: Path, summary: dict, image_set: str) -> Path | None:
    """为评估准备 COCO ground truth；多数据集时生成带 offset 的合并标注。"""
    if not summary.get('sources'):
        return None

    if len(summary['sources']) == 1:
        return Path(summary['sources'][0]['ann_file'])

    merged = {
        'images': [],
        'annotations': [],
        'categories': summary.get('categories', []),
    }
    next_ann_id = 1

    for source in summary['sources']:
        with open(source['ann_file'], 'r', encoding='utf-8') as f:
            data = json.load(f)

        offset = int(source['image_id_offset'])
        for image in data.get('images', []):
            new_image = dict(image)
            new_image['id'] = int(image['id']) + offset
            merged['images'].append(new_image)

        for ann in data.get('annotations', []):
            new_ann = dict(ann)
            new_ann['id'] = next_ann_id
            next_ann_id += 1
            new_ann['image_id'] = int(ann['image_id']) + offset
            merged['annotations'].append(new_ann)

    merged_path = output_dir / f'merged_{image_set}_annotations.json'
    _write_json(merged_path, merged)
    return merged_path


def write_run_manifest(output_dir: Path, payload: dict):
    """写出运行清单，供论文与复现实验追踪。"""
    _write_json(output_dir / 'run_manifest.json', payload)


def validate_runtime_config(config: dict, logger):
    """对当前实际支持的配置能力做显式校验与提示。"""
    model_type = config.get('model', {}).get('type', 'detr').lower()
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    testing_config = config.get('testing', {})
    device_config = config.get('device', {})

    if output_config.get('tensorboard') is not None:
        logger.warning("配置项 output.tensorboard 当前未实现，已忽略")

    if testing_config.get('nms_threshold') is not None:
        logger.warning("配置项 testing.nms_threshold 当前未生效；DETR/Deformable DETR 评估未使用额外 NMS")

    if device_config.get('gpu_ids') is not None:
        logger.warning("配置项 device.gpu_ids 当前未生效；统一训练入口仅根据 device.type 选择单设备")

    if training_config.get('resize_schedule') and model_type == 'detr':
        raise ValueError(
            "HF DETR 当前不支持真正生效的 progressive resizing；"
            "请改用 Deformable DETR 主线或移除 training.resize_schedule"
        )


def resolve_device(config: dict) -> torch.device:
    preferred = config.get('device', {}).get('type', 'cuda')
    if preferred == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def resolve_amp(config, device):
    """解析 AMP 配置与 dtype。"""
    use_amp = config['training'].get('amp')
    if use_amp is None:
        use_amp = config['training'].get('use_amp')
    if use_amp is None:
        use_amp = config.get('amp', {}).get('enabled', False)

    use_amp = bool(use_amp) and device.type == 'cuda'
    if not use_amp:
        return False, torch.float16, None

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = GradScaler('cuda') if amp_dtype == torch.float16 else None
    return True, amp_dtype, scaler


def maybe_compile_model(model, config, logger):
    """按配置启用 torch.compile。"""
    if not config.get('training', {}).get('compile', False):
        return model
    if not hasattr(torch, 'compile'):
        logger.warning("当前 PyTorch 不支持 torch.compile，跳过")
        return model
    try:
        compiled = torch.compile(model, mode='reduce-overhead')
        logger.info("torch.compile 已启用")
        return compiled
    except Exception as exc:  # pragma: no cover - 依赖运行环境
        logger.warning(f"torch.compile 失败，回退 eager 模式: {exc}")
        return model


def update_dataset_resolution(dataloader, new_size, config, logger):
    """
    更新数据集的分辨率以支持渐进式缩放。
    仅 Deformable DETR 主线支持真正生效的动态更新。
    """
    model_type = config.get('model', {}).get('type', 'detr').lower()
    
    if model_type in ('deformable_detr', 'deformable-detr'):
        from dataset.deformable_dataset import make_deformable_transforms
        
        # 创建新的临时配置以生成新的 transforms
        temp_config = deepcopy(config)
        if 'dataset' not in temp_config:
            temp_config['dataset'] = {}
        if 'augmentation' not in temp_config['dataset']:
            temp_config['dataset']['augmentation'] = {}
            
        # 设置新的目标尺寸 (单一边长)
        temp_config['dataset']['augmentation']['train_max_size'] = new_size
        temp_config['dataset']['augmentation']['train_scales'] = [new_size]
        
        new_transforms = make_deformable_transforms('train', temp_config)
        
        # 更新底层数据集的 transforms
        dataset = dataloader.dataset
        
        # 处理 Subset/ConcatDataset 的嵌套
        def _update_ds_transforms(ds):
            if hasattr(ds, 'datasets'):
                for sub_ds in ds.datasets:
                    _update_ds_transforms(sub_ds)
            elif hasattr(ds, 'dataset'):
                _update_ds_transforms(ds.dataset)
            elif hasattr(ds, '_transforms'):
                ds._transforms = new_transforms
                
        _update_ds_transforms(dataset)
        logger.info(f"🔄 [渐进式缩放] 已更新数据增强最大分辨率为: {new_size}")
        
    else:
        raise RuntimeError(
            "HF DETR 不支持运行时动态重建 progressive resizing；"
            "该分支只保留在配置校验层做 fail-fast，请移除 training.resize_schedule 或切换到 Deformable DETR"
        )

def main(args):
    """主训练流程"""
    config = load_config(args.config)
    output_dir = resolve_output_dir(config, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger('train', output_dir / 'train.log')
    logger.info("🚀 开始训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"模型类型: {config.get('model', {}).get('type', 'detr')}")
    logger.info(f"输出目录: {output_dir}")
    validate_runtime_config(config, logger)

    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)

    train_sources = resolve_dataset_sources(config, 'train')
    if not train_sources:
        raise ValueError("未找到 train split 配置，请检查 dataset.train_ann 或 dataset.datasets[*].train_ann")
    train_summary = collect_split_summary(train_sources, 'train')
    _write_json(output_dir / 'dataset_train_summary.json', train_summary)

    val_sources = resolve_dataset_sources(config, 'val')
    val_summary = collect_split_summary(val_sources, 'val') if val_sources else None
    if val_summary:
        _write_json(output_dir / 'dataset_val_summary.json', val_summary)

    setup_seed(config, logger)

    device = resolve_device(config)
    logger.info(f"设备: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    logger.info("构建模型...")
    model = build_model(config)
    model = model.to(device)

    logger.info("构建数据加载器...")
    train_loader, _ = build_dataloader_for_model(config, 'train')

    val_loader = None
    coco_gt = None
    val_ann_path = None
    if val_summary:
        val_loader, _ = build_dataloader_for_model(config, 'val')
        val_ann_path = prepare_coco_ground_truth(output_dir, val_summary, 'val')
        if val_ann_path is not None and val_ann_path.exists():
            coco_gt = COCO(str(val_ann_path))
            logger.info(f"验证标注: {val_ann_path}")
        else:
            logger.warning(f"未找到验证标注文件: {val_ann_path}，将跳过评估")
            val_loader = None

    write_run_manifest(output_dir, {
        'config_path': str(Path(args.config).resolve()),
        'output_dir': str(output_dir.resolve()),
        'model_type': config.get('model', {}).get('type', 'detr'),
        'device': str(device),
        'train_summary_path': str((output_dir / 'dataset_train_summary.json').resolve()),
        'val_summary_path': str((output_dir / 'dataset_val_summary.json').resolve()) if val_summary else None,
        'merged_val_ann': str(val_ann_path.resolve()) if val_ann_path is not None and val_ann_path.exists() else None,
        'resume_checkpoint': args.resume or config['training'].get('resume'),
    })

    optimizer = build_optimizer(model, config, logger)
    scheduler = build_scheduler(optimizer, config)
    if scheduler:
        logger.info(f"学习率调度器: {config['training']['lr_scheduler']['type']}")

    use_amp, amp_dtype, scaler = resolve_amp(config, device)
    if use_amp:
        logger.info(f"启用 AMP: dtype={amp_dtype}")

    resume_checkpoint = args.resume or config['training'].get('resume')
    is_resume = bool(resume_checkpoint)
    metrics_logger = MetricsLogger(output_dir, resume=is_resume)

    start_epoch = 1
    best_metric = None
    best_loss = float('inf')
    if resume_checkpoint:
        logger.info(f"从检查点恢复: {resume_checkpoint}")
        checkpoint = load_checkpoint(
            checkpoint_path=Path(resume_checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=str(device),
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_metric = checkpoint.get('best_metric')
        best_loss = checkpoint.get('best_loss', best_loss)
    
    model = maybe_compile_model(model, config, logger)

    num_epochs = config['training'].get('num_epochs') or config['training'].get('max_epochs', 50)
    log_interval = config['training'].get('log_interval', 50)
    save_interval = config['training'].get('save_interval', 5)
    eval_interval = config['training'].get('eval_interval', 1)
    max_iters = config['training'].get('max_iters')
    score_threshold = config.get('testing', {}).get('confidence_threshold', 0.05)
    save_best_only = bool(config.get('output', {}).get('save_best_only', False))

    if start_epoch > num_epochs:
        logger.warning(f"恢复后的起始 epoch={start_epoch} 已超过 max_epochs={num_epochs}，训练结束")
        return

    logger.info(f"开始训练: Epoch {start_epoch} -> {num_epochs}")
    logger.info(f"梯度累积: {config['training'].get('grad_accum_steps', config['training'].get('grad_accum', 1))}")
    logger.info(f"梯度裁剪: {config['training'].get('clip_max_norm', 0.1)}")

    last_metrics = {}
    
    # 提取 progressive resizing schedule
    resize_schedule = config['training'].get('resize_schedule', None)
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'=' * 60}")
        
        # 检查是否需要更新分辨率
        if resize_schedule:
            # schedule 格式例如: [[1, 640], [20, 800], [40, 960]]
            # 找到当前 epoch 对应的目标尺寸
            target_size = None
            for sched_epoch, size in sorted(resize_schedule, key=lambda x: x[0]):
                if epoch >= sched_epoch:
                    target_size = size
            
            # 如果这是一个新的分辨率变更点
            if target_size and any(epoch == sched_epoch for sched_epoch, _ in resize_schedule):
                logger.info(f"📈 触发 Progressive Resizing, 调整输入尺寸为: {target_size}")
                update_dataset_resolution(train_loader, target_size, config, logger)
                
        epoch_start = time.time()

        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            config=config,
            log_interval=log_interval,
            max_iters=max_iters,
            use_amp=use_amp,
            scaler=scaler,
            amp_dtype=amp_dtype,
        )

        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
        logger.info(f"Epoch {epoch} 学习率: {current_lr:.6e}")

        val_metrics = {}
        if val_loader is not None and coco_gt is not None and epoch % eval_interval == 0:
            logger.info("开始验证...")
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                coco_gt=coco_gt,
                logger=logger,
                config=config,
                score_threshold=score_threshold,
            )
            logger.info(f"验证结果: {val_metrics}")
            _write_json(output_dir / 'eval_latest.json', {
                'epoch': epoch,
                'score_threshold': score_threshold,
                'metrics': val_metrics,
            })

        metrics = {
            'loss': avg_loss,
            'lr': current_lr,
            **val_metrics,
        }
        metrics_logger.log(metrics, step=epoch, epoch=epoch)
        last_metrics = metrics

        if (not save_best_only) and epoch % save_interval == 0:
            checkpoint_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_checkpoint(
                model=checkpoint_model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * len(train_loader),
                metrics=metrics,
                output_dir=output_dir,
                filename=f'checkpoint_epoch_{epoch}.pth',
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_loss=best_loss if best_loss < float('inf') else None,
            )

        if val_metrics and 'mAP' in val_metrics:
            current_metric = val_metrics['mAP']
            if best_metric is None or current_metric > best_metric:
                best_metric = current_metric
                logger.info(f"🏆 新最佳 mAP: {best_metric:.4f}")
                checkpoint_model = model._orig_mod if hasattr(model, '_orig_mod') else model
                save_checkpoint(
                    model=checkpoint_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * len(train_loader),
                    metrics=metrics,
                    output_dir=output_dir,
                    filename='best.pth',
                    is_best=True,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_metric=best_metric,
                    best_loss=best_loss if best_loss < float('inf') else None,
                )
        elif avg_loss < best_loss:
            best_loss = avg_loss
            logger.info(f"🏆 新最佳 loss: {best_loss:.4f}")
            checkpoint_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            save_checkpoint(
                model=checkpoint_model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * len(train_loader),
                metrics=metrics,
                output_dir=output_dir,
                filename='best.pth',
                is_best=True,
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_metric,
                best_loss=best_loss,
            )

        logger.info(f"Epoch {epoch} 耗时: {epoch_time:.1f}s")

    if not save_best_only:
        checkpoint_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        save_checkpoint(
            model=checkpoint_model,
            optimizer=optimizer,
            epoch=num_epochs,
            step=num_epochs * len(train_loader),
            metrics=last_metrics,
            output_dir=output_dir,
            filename='last.pth',
            scheduler=scheduler,
            scaler=scaler,
            best_metric=best_metric,
            best_loss=best_loss if best_loss < float('inf') else None,
        )

    logger.info("🎉 训练完成")
    logger.info(f"模型保存在: {output_dir}")
    write_run_manifest(output_dir, {
        'config_path': str(Path(args.config).resolve()),
        'output_dir': str(output_dir.resolve()),
        'model_type': config.get('model', {}).get('type', 'detr'),
        'device': str(device),
        'train_summary_path': str((output_dir / 'dataset_train_summary.json').resolve()),
        'val_summary_path': str((output_dir / 'dataset_val_summary.json').resolve()) if val_summary else None,
        'merged_val_ann': str(val_ann_path.resolve()) if val_ann_path is not None and val_ann_path.exists() else None,
        'resume_checkpoint': args.resume or config['training'].get('resume'),
        'best_metric': best_metric,
        'best_loss': None if best_loss == float('inf') else best_loss,
        'last_metrics': last_metrics,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DETR/Deformable DETR 统一训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录（可选，覆盖配置）')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')

    main(parser.parse_args())
