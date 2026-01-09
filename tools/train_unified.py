#!/usr/bin/env python3
"""
统一训练脚本
支持 DETR (HF) 和 Deformable DETR (官方) 双数据流
根据配置文件中的 model.type 自动选择
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.amp import GradScaler

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import build_model, build_image_processor
from utils import setup_logger, save_checkpoint, load_checkpoint, train_one_epoch


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_dataloader_for_model(config, image_set='train'):
    """
    根据模型类型构建对应的数据加载器
    
    Args:
        config: 配置字典
        image_set: 'train' 或 'val'
    
    Returns:
        (dataloader, dataset) 元组
    """
    model_type = config.get('model', {}).get('type', 'detr').lower()
    
    if model_type == 'deformable_detr' or model_type == 'deformable-detr':
        # Deformable DETR: 使用官方数据流
        from dataset.deformable_dataset import build_deformable_dataloader
        return build_deformable_dataloader(config, image_set)
    else:
        # DETR: 使用 HF 数据流
        from dataset import build_dataloader
        
        image_processor = build_image_processor(config)
        dataloader = build_dataloader(
            config=config,
            image_set=image_set,
            image_processor=image_processor
        )
        # build_dataloader 只返回 dataloader，我们需要返回 (dataloader, dataset)
        return dataloader, dataloader.dataset


def build_optimizer(model, config):
    """构建优化器"""
    train_config = config['training']
    opt_config = train_config['optimizer']
    
    if opt_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0001),
            betas=tuple(opt_config.get('betas', [0.9, 0.999]))
        )
    elif opt_config['type'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"不支持的优化器: {opt_config['type']}")
    
    return optimizer


def build_scheduler(optimizer, config):
    """构建学习率调度器"""
    train_config = config['training']
    
    if 'lr_scheduler' not in train_config:
        return None
    
    sch_config = train_config['lr_scheduler']
    
    if sch_config['type'] == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_config['step_size'],
            gamma=sch_config.get('gamma', 0.1)
        )
    elif sch_config['type'] == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sch_config['milestones'],
            gamma=sch_config.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"不支持的调度器: {sch_config['type']}")
    
    return scheduler


def main(args):
    """主训练流程"""
    # 加载配置
    config = load_config(args.config)
    
    # 设置输出目录（支持多种配置键）
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(
            config['training'].get('output_dir') or 
            config.get('output', {}).get('base_dir', 'outputs') + '/' + 
            config.get('output', {}).get('experiment_name', 'experiment')
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger('train', output_dir / 'train.log')
    logger.info(f"🚀 开始训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"模型类型: {config.get('model', {}).get('type', 'detr')}")
    logger.info(f"输出目录: {output_dir}")
    
    # 保存配置
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")
    
    # 构建模型
    logger.info("构建模型...")
    model = build_model(config)
    model.to(device)
    
    # 构建数据加载器
    logger.info("构建数据加载器...")
    train_loader, train_dataset = build_dataloader_for_model(config, 'train')
    
    # 构建优化器
    optimizer = build_optimizer(model, config)
    logger.info(f"优化器: {config['training']['optimizer']['type']}")
    
    # 构建调度器
    scheduler = build_scheduler(optimizer, config)
    if scheduler:
        logger.info(f"学习率调度器: {config['training']['lr_scheduler']['type']}")
    
    # 混合精度（支持多种配置键，优先级：training.amp > training.use_amp > amp.enabled）
    use_amp = config['training'].get('amp')
    if use_amp is None:
        use_amp = config['training'].get('use_amp')
    if use_amp is None:
        use_amp = config.get('amp', {}).get('enabled', False)
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("启用混合精度训练 (AMP)")
    
    # 加载检查点
    start_epoch = 1
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    # 训练配置（支持多种键名）
    num_epochs = config['training'].get('num_epochs') or config['training'].get('max_epochs', 50)
    log_interval = config['training'].get('log_interval', 50)
    save_interval = config['training'].get('save_interval', 5)
    
    # 训练循环
    logger.info(f"开始训练: Epoch {start_epoch} -> {num_epochs}")
    
    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # 训练一个 epoch
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=logger,
            config=config,
            log_interval=log_interval,
            use_amp=use_amp,
            scaler=scaler
        )
        
        logger.info(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
        
        # 更新学习率
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"学习率: {current_lr:.6f}")
        
        # 保存检查点
        if epoch % save_interval == 0 or epoch == num_epochs:
            checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=0,  # 简化版不跟踪 step
                metrics={'avg_loss': avg_loss},
                output_dir=output_dir,
                filename=checkpoint_filename,
                scheduler=scheduler,
                scaler=scaler
            )
            logger.info(f"✅ 检查点已保存: {output_dir / checkpoint_filename}")
    
    logger.info(f"\n🎉 训练完成！")
    logger.info(f"模型保存在: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DETR/Deformable DETR 统一训练脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output-dir', type=str, help='输出目录（可选，覆盖配置）')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    main(args)
