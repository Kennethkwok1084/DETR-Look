#!/usr/bin/env python3
"""
DETR 训练脚本
支持从配置文件加载参数，进行交通场景目标检测训练
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from transformers import DetrImageProcessor

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset import build_dataloader
from models import build_detr_model
from utils import MetricsLogger, save_checkpoint, setup_logger
from tools.eval_detr import evaluate


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dir(config: dict, args) -> Path:
    """设置输出目录"""
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        base_dir = config['output']['base_dir']
        exp_name = config['output']['experiment_name']
        output_dir = Path(base_dir) / exp_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_config(config: dict, output_dir: Path):
    """保存配置到输出目录"""
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 配置已保存: {config_save_path}")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    image_processor,
    max_iters,
    log_interval,
    logger,
):
    """训练一个epoch"""
    model.train()
    
    epoch_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 使用DetrImageProcessor处理可变尺寸图像（自动padding并生成pixel_mask）
        # 将Tensor列表转为PIL/numpy格式以供processor处理
        images_pil = [img.cpu().numpy().transpose(1, 2, 0) for img in images]
        
        # processor会自动padding到最大尺寸并返回pixel_values和pixel_mask
        encoding = image_processor(
            images=images_pil,
            annotations=[{'boxes': t['boxes'].tolist(), 'labels': t['labels'].tolist()} for t in targets],
            return_tensors='pt'
        )
        
        # 移到设备
        pixel_values = encoding['pixel_values'].to(device)
        pixel_mask = encoding['pixel_mask'].to(device)
        
        # 重构targets（processor可能重新排序/归一化boxes）
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # 前向传播
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
        
        # 计算loss
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选）
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        
        optimizer.step()
        
        # 记录
        epoch_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{epoch_loss / num_batches:.4f}"
        })
        
        # 日志输出
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch [{epoch}] Iter [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} Avg Loss: {epoch_loss / num_batches:.4f}"
            )
        
        # 检查是否达到最大迭代
        if max_iters and num_batches >= max_iters:
            logger.info(f"达到最大迭代数 {max_iters}，停止训练")
            break
    
    return epoch_loss / num_batches if num_batches > 0 else 0.0


def train(config: dict, args):
    """
    训练主函数
    """
    print("\n" + "="*60)
    print("🚀 开始训练 DETR 模型")
    print("="*60)
    
    # 设置设备
    device = torch.device(config['device']['type'] if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 设备: {device}")
    
    # 打印配置信息
    print(f"\n📋 训练配置:")
    print(f"  数据集: {config['dataset']['name']}")
    print(f"  类别数: {config['dataset']['num_classes']}")
    print(f"  模型: {config['model']['name']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Max Epochs: {config['training']['max_epochs']}")
    print(f"  学习率: {config['training']['optimizer']['lr']}")
    
    if args.max_iter:
        print(f"  最大迭代: {args.max_iter}")
        config['training']['max_iters'] = args.max_iter
    
    # 设置输出目录
    output_dir = setup_output_dir(config, args)
    print(f"  输出目录: {output_dir}")
    
    # 保存配置
    save_config(config, output_dir)
    
    # 设置日志
    logger = setup_logger('train', output_dir / 'train.log')
    metrics_logger = MetricsLogger(output_dir)
    
    # 构建数据加载器
    print("\n" + "="*60)
    print("📦 构建数据加载器")
    print("="*60)
    
    train_loader = build_dataloader(config, 'train')
    val_loader = build_dataloader(config, 'val')
    
    # 构建模型
    print("\n" + "="*60)
    print("🏗️  构建模型")
    print("="*60)
    
    model = build_detr_model(config)
    model = model.to(device)
    
    # 初始化图像处理器（用于可变尺寸padding）
    image_processor = DetrImageProcessor.from_pretrained(config['model']['name'])
    
    # 构建优化器
    print("\n📊 构建优化器")
    optimizer_config = config['training']['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['lr'],
        weight_decay=optimizer_config['weight_decay'],
        betas=optimizer_config['betas'],
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_scheduler']['step_size'],
        gamma=config['training']['lr_scheduler']['gamma'],
    )
    
    # 训练循环
    print("\n" + "="*60)
    print("🎯 开始训练")
    print("="*60)
    
    max_epochs = config['training']['max_epochs']
    max_iters = config['training'].get('max_iters')
    log_interval = config['training']['log_interval']
    save_interval = config['training']['save_interval']
    eval_interval = config['training'].get('eval_interval', 5)  # 默认每5个epoch验证一次
    
    # 加载COCO ground truth用于验证
    root_dir = Path(config['dataset']['root_dir'])
    val_ann_file = root_dir / config['dataset']['val_ann']
    coco_gt = COCO(val_ann_file)
    
    best_loss = float('inf')
    best_map = 0.0
    start_time = time.time()
    
    for epoch in range(1, max_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{max_epochs}")
        logger.info(f"{'='*60}")
        
        # 训练一个epoch
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            image_processor=image_processor,
            max_iters=max_iters,
            log_interval=log_interval,
            logger=logger,
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证（定期进行）
        val_metrics = {}
        if epoch % eval_interval == 0:
            logger.info(f"\n{'='*60}")
            logger.info("开始验证...")
            logger.info(f"{'='*60}")
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                coco_gt=coco_gt,
                logger=logger,
                score_threshold=0.05,
                image_processor=image_processor,
            )
            logger.info(f"验证结果: mAP={val_metrics.get('mAP', 0):.4f}, "
                       f"mAP@50={val_metrics.get('mAP_50', 0):.4f}, "
                       f"mAP@75={val_metrics.get('mAP_75', 0):.4f}")
        
        # 记录指标
        metrics = {
            'loss': avg_loss,
            'lr': current_lr,
        }
        metrics.update(val_metrics)  # 添加验证指标
        
        metrics_logger.log(metrics, step=epoch, epoch=epoch)
        
        logger.info(f"Epoch {epoch} 完成 - Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # 保存checkpoint
        if epoch % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * len(train_loader),
                metrics=metrics,
                output_dir=output_dir,
                filename=f"checkpoint_epoch_{epoch}.pth",
            )
        
        # 保存最佳模型（基于验证mAP，如果没有验证则使用训练loss）
        current_map = val_metrics.get('mAP', 0)
        if current_map > 0:  # 有验证结果时使用mAP
            if current_map > best_map:
                best_map = current_map
                logger.info(f"🎉 新的最佳mAP: {best_map:.4f}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * len(train_loader),
                    metrics=metrics,
                    output_dir=output_dir,
                    filename="best.pth",
                    is_best=True,
                )
        else:  # 没有验证时使用训练loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                logger.info(f"🎉 新的最佳Loss: {best_loss:.4f}")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=epoch * len(train_loader),
                    metrics=metrics,
                    output_dir=output_dir,
                    filename="best.pth",
                    is_best=True,
                )
        
        # 如果设置了max_iters且已达到预期epoch数，停止训练
        # 注意：只有在max_iters很小时才提前停止（真正的冒烟测试）
        if max_iters and max_iters <= 200 and epoch >= 2:
            logger.info(f"冒烟测试模式：已完成 {epoch} 个epoch，停止训练")
            break
    
    # 保存最终模型
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=epoch * len(train_loader),
        metrics=metrics,
        output_dir=output_dir,
        filename="last.pth",
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 训练完成！")
    logger.info(f"   总耗时: {elapsed_time / 60:.2f} 分钟")
    logger.info(f"   最佳Loss: {best_loss:.4f}")
    logger.info(f"   输出目录: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"   总耗时: {elapsed_time / 60:.2f} 分钟")
    print(f"   最佳Loss: {best_loss:.4f}")
    print(f"   输出目录: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="训练DETR模型用于交通场景目标检测"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detr_baseline.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（覆盖配置文件中的设置）",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="最大迭代次数（用于冒烟测试）",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="评估间隔",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="保存间隔",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从checkpoint恢复训练",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"📖 加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 命令行参数覆盖配置
    if args.max_iter:
        config['training']['max_iters'] = args.max_iter
    if args.eval_interval:
        config['training']['eval_interval'] = args.eval_interval
    if args.save_interval:
        config['training']['save_interval'] = args.save_interval
    
    # 设置输出目录
    output_dir = setup_output_dir(config, args)
    print(f"📂 输出目录: {output_dir}")
    
    # 保存配置
    save_config(config, output_dir)
    
    # 开始训练
    train(config, args)


if __name__ == "__main__":
    main()
