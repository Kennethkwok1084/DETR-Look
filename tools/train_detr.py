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
from torch.amp import autocast, GradScaler

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataset import build_dataloader
from models import build_detr_model
from utils import MetricsLogger, save_checkpoint, load_checkpoint, setup_logger
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
    scaler=None,
    use_amp=False,
    amp_dtype=None,
):
    """训练一个epoch
    
    Args:
        scaler: AMP GradScaler（如启用AMP）
        use_amp: 是否使用混合精度训练
        amp_dtype: AMP数据类型（torch.float16或torch.bfloat16）
    """
    model.train()
    
    epoch_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, encoding in enumerate(pbar):
        # encoding已经是预处理好的dict，包含pixel_values, pixel_mask, labels
        # 由collate_fn在worker进程中并行处理完成
        
        # 移到设备
        pixel_values = encoding['pixel_values'].to(device)
        pixel_mask = encoding['pixel_mask'].to(device)
        labels = encoding['labels']  # 已经是正确的格式
        
        # 将labels移到设备
        labels = [
            {
                'class_labels': item['class_labels'].to(device),
                'boxes': item['boxes'].to(device),
            }
            for item in labels
        ]
        
        # 前向传播（支持AMP）
        if use_amp:
            with autocast('cuda', dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = outputs.loss
        
        # 反向传播（支持AMP）
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # 梯度裁剪（启用，防止梯度爆炸）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪（启用，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
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
    
    # ===== 高级优化配置 =====
    
    # 1. TF32 加速（Ampere架构免费提速，几乎无精度损失）
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TF32 已启用（Ampere架构加速）")
    
    # 2. AMP 配置（优先使用BF16，回退FP16）
    use_amp = config['training'].get('amp', False) and torch.cuda.is_available()
    amp_dtype = None
    if use_amp:
        # 检查是否支持BF16（Ampere及以上架构）
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("✅ AMP使用BF16（更稳定，动态范围更大）")
        else:
            amp_dtype = torch.float16
            print("⚠️  AMP使用FP16（BF16不支持，使用传统混合精度）")
    
    scaler = GradScaler('cuda') if (use_amp and amp_dtype == torch.float16) else None
    
    # 打印配置信息
    print(f"\n📋 训练配置:")
    print(f"  数据集: {config['dataset']['name']}")
    print(f"  类别数: {config['dataset']['num_classes']}")
    print(f"  模型: {config['model']['name']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Max Epochs: {config['training']['max_epochs']}")
    print(f"  学习率: {config['training']['optimizer']['lr']}")
    print(f"  AMP 混合精度: {'✓ 启用' if use_amp else '✗ 禁用'}")
    
    if args.max_iter:
        print(f"  最大迭代: {args.max_iter}")
        config['training']['max_iters'] = args.max_iter
    
    # 子集采样配置
    subset_size = args.subset_size or config['training'].get('subset_size')
    if subset_size:
        print(f"  子集大小: {subset_size}")
        config['training']['subset_size'] = subset_size
    
    overfit_mode = args.overfit or config['training'].get('overfit', False)
    if overfit_mode:
        print(f"  ⚠️  过拟合模式：已启用（用于验证训练流程）")
        config['training']['overfit'] = True
        
        # 设置全局随机种子（保证过拟合测试可复现）
        import random
        import numpy as np
        seed = config['training'].get('subset_seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"  🎲 全局随机种子已设置: {seed}（保证可复现）")
    
    # Progressive Resizing 配置
    resize_schedule = config['training'].get('resize_schedule')
    if resize_schedule:
        print(f"  Progressive Resizing: {resize_schedule}")
    
    # 设置输出目录
    output_dir = setup_output_dir(config, args)
    print(f"  输出目录: {output_dir}")
    
    # 保存配置
    save_config(config, output_dir)
    
    # Resume 检查（在使用前定义）
    resume_checkpoint = args.resume or config['training'].get('resume')
    is_resume = bool(resume_checkpoint)
    
    # 设置日志（Resume 模式续写）
    logger = setup_logger('train', output_dir / 'train.log')
    metrics_logger = MetricsLogger(output_dir, resume=is_resume)
    
    # 构建数据加载器
    print("\n" + "="*60)
    print("📦 构建数据加载器")
    print("="*60)
    
    # 先创建image_processor（用于在worker中预处理）
    model_name = config['model']['name']
    if not model_name.startswith('facebook/'):
        model_name = f'facebook/{model_name}'
    image_processor = DetrImageProcessor.from_pretrained(model_name)
    
    # 构建DataLoader（传入processor实现worker中并行预处理）
    train_loader = build_dataloader(config, 'train', image_processor=image_processor)
    val_loader = build_dataloader(config, 'val', image_processor=image_processor)
    
    # 构建模型
    print("\n" + "="*60)
    print("🏗️  构建模型")
    print("="*60)
    
    model = build_detr_model(config)
    model = model.to(device)
    
    # ===== torch.compile 优化（PyTorch 2.0+ Transformer加速）=====
    use_compile = config['training'].get('compile', False)
    if use_compile and hasattr(torch, 'compile'):
        print("\n🚀 启用 torch.compile 优化...")
        try:
            # mode='reduce-overhead' 对Transformer效果好
            model = torch.compile(model, mode='reduce-overhead')
            print("✅ torch.compile 启用成功（预期提速10-30%）")
        except Exception as e:
            print(f"⚠️  torch.compile 失败，继续使用eager模式: {e}")
    
    # 构建优化器
    print("\n📊 构建优化器")
    optimizer_config = config['training']['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config['lr']),
        weight_decay=float(optimizer_config['weight_decay']),
        betas=tuple(optimizer_config['betas']),
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config['training']['lr_scheduler']['step_size']),
        gamma=float(config['training']['lr_scheduler']['gamma']),
    )
    
    # Resume 逻辑：从 checkpoint 恢复（已在前面定义 resume_checkpoint）
    start_epoch = 1
    best_metric_value = None
    loaded_checkpoint = None  # 用于后续访问checkpoint字典
    
    if resume_checkpoint:
        print("\n" + "="*60)
        print(f"🔄 从 checkpoint 恢复训练: {resume_checkpoint}")
        print("="*60)
        loaded_checkpoint = load_checkpoint(
            checkpoint_path=Path(resume_checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            restore_rng_state=True,
        )
        start_epoch = loaded_checkpoint.get('epoch', 0) + 1  # 从下一个epoch继续
        best_metric_value = loaded_checkpoint.get('best_metric')
        print(f"将从 Epoch {start_epoch} 继续训练")
        if best_metric_value is not None:
            print(f"历史最佳指标: {best_metric_value:.4f}")
    
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
    
    # Resume时恢复best_loss（避免validation跳过时第一个epoch总是覆盖best.pth）
    best_loss = loaded_checkpoint.get('best_loss', float('inf')) if loaded_checkpoint else float('inf')
    best_map = 0.0 if best_metric_value is None else best_metric_value
    start_time = time.time()
    
    # 检查 Resume 后 epoch 范围是否有效
    skip_training = False
    if start_epoch > max_epochs:
        logger.warning(f"⚠️  Resume 的起始 epoch ({start_epoch}) 已超过 max_epochs ({max_epochs})")
        logger.warning(f"    → 训练将直接结束，不会执行新的 epoch")
        logger.warning(f"    → 建议增加 max_epochs 或检查 checkpoint")
        logger.warning(f"    → 将跳过训练和最终保存，避免覆盖已有 checkpoint")
        skip_training = True
    
    for epoch in range(start_epoch, max_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{max_epochs}")
        logger.info(f"{'='*60}")
        
        # Progressive Resizing: 根据 epoch 调整输入分辨率
        if resize_schedule:
            # resize_schedule 格式: [[epoch1, size1], [epoch2, size2], ...]
            # 或 [[epoch1, {"shortest": s, "longest": l}], ...]
            current_size = None
            for schedule_epoch, size_config in resize_schedule:
                if epoch >= schedule_epoch:
                    current_size = size_config
            
            if current_size:
                # 支持两种格式：整数或字典
                if isinstance(current_size, dict):
                    # 字典格式：兼容两种键名
                    # {"shortest": 640, "longest": 1333} 或 {"shortest_edge": 640, "longest_edge": 1333}
                    shortest = current_size.get('shortest') or current_size.get('shortest_edge', 800)
                    longest = current_size.get('longest') or current_size.get('longest_edge', 1333)
                else:
                    # 整数格式：短边为该值，长边使用默认上限
                    shortest = current_size
                    longest = 1333  # DETR 默认上限
                
                # 基本数值验证，避免无效尺寸
                try:
                    shortest = int(shortest)
                    longest = int(longest)
                    if shortest <= 0 or longest <= 0 or shortest > longest:
                        raise ValueError(f"Invalid size: shortest={shortest}, longest={longest}")
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Progressive Resizing 跳过：无效的尺寸配置 "
                        f"(shortest={shortest}, longest={longest}): {e}"
                    )
                    continue  # 跳过该 epoch 的 resizing
                
                # 兼容不同版本的 transformers API
                # 旧版本：size + max_size
                # 新版本：size={"shortest_edge": ..., "longest_edge": ...}
                try:
                    # 尝试新版本 API (transformers >= 4.26)
                    image_processor.size = {"shortest_edge": shortest, "longest_edge": longest}
                except (TypeError, AttributeError):
                    # 回退到旧版本 API
                    image_processor.size = shortest
                    image_processor.max_size = longest
                
                logger.info(f"Progressive Resizing: shortest_edge={shortest}, max_size={longest}")
        
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
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,  # 传递amp_dtype
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
                config=config,
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
        
        # 保存checkpoint（完整状态）
        if epoch % save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * len(train_loader),
                metrics=metrics,
                output_dir=output_dir,
                filename=f"checkpoint_epoch_{epoch}.pth",
                scheduler=scheduler,
                scaler=scaler,
                best_metric=best_map if best_map > 0 else None,
                best_loss=best_loss if best_loss < float('inf') else None,
                save_rng_state=True,
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
                    scheduler=scheduler,
                    scaler=scaler,
                    best_metric=best_map,
                    best_loss=best_loss if best_loss < float('inf') else None,
                    save_rng_state=True,
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
                    scheduler=scheduler,
                    scaler=scaler,
                    best_metric=None,  # 使用loss时不记录best_metric
                    best_loss=best_loss,
                    save_rng_state=True,
                )
        
        # 如果设置了max_iters且已达到预期epoch数，停止训练
        # 注意：只有在max_iters很小时才提前停止（真正的冒烟测试）
        if max_iters and max_iters <= 200 and epoch >= 2:
            logger.info(f"冒烟测试模式：已完成 {epoch} 个epoch，停止训练")
            break
    
    # 如果因 start_epoch > max_epochs 跳过了训练，不保存 last.pth
    if skip_training:
        logger.warning("⚠️  已跳过训练，不保存 last.pth 以避免覆盖已有模型")
        logger.info(f"\n{'='*60}")
        logger.info("训练已结束（未执行新 epoch）")
        logger.info(f"{'='*60}")
        return
    
    # 确保 epoch 和 metrics 始终有定义（避免空循环崩溃）
    if 'epoch' not in locals():
        epoch = start_epoch - 1
    if 'metrics' not in locals():
        # 处理 best_map 为 None 的情况（避免 metrics 包含 None 值）
        safe_best_map = best_map if best_map is not None else 0.0
        metrics = {'loss': 0.0, 'mAP': safe_best_map}
    
    # 保存最终模型
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=epoch * len(train_loader) if epoch > 0 else 0,
        metrics=metrics,
        output_dir=output_dir,
        filename="last.pth",
        scheduler=scheduler,
        scaler=scaler,
        best_metric=best_map if best_map > 0 else None,
        best_loss=best_loss if best_loss < float('inf') else None,
        save_rng_state=True,
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ 训练完成！")
    logger.info(f"   总耗时: {elapsed_time / 60:.2f} 分钟")
    
    # 显示最佳指标（优先显示mAP，否则显示Loss）
    if best_map > 0:
        logger.info(f"   最佳mAP: {best_map:.4f}")
    elif best_loss < float('inf'):
        logger.info(f"   最佳Loss: {best_loss:.4f}")
    else:
        logger.info(f"   最佳指标: 未记录")
    
    logger.info(f"   输出目录: {output_dir}")
    logger.info(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("✅ 训练完成！")
    print(f"   总耗时: {elapsed_time / 60:.2f} 分钟")
    
    # 显示最佳指标（优先显示mAP，否则显示Loss）
    if best_map > 0:
        print(f"   最佳mAP: {best_map:.4f}")
    elif best_loss < float('inf'):
        print(f"   最佳Loss: {best_loss:.4f}")
    else:
        print(f"   最佳指标: 未记录")
    
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
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="子集大小（用于快速验证或小样本过拟合）",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="过拟合模式（关闭数据增强，固定随机种子）",
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
