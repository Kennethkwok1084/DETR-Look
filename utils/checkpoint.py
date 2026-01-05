#!/usr/bin/env python3
"""
模型checkpoint保存与加载
支持完整训练状态：optimizer, scheduler, AMP scaler, epoch/iter, best metric, RNG状态
"""

import random
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, Any],
    output_dir: Path,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    best_metric: Optional[float] = None,
    best_loss: Optional[float] = None,
    save_rng_state: bool = True,
):
    """
    保存完整checkpoint状态
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        step: 当前step
        metrics: 当前指标
        output_dir: 输出目录
        filename: 文件名
        is_best: 是否为最佳模型
        scheduler: 学习率调度器（可选）
        scaler: AMP GradScaler（可选）
        best_metric: 最佳指标值/mAP（可选）
        best_loss: 最佳损失值（可选）
        save_rng_state: 是否保存RNG状态（可选但推荐）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础状态
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    # 学习率调度器状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # AMP scaler状态
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # 最佳指标
    if best_metric is not None:
        checkpoint['best_metric'] = best_metric
    if best_loss is not None:
        checkpoint['best_loss'] = best_loss
    
    # RNG状态（用于完全可复现）
    if save_rng_state:
        checkpoint['rng_state'] = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state_all()
    
    # 保存
    save_path = output_dir / filename
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint 已保存: {save_path}")
    print(f"   Epoch: {epoch}, Step: {step}, Metrics: {metrics}")
    
    # 保存最佳模型
    if is_best:
        best_path = output_dir / "best.pth"
        torch.save(checkpoint, best_path)
        print(f"🏆 最佳模型已保存: {best_path}")
        if best_metric is not None:
            print(f"   最佳指标: {best_metric:.4f}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = 'cpu',
    restore_rng_state: bool = True,
) -> Dict[str, Any]:
    """
    加载完整checkpoint状态
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        scaler: AMP GradScaler（可选）
        device: 设备
        restore_rng_state: 是否恢复RNG状态
    
    Returns:
        checkpoint字典
    """
    print(f"📂 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"   ✓ 优化器状态已恢复")
    
    # 加载学习率调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"   ✓ 学习率调度器状态已恢复")
    
    # 加载AMP scaler状态
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"   ✓ AMP Scaler状态已恢复")
    
    # 恢复RNG状态
    if restore_rng_state and 'rng_state' in checkpoint:
        rng_state = checkpoint['rng_state']
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and 'cuda' in rng_state:
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        print(f"   ✓ RNG状态已恢复（完全可复现）")
    
    # 打印恢复信息
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    best_metric = checkpoint.get('best_metric')
    best_loss = checkpoint.get('best_loss')
    
    print(f"✅ Checkpoint加载成功")
    print(f"   Epoch: {epoch}, Step: {step}")
    if best_metric is not None:
        print(f"   最佳指标 (mAP): {best_metric:.4f}")
    if best_loss is not None:
        print(f"   最佳损失: {best_loss:.4f}")
    
    return checkpoint
