#!/usr/bin/env python3
"""
模型checkpoint保存与加载
"""

import torch
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
):
    """
    保存checkpoint
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        step: 当前step
        metrics: 当前指标
        output_dir: 输出目录
        filename: 文件名
        is_best: 是否为最佳模型
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    save_path = output_dir / filename
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint 已保存: {save_path}")
    
    if is_best:
        best_path = output_dir / "best.pth"
        torch.save(checkpoint, best_path)
        print(f"🏆 最佳模型已保存: {best_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
) -> Dict[str, Any]:
    """
    加载checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 模型
        optimizer: 优化器（可选）
        device: 设备
    
    Returns:
        checkpoint字典
    """
    print(f"📂 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ Checkpoint加载成功 (Epoch {checkpoint.get('epoch', 0)}, Step {checkpoint.get('step', 0)})")
    
    return checkpoint
