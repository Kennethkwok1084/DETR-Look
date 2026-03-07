#!/usr/bin/env python3
"""
通用训练工具函数
支持 DETR (HF格式) 和 Deformable DETR (官方格式) 双数据流
"""

import torch
from tqdm import tqdm
from torch.amp import autocast
import sys
from pathlib import Path

# === 模块缓存：避免 sys.path 污染 ===
_deformable_utils_cache = {}

def _import_deformable_utils():
    """隔离导入 Deformable DETR 工具函数，不污染 sys.path"""
    if _deformable_utils_cache:
        return _deformable_utils_cache
    
    _original_sys_path = sys.path.copy()
    _third_party_path = Path(__file__).parent.parent / "third_party" / "deformable_detr"
    sys.path.insert(0, str(_third_party_path))
    
    try:
        from util.misc import NestedTensor
        _deformable_utils_cache['NestedTensor'] = NestedTensor
    finally:
        # 恢复 sys.path（不清理 util 模块，支持 DataLoader 反序列化）
        sys.path[:] = _original_sys_path
    
    return _deformable_utils_cache


def _get_train_hparams(config):
    """提取训练循环需要的通用超参数。"""
    train_config = config.get('training', {})
    grad_accum = int(train_config.get('grad_accum_steps', train_config.get('grad_accum', 1)) or 1)
    clip_max_norm = train_config.get('clip_max_norm', 0.1)
    return max(1, grad_accum), clip_max_norm


def train_one_epoch_detr(model, dataloader, optimizer, device, epoch, logger, 
                         log_interval=50, max_iters=None, use_amp=False, scaler=None, amp_dtype=torch.float16):
    """
    DETR 训练一个 epoch（HuggingFace 数据流）
    
    Args:
        model: DETR 模型
        dataloader: HF 格式的数据加载器（返回 pixel_values, pixel_mask, labels）
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        logger: 日志记录器
        log_interval: 日志输出间隔
        max_iters: 最大迭代数（用于快速测试）
        use_amp: 是否使用混合精度
        scaler: GradScaler
        amp_dtype: AMP 数据类型
    
    Returns:
        平均损失
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    grad_accum, clip_max_norm = _get_train_hparams(getattr(model, 'config', {}))
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (DETR-HF)", ncols=100)
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch['pixel_values'].to(device)
        pixel_mask = batch['pixel_mask'].to(device) if 'pixel_mask' in batch else None
        labels = batch['labels']
        
        # 将 labels 移到设备
        labels = [
            {
                'class_labels': item['class_labels'].to(device),
                'boxes': item['boxes'].to(device)
            }
            for item in labels
        ]
        
        # 前向传播
        if use_amp:
            with autocast('cuda', dtype=amp_dtype):
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                raw_loss = outputs.loss
        else:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            raw_loss = outputs.loss
        
        loss = raw_loss / grad_accum if grad_accum > 1 else raw_loss
        
        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        should_step = ((batch_idx + 1) % grad_accum == 0)
        if should_step:
            if use_amp and scaler is not None:
                if clip_max_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # 记录
        epoch_loss += raw_loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{raw_loss.item():.4f}",
            'avg': f"{epoch_loss / num_batches:.4f}"
        })
        
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch [{epoch}] Iter [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {raw_loss.item():.4f} Avg: {epoch_loss / num_batches:.4f}"
            )
        
        if max_iters and num_batches >= max_iters:
            logger.info(f"达到最大迭代数 {max_iters}")
            break
    
    if num_batches > 0 and num_batches % grad_accum != 0:
        if use_amp and scaler is not None:
            if clip_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return epoch_loss / num_batches if num_batches > 0 else 0.0


def train_one_epoch_deformable(model, dataloader, optimizer, device, epoch, logger,
                                log_interval=50, max_iters=None, use_amp=False, scaler=None, amp_dtype=torch.float16):
    """
    Deformable DETR 训练一个 epoch（官方数据流）
    
    Args:
        model: Deformable DETR 模型封装
        dataloader: 官方格式的数据加载器（返回 NestedTensor, targets）
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch
        logger: 日志记录器
        log_interval: 日志输出间隔
        max_iters: 最大迭代数
        use_amp: 是否使用混合精度
        scaler: GradScaler
        amp_dtype: AMP 数据类型
    
    Returns:
        平均损失
    """
    # 使用模块缓存（避免 sys.path 污染）
    utils = _import_deformable_utils()
    NestedTensor = utils['NestedTensor']
    
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    grad_accum, clip_max_norm = _get_train_hparams(getattr(model, 'config', {}))
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} (Deformable)", ncols=100)
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (samples, targets) in enumerate(pbar):
        # 将数据移到设备
        if isinstance(samples, NestedTensor):
            samples = NestedTensor(samples.tensors.to(device), samples.mask.to(device))
        else:
            samples = samples.to(device)
        
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # 前向传播
        if use_amp:
            with autocast('cuda', dtype=amp_dtype):
                loss_dict = model(samples, targets)
                # 应用 weight_dict（与官方实现一致）
                # SetCriterion 返回未加权的 loss，需要手动应用 weight_dict
                weight_dict = model.criterion.weight_dict
                raw_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        else:
            loss_dict = model(samples, targets)
            weight_dict = model.criterion.weight_dict
            raw_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        loss = raw_loss / grad_accum if grad_accum > 1 else raw_loss
        
        # 反向传播
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        should_step = ((batch_idx + 1) % grad_accum == 0)
        if should_step:
            if use_amp and scaler is not None:
                if clip_max_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # 记录
        epoch_loss += raw_loss.item()
        num_batches += 1
        
        # 显示关键损失
        loss_str = f"loss={raw_loss.item():.3f}"
        if 'loss_ce' in loss_dict:
            loss_str += f" ce={loss_dict['loss_ce'].item():.3f}"
        if 'loss_bbox' in loss_dict:
            loss_str += f" bbox={loss_dict['loss_bbox'].item():.3f}"
        if 'loss_giou' in loss_dict:
            loss_str += f" giou={loss_dict['loss_giou'].item():.3f}"
        
        pbar.set_postfix_str(loss_str)
        
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch [{epoch}] Iter [{batch_idx + 1}/{len(dataloader)}] {loss_str}"
            )
        
        if max_iters and num_batches >= max_iters:
            logger.info(f"达到最大迭代数 {max_iters}")
            break
    
    if num_batches > 0 and num_batches % grad_accum != 0:
        if use_amp and scaler is not None:
            if clip_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    return epoch_loss / num_batches if num_batches > 0 else 0.0


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger, config,
                    log_interval=50, max_iters=None, use_amp=False, scaler=None, amp_dtype=torch.float16):
    """
    统一训练入口（自动选择数据流）
    
    根据 config['model']['type'] 自动选择：
    - 'detr': 使用 HF 数据流
    - 'deformable_detr': 使用官方数据流
    """
    model_type = config.get('model', {}).get('type', 'detr').lower()
    model.config = config
    
    if model_type == 'deformable_detr' or model_type == 'deformable-detr':
        return train_one_epoch_deformable(
            model, dataloader, optimizer, device, epoch, logger,
            log_interval, max_iters, use_amp, scaler, amp_dtype
        )
    else:
        return train_one_epoch_detr(
            model, dataloader, optimizer, device, epoch, logger,
            log_interval, max_iters, use_amp, scaler, amp_dtype
        )
