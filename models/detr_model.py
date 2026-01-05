#!/usr/bin/env python3
"""
DETR模型构建
基于Hugging Face transformers库，支持配置化
"""

import torch
import torch.nn as nn
from transformers import DetrForObjectDetection, DetrConfig


class DETRModel(nn.Module):
    """
    DETR模型封装
    支持从预训练权重加载并fine-tune到自定义类别数
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，包含模型参数
        """
        super().__init__()
        self.config = config
        model_config = config['model']
        
        # 获取类别数（+1 for background）
        num_classes = config['dataset']['num_classes']
        
        # 构建DETR配置
        if model_config.get('pretrained', True):
            # 从预训练模型加载
            print(f"🔄 加载预训练DETR模型: {model_config['name']}")
            self.model = DetrForObjectDetection.from_pretrained(
                f"facebook/{model_config['name']}",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,  # 允许类别数不匹配
            )
            print(f"✅ 预训练模型加载成功，已调整为 {num_classes} 个类别")
        else:
            # 从头训练
            print(f"🔨 从头构建DETR模型")
            detr_config = DetrConfig(
                num_labels=num_classes,
                num_queries=model_config.get('num_queries', 100),
                d_model=model_config.get('hidden_dim', 256),
                encoder_attention_heads=model_config.get('nheads', 8),
                decoder_attention_heads=model_config.get('nheads', 8),
                encoder_layers=model_config.get('num_encoder_layers', 6),
                decoder_layers=model_config.get('num_decoder_layers', 6),
                encoder_ffn_dim=model_config.get('dim_feedforward', 2048),
                decoder_ffn_dim=model_config.get('dim_feedforward', 2048),
                dropout=model_config.get('dropout', 0.1),
            )
            self.model = DetrForObjectDetection(detr_config)
            print(f"✅ 模型创建成功：{num_classes} 个类别")
        
        # 设置损失权重
        self.class_loss_coef = model_config['loss_weights']['class_loss_coef']
        self.bbox_loss_coef = model_config['loss_weights']['bbox_loss_coef']
        self.giou_loss_coef = model_config['loss_weights']['giou_loss_coef']
        
    def forward(self, images: torch.Tensor, targets: list = None):
        """
        前向传播
        
        Args:
            images: [B, 3, H, W] tensor
            targets: List[Dict] 包含 'boxes', 'labels' 等
        
        Returns:
            如果targets不为None，返回loss dict
            否则返回预测结果
        """
        if targets is not None:
            # 训练模式：计算loss
            # transformers的DETR需要特定格式的labels
            labels = []
            for t in targets:
                # 转换boxes格式: xyxy -> cxcywh (相对坐标)
                boxes = t['boxes'].clone()
                img_h, img_w = t['size']
                
                # xyxy -> cxcywh
                boxes_cxcywh = torch.zeros_like(boxes)
                boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
                boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
                boxes_cxcywh[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
                boxes_cxcywh[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
                
                # 归一化到[0, 1]
                boxes_cxcywh[:, [0, 2]] /= img_w
                boxes_cxcywh[:, [1, 3]] /= img_h
                
                labels.append({
                    'class_labels': t['labels'],
                    'boxes': boxes_cxcywh,
                })
            
            outputs = self.model(pixel_values=images, labels=labels)
            return outputs
        else:
            # 推理模式
            outputs = self.model(pixel_values=images)
            return outputs


def build_detr_model(config: dict) -> nn.Module:
    """
    构建DETR模型
    
    Args:
        config: 配置字典
    
    Returns:
        DETRModel实例
    """
    model = DETRModel(config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 模型统计:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    return model
