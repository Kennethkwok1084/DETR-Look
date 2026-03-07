#!/usr/bin/env python3
"""
本地图像处理器
为 Deformable DETR 提供与 HuggingFace 兼容的接口，无需下载预训练权重
"""

import torch
import torchvision.transforms as T
from PIL import Image
from typing import List, Dict, Union, Optional
import numpy as np


class LocalDeformableDetrImageProcessor:
    """
    本地 Deformable DETR 图像处理器
    提供与 HuggingFace DeformableDetrImageProcessor 兼容的接口
    """
    
    def __init__(
        self,
        size: Dict[str, int] = None,
        do_resize: bool = True,
        do_normalize: bool = True,
        image_mean: List[float] = None,
        image_std: List[float] = None,
    ):
        """
        Args:
            size: 目标尺寸，格式 {'height': H, 'width': W}，默认 800x1333
            do_resize: 是否调整大小
            do_normalize: 是否归一化
            image_mean: 归一化均值，默认 ImageNet 均值
            image_std: 归一化标准差，默认 ImageNet 标准差
        """
        self.size = size or {'height': 800, 'width': 1333}
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        
        # 构建转换流程
        transforms = []
        
        if do_resize:
            transforms.append(T.Resize((self.size['height'], self.size['width'])))
        
        transforms.append(T.ToTensor())
        
        if do_normalize:
            transforms.append(T.Normalize(mean=self.image_mean, std=self.image_std))
        
        self.transform = T.Compose(transforms)
    
    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: str = 'pt',
    ) -> Dict[str, torch.Tensor]:
        """
        处理图像
        
        Args:
            images: PIL 图像或图像列表
            return_tensors: 返回格式，仅支持 'pt'
        
        Returns:
            字典，包含:
                - pixel_values: 图像张量 (batch_size, 3, H, W)
                - pixel_mask: 掩码张量 (batch_size, H, W)
        """
        if not isinstance(images, list):
            images = [images]
        
        # 处理每张图像
        pixel_values = []
        pixel_masks = []
        
        for img in images:
            # 转换为 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 应用转换
            pixel_value = self.transform(img)
            pixel_values.append(pixel_value)
            
            # 创建全 True 的 mask（表示所有像素都有效）
            h, w = pixel_value.shape[-2:]
            pixel_mask = torch.ones((h, w), dtype=torch.bool)
            pixel_masks.append(pixel_mask)
        
        # 堆叠为 batch
        pixel_values = torch.stack(pixel_values, dim=0)
        pixel_masks = torch.stack(pixel_masks, dim=0)
        
        return {
            'pixel_values': pixel_values,
            'pixel_mask': pixel_masks,
        }
    
    def post_process_object_detection(
        self,
        outputs,
        threshold: float = 0.5,
        target_sizes: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        后处理目标检测结果
        兼容 HuggingFace 接口，内部调用官方后处理函数
        
        Args:
            outputs: 模型输出（官方格式或 HF 格式）
            threshold: 置信度阈值
            target_sizes: 目标尺寸 (batch_size, 2) [height, width]
        
        Returns:
            检测结果列表，每个元素包含 scores, labels, boxes
        """
        from models.deformable_detr_model import post_process_deformable_detr
        
        # 如果是训练输出格式，提取预测字段
        if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
            outputs_dict = {
                'pred_logits': outputs.logits,
                'pred_boxes': outputs.pred_boxes,
            }
        else:
            outputs_dict = outputs
        
        # 调用官方后处理
        return post_process_deformable_detr(outputs_dict, target_sizes, threshold)


def build_local_image_processor(config: dict):
    """
    构建本地图像处理器
    
    Args:
        config: 配置字典
    
    Returns:
        LocalDeformableDetrImageProcessor 实例
    """
    model_config = config.get('model', {})
    
    # 从配置中读取图像处理参数
    size = {
        'height': model_config.get('image_height', 800),
        'width': model_config.get('image_width', 1333),
    }
    
    return LocalDeformableDetrImageProcessor(
        size=size,
        do_resize=True,
        do_normalize=True,
    )
