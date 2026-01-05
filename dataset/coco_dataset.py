#!/usr/bin/env python3
"""
COCO格式数据集加载器
支持DETR训练所需的数据增强和格式转换
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset


class CocoDetectionDataset(Dataset):
    """
    COCO格式目标检测数据集
    
    返回格式符合DETR要求：
    - image: [3, H, W] 的 tensor
    - target: dict包含 'boxes', 'labels', 'image_id' 等
    """
    
    def __init__(
        self,
        img_folder: str,
        ann_file: str,
        transforms: Optional[Any] = None,
        return_masks: bool = False,
    ):
        """
        Args:
            img_folder: 图像文件夹路径
            ann_file: COCO格式标注文件路径
            transforms: 数据增强pipeline
            return_masks: 是否返回分割mask（本项目不需要）
        """
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.return_masks = return_masks
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        返回单个样本
        
        Returns:
            image: [3, H, W] tensor
            target: {
                'boxes': [N, 4] tensor (xyxy格式，归一化到[0,1])
                'labels': [N] tensor
                'image_id': tensor
                'area': [N] tensor
                'iscrowd': [N] tensor
                'orig_size': [2] tensor (H, W)
                'size': [2] tensor (H, W)
            }
        """
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_folder / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 解析标注
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO格式: [x, y, w, h]
            x, y, w, h = ann['bbox']
            # 转换为 [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))
        
        # 转为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id])
        
        # 构建target字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': iscrowd,
            'orig_size': torch.as_tensor([int(img_info['height']), int(img_info['width'])]),
            'size': torch.as_tensor([int(img_info['height']), int(img_info['width'])]),
        }
        
        # 应用数据增强
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target


def make_transforms(image_set: str, config: dict) -> Any:
    """
    构建数据增强pipeline
    
    Args:
        image_set: 'train' 或 'val'
        config: 配置字典
    
    Returns:
        transform函数
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(
            config['dataset']['augmentation']['normalize']['mean'],
            config['dataset']['augmentation']['normalize']['std']
        )
    ])
    
    # 简化版transforms：只做归一化
    # 因为DETR的transforms需要特殊处理（同时变换image和boxes）
    # 这里先实现最简版本
    return normalize


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    自定义collate函数，因为target是dict且每张图的目标数量不同
    对于可变尺寸的图像，不使用stack，直接返回列表
    
    Args:
        batch: [(image, target), ...] 列表
    
    Returns:
        images: List[Tensor] - 列表形式，支持可变尺寸
        targets: List[Dict]
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloader(
    config: dict,
    image_set: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    构建DataLoader
    
    Args:
        config: 配置字典
        image_set: 'train' 或 'val'
        batch_size: 批大小（None则从config读取）
        num_workers: 工作进程数（None则从config读取）
        shuffle: 是否打乱（None则train=True, val=False）
    
    Returns:
        DataLoader
    """
    # 确定参数
    if batch_size is None:
        batch_size = config['training']['batch_size'] if image_set == 'train' else config['validation']['batch_size']
    
    if num_workers is None:
        num_workers = config['training']['num_workers'] if image_set == 'train' else config['validation']['num_workers']
    
    if shuffle is None:
        shuffle = (image_set == 'train')
    
    # 构建数据集
    root = Path(config['dataset']['root_dir'])
    ann_file = root / config['dataset'][f'{image_set}_ann']
    img_folder = root / 'images' / image_set
    
    transforms = make_transforms(image_set, config)
    
    dataset = CocoDetectionDataset(
        img_folder=str(img_folder),
        ann_file=str(ann_file),
        transforms=transforms,
    )
    
    # 构建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    print(f"✅ {image_set.upper()} DataLoader 创建成功:")
    print(f"   数据集大小: {len(dataset)}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print(f"   Shuffle: {shuffle}")
    
    return dataloader
