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
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        返回单个样本（原始格式，供DetrImageProcessor处理）
        
        Returns:
            image: PIL.Image (RGB，未归一化)
            target: COCO格式标注字典 {
                'image_id': int,
                'annotations': List[{
                    'bbox': [x, y, w, h],  # COCO格式：xywh像素坐标
                    'category_id': int,
                    'area': float,
                    'iscrowd': int,
                }]
            }
        """
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 加载图像（PIL格式，不转tensor）
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_folder / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 构建COCO格式标注（DetrImageProcessor期望的格式）
        annotations = []
        for ann in anns:
            annotations.append({
                'bbox': ann['bbox'],  # 保持COCO的[x, y, w, h]格式
                'category_id': ann['category_id'],
                'area': ann['area'],
                'iscrowd': ann.get('iscrowd', 0),
            })
        
        target = {
            'image_id': img_id,
            'annotations': annotations,
        }
        
        # 应用数据增强（如果提供）
        # ⚠️  警告：当前实现仅对图像应用增强，bbox 不会同步变换
        # 如需几何变换（翻转/裁剪/旋转），请使用支持 bbox 变换的库如 albumentations
        # 或确保 transforms 仅包含颜色增强（ColorJitter 等）
        if self.transforms is not None:
            import warnings
            warnings.warn(
                "当前数据增强仅作用于图像，bbox 不会同步变换。"
                "几何变换（翻转/裁剪）会导致标注错位。"
                "建议使用 albumentations 或仅使用颜色增强。",
                UserWarning
            )
            image = self.transforms(image)
        
        return image, target


def make_transforms(image_set: str, config: dict) -> Any:
    """
    构建数据增强pipeline
    
    注意：当前版本由DetrImageProcessor统一处理resize/pad/normalize，
    因此返回None。如需额外数据增强（如RandomHorizontalFlip、ColorJitter），
    可在此构建torchvision.transforms.Compose并在__getitem__中对PIL图像应用。
    
    参考实现：
    if image_set == 'train':
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ])
    
    Args:
        image_set: 'train' 或 'val'
        config: 配置字典
    
    Returns:
        None (当前不使用transforms，直接返回PIL图像)
    """
    # DetrImageProcessor会自动处理resize/pad/normalize
    # 如需额外增强可在此添加，在__getitem__中应用到PIL图像后再传给processor
    return None


def collate_fn(batch: List[Tuple[Image.Image, Dict]]) -> Tuple[List[Image.Image], List[Dict]]:
    """
    自定义collate函数，返回原始PIL图像和COCO标注
    供DetrImageProcessor批量处理
    
    Args:
        batch: [(PIL.Image, target_dict), ...] 列表
    
    Returns:
        images: List[PIL.Image]
        targets: List[Dict] - COCO格式标注
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
    构建DataLoader，支持子集采样和过拟合模式
    
    Args:
        config: 配置字典
        image_set: 'train' 或 'val'
        batch_size: 批大小（None则从config读取）
        num_workers: 工作进程数（None则从config读取）
        shuffle: 是否打乱（None则train=True, val=False）
    
    Returns:
        DataLoader
    """
    import random
    from torch.utils.data import Subset
    
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
    
    # 检查是否为过拟合模式（需要在 make_transforms 之前检查）
    overfit_mode = config['training'].get('overfit', False)
    
    # 构建 transforms（过拟合模式下强制为 None）
    if overfit_mode and image_set == 'train':
        transforms = None
        print("📌 过拟合模式：禁用数据增强（transforms=None）")
    else:
        transforms = make_transforms(image_set, config)
    
    dataset = CocoDetectionDataset(
        img_folder=str(img_folder),
        ann_file=str(ann_file),
        transforms=transforms,
    )
    
    # 子集采样逻辑（用于快速验证或小样本过拟合）
    subset_size = config['training'].get('subset_size')
    
    if subset_size and image_set == 'train':
        # 固定随机种子以保证子集可复现
        subset_seed = config['training'].get('subset_seed', 42)
        random.seed(subset_seed)
        
        # 是否过滤空标注样本（默认仅过拟合模式下过滤）
        # filter_empty: True=强制有标注, False=允许空标注（保持原始分布）
        filter_empty = config['training'].get('subset_filter_empty', overfit_mode)
        
        if filter_empty:
            # 筛选有标注的样本（过拟合测试必须有标注）
            valid_indices = []
            for idx in range(len(dataset)):
                _, target = dataset[idx]
                if target.get('annotations') and len(target['annotations']) > 0:
                    valid_indices.append(idx)
            
            if len(valid_indices) == 0:
                raise ValueError(f"数据集中没有找到有标注的样本，无法进行训练")
            
            pool_indices = valid_indices
            print(f"🔍 已过滤空标注样本：{len(dataset)} → {len(pool_indices)} 个有效样本")
        else:
            # 不过滤，使用全量样本池（保持原始分布）
            pool_indices = list(range(len(dataset)))
            print(f"📊 使用全量样本池（包含空标注）：{len(pool_indices)} 个样本")
        
        # 随机选择或顺序选择
        if overfit_mode:
            # 过拟合模式：选择前N个样本
            indices = pool_indices[:min(subset_size, len(pool_indices))]
            print(f"📌 过拟合模式：从 {len(pool_indices)} 个样本中选择前 {len(indices)} 个（固定种子={subset_seed}）")
        else:
            # 正常子集模式：随机采样
            sample_size = min(subset_size, len(pool_indices))
            indices = random.sample(pool_indices, sample_size)
            print(f"🎲 子集采样：从 {len(pool_indices)} 个样本中随机选择 {len(indices)} 个（种子={subset_seed}）")
        
        dataset = Subset(dataset, indices)
        
        # 过拟合模式下强制不打乱
        if overfit_mode:
            shuffle = False
            print(f"📌 过拟合模式：关闭打乱（transforms 已在前面禁用）")
    
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
