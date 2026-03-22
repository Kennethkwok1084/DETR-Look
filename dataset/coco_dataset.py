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
        image_id_offset: int = 0,
        dataset_name: Optional[str] = None,
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
        self.image_id_offset = int(image_id_offset)
        self.dataset_name = dataset_name or Path(img_folder).name
        self.max_image_id = max(self.ids) if self.ids else -1
        categories = sorted(self.coco.dataset.get('categories', []), key=lambda cat: cat.get('id', -1))
        self.categories_signature = tuple(
            (cat.get('id'), cat.get('name'))
            for cat in categories
        )
        
        # 检查 transforms 兼容性（仅警告一次，避免日志爆炸）
        if self.transforms is not None:
            import warnings
            # 检测不兼容的 transform 类型
            incompatible = []
            if hasattr(transforms, 'transforms'):  # Compose
                for t in transforms.transforms:
                    t_name = type(t).__name__
                    if t_name in ['ToTensor', 'RandomHorizontalFlip', 'RandomVerticalFlip', 
                                  'RandomCrop', 'CenterCrop', 'RandomResizedCrop', 'RandomRotation']:
                        incompatible.append(t_name)
            elif type(transforms).__name__ in ['ToTensor', 'RandomHorizontalFlip', 'RandomCrop']:
                incompatible.append(type(transforms).__name__)
            
            if incompatible:
                warnings.warn(
                    f"⚠️  检测到不兼容的 transforms: {incompatible}。\n"
                    "  - ToTensor 会打断 DetrImageProcessor（期望 PIL/ndarray）\n"
                    "  - 几何变换（翻转/裁剪）的 bbox 不会同步，会导致训练失真。\n"
                    "  建议：仅使用 ColorJitter/GaussianBlur 等颜色增强，或改用 albumentations。",
                    UserWarning,
                    stacklevel=2
                )
            else:
                warnings.warn(
                    "当前 transforms 仅作用于图像，bbox 不会同步变换。\n"
                    "请确保仅包含颜色增强（ColorJitter/GaussianBlur），避免几何变换。",
                    UserWarning,
                    stacklevel=2
                )
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        返回单个样本（原始格式，供DetrImageProcessor处理）
        
        Returns:
            image: PIL.Image (RGB，未归一化)
            target: COCO格式标注字典 {
                'image_id': int,
                'annotations': List[{...}],
                'orig_size': [height, width],  # 原图尺寸（评估时必须）
                'size': [height, width],       # 同 orig_size（兼容性）
            }
        """
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 加载图像（PIL格式，不转tensor）
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_folder / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # 获取原图尺寸（height, width）
        orig_width, orig_height = image.size
        orig_size = [orig_height, orig_width]
        
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
            'image_id': img_id + self.image_id_offset,
            'annotations': annotations,
            'orig_size': orig_size,  # 评估时用于计算 target_sizes
            'size': orig_size,       # 兼容性字段
        }
        
        # 应用数据增强（如果提供）
        # 注意：已在 __init__ 中检查兼容性，此处直接应用
        if self.transforms is not None:
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


def make_collate_fn_with_processor(image_processor):
    """
    创建带预处理的collate_fn（在worker进程中并行处理）
    
    注意：保留原始 targets 以确保 image_id 等元数据存在
    
    Args:
        image_processor: DetrImageProcessor实例
    
    Returns:
        collate_fn: 返回已处理的tensor和labels + 原始 targets
    """
    def collate_with_processor(batch: List[Tuple[Image.Image, Dict]]):
        images, targets = zip(*batch)
        
        # 在worker进程中预处理（并行）
        encoding = image_processor(
            images=list(images),
            annotations=list(targets),
            return_tensors='pt'
        )
        
        # 保留原始 targets（包含 image_id）
        # HF processor 的 labels 可能不包含 image_id
        encoding['targets'] = list(targets)
        
        return encoding
    
    return collate_with_processor


def build_dataloader(
    config: dict,
    image_set: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    image_processor: Optional[Any] = None,  # DetrImageProcessor实例
) -> DataLoader:
    """
    构建DataLoader，支持子集采样和过拟合模式
    
    Args:
        config: 配置字典
        image_set: 'train' 或 'val'
        batch_size: 批大小（None则从config读取）
        num_workers: 工作进程数（None则从config读取）
        shuffle: 是否打乱（None则train=True, val=False）
        image_processor: DetrImageProcessor实例（如提供则在worker中预处理）
    
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
    
    overfit_mode = config['training'].get('overfit', False)

    # 构建数据集
    datasets_list = []
    image_id_offset = 0
    
    # 检查是否配置了多数据集 (datasets列表)
    if 'datasets' in config['dataset']:
        category_signature = None
        for ds_conf in config['dataset']['datasets']:
            ann_key = f'{image_set}_ann'
            if ann_key not in ds_conf:
                if image_set == 'train':
                    raise KeyError(f"多数据集配置缺少必需字段: {ann_key}")
                print(f"⚠️  跳过未配置 {ann_key} 的数据集: {ds_conf.get('name', 'unnamed')}")
                continue

            root = Path(ds_conf['root_dir'])
            ann_file = Path(ds_conf[ann_key])
            if not ann_file.is_absolute():
                ann_file = root / ann_file
            
            # 支持每个数据集单独配置 img 目录
            img_folder = ds_conf.get(f'{image_set}_img')
            if not img_folder:
                img_folder = root / 'images' / image_set
            elif not Path(img_folder).is_absolute():
                img_folder = root / img_folder
            
            # 构建 transforms（过拟合模式下强制为 None）
            if overfit_mode and image_set == 'train':
                transforms = None
            else:
                transforms = make_transforms(image_set, config)
            
            ds = CocoDetectionDataset(
                img_folder=str(img_folder),
                ann_file=str(ann_file),
                transforms=transforms,
                image_id_offset=image_id_offset,
                dataset_name=ds_conf.get('name'),
            )
            if category_signature is None:
                category_signature = ds.categories_signature
            elif ds.categories_signature != category_signature:
                raise ValueError(
                    f"数据集类别定义不一致: {ds_conf.get('name', 'unnamed')} 与前序数据集不匹配，"
                    "请先统一 category_id/name 映射后再混训"
                )
            datasets_list.append(ds)
            print(
                f"📦 已添加数据集: {ds_conf.get('name', 'unnamed')} "
                f"({len(ds)} 样本, image_id_offset={image_id_offset})"
            )
            image_id_offset += ds.max_image_id + 1

        if not datasets_list:
            raise ValueError(f"未找到可用于 {image_set} 的数据集配置")
            
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset(datasets_list)
        print(f"🔗 已合并 {len(datasets_list)} 个数据集，总样本数: {len(dataset)}")
    else:
        # 单一数据集模式 (兼容旧配置)
        root = Path(config['dataset']['root_dir'])
        ann_file = root / config['dataset'][f'{image_set}_ann']
        img_folder = root / 'images' / image_set
        
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
            image_id_offset=0,
            dataset_name=config['dataset'].get('name'),
        )
    
    # 子集采样逻辑（用于快速验证或小样本过拟合）
    subset_size = config['training'].get('subset_size')
    
    if subset_size and image_set == 'train':
        # 固定随机种子以保证子集可复现
        subset_seed = config['training'].get('subset_seed', 42)
        random.seed(subset_seed)
        
        # 是否过滤空标注样本（默认仅过拟合模式下过滤）
        # filter_empty: True=强制有标注, False=允许空标注（保持原始分布）
        # None (或 null) 表示 "auto"：在 overfit 模式下过滤，否则不过滤
        filter_empty = config['training'].get('subset_filter_empty', overfit_mode)
        if filter_empty is None:  # 处理 subset_filter_empty: null 的情况
            filter_empty = overfit_mode
        
        if filter_empty:
            # 对于 ConcatDataset，需要遍历底层数据集收集 valid_indices
            if hasattr(dataset, 'datasets'):
                valid_indices = []
                offset = 0
                for ds in dataset.datasets:
                    if hasattr(ds, 'coco') and hasattr(ds, 'ids'):
                        ann_list = ds.coco.dataset.get('annotations', [])
                        img_ids_with_ann = {ann['image_id'] for ann in ann_list if 'image_id' in ann}
                        for i, img_id in enumerate(ds.ids):
                            if img_id in img_ids_with_ann:
                                valid_indices.append(offset + i)
                    else:
                        for i in range(len(ds)):
                            _, target = ds[i]
                            if target.get('annotations') and len(target['annotations']) > 0:
                                valid_indices.append(offset + i)
                    offset += len(ds)
                print(f"🚀 跨数据集过滤：{len(dataset)} → {len(valid_indices)} 个有效样本")
            else:
                # 优先使用 COCO 元数据以避免逐样本加载图像
                if hasattr(dataset, 'coco') and hasattr(dataset, 'ids'):
                    # 从 COCO 标注中收集所有有标注的 image_id
                    ann_list = dataset.coco.dataset.get('annotations', [])
                    img_ids_with_ann = {ann['image_id'] for ann in ann_list if 'image_id' in ann}
                    # 根据 dataset.ids 中的 image_id 映射回数据集索引
                    valid_indices = [
                        idx for idx, img_id in enumerate(dataset.ids)
                        if img_id in img_ids_with_ann
                    ]
                    print(f"🚀 使用 COCO API 快速过滤：{len(dataset)} → {len(valid_indices)} 个有效样本")
                else:
                    # 回退到逐样本检查逻辑（可能较慢）
                    print("⚠️  未检测到 COCO API，使用逐样本检查（可能较慢）...")
                    valid_indices = []
                    for idx in range(len(dataset)):
                        _, target = dataset[idx]
                        if target.get('annotations') and len(target['annotations']) > 0:
                            valid_indices.append(idx)
                    print(f"🔍 已过滤空标注样本：{len(dataset)} → {len(valid_indices)} 个有效样本")
            
            if len(valid_indices) == 0:
                raise ValueError(f"数据集中没有找到有标注的样本，无法进行训练")
            
            pool_indices = valid_indices
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
    
    # 选择collate_fn（如果提供processor则在worker中预处理）
    if image_processor is not None:
        _collate_fn = make_collate_fn_with_processor(image_processor)
        print(f"✅ 启用worker中预处理（加速训练）")
    else:
        _collate_fn = collate_fn
    
    # 构建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,  # 保持worker进程复用
        prefetch_factor=8 if num_workers > 0 else None,  # 每个worker预取8个batch（大缓冲）
    )
    
    print(f"✅ {image_set.upper()} DataLoader 创建成功:")
    print(f"   数据集大小: {len(dataset)}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Workers: {num_workers}")
    print(f"   Shuffle: {shuffle}")
    
    return dataloader
