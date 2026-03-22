#!/usr/bin/env python3
"""
Deformable DETR 数据集适配器
生成官方格式的 NestedTensor 和 targets
"""

import sys
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import numpy as np

# === 模块缓存：避免 sys.path 污染 ===
_official_transforms_cache = {}

def _import_official_transforms():
    """隔离导入官方 transforms，不污染 sys.path 和 sys.modules"""
    if _official_transforms_cache:
        return _official_transforms_cache
    
    _original_sys_path = sys.path.copy()
    _third_party_path = Path(__file__).parent.parent / "third_party" / "deformable_detr"
    sys.path.insert(0, str(_third_party_path))
    
    try:
        import datasets.transforms as T
        from util.misc import collate_fn, NestedTensor
        
        _official_transforms_cache.update({
            'T': T,
            'collate_fn': collate_fn,
            'NestedTensor': NestedTensor,
        })
    finally:
        # 恢复 sys.path
        sys.path[:] = _original_sys_path
        
        # 关键：保留 datasets.* 和 util.* 模块在 sys.modules
        # 原因：
        # 1. DataLoader 在 Windows/spawn 模式下会 pickle 序列化 transforms/collate_fn
        # 2. worker 进程反序列化时需要 import datasets.transforms 和 util.misc
        # 3. 如果清理了这些模块，会触发 ModuleNotFoundError
        # 
        # 权衡：
        # - 保留模块：支持多进程 DataLoader（Windows/macOS spawn 模式）
        # - 污染风险：后续 import datasets 可能拿到第三方版本
        # - 实际影响：Deformable 训练时基本不会同时使用 HF datasets
        # 
        # 结论：优先保证 DataLoader 正常工作，接受有限的模块污染
        # （sys.path 已恢复，只是 sys.modules 中保留了已加载的模块）
    
    return _official_transforms_cache

# 导入并缓存
_transforms_modules = _import_official_transforms()
T = _transforms_modules['T']
collate_fn = _transforms_modules['collate_fn']
NestedTensor = _transforms_modules['NestedTensor']


def make_deformable_transforms(image_set, config):
    """
    创建官方风格的数据增强
    
    Args:
        image_set: 'train' 或 'val'
        config: 配置字典
    
    Returns:
        transforms 组合
    """
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    aug_config = config.get('dataset', {}).get('augmentation', {})
    scales = aug_config.get('train_scales', [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])
    crop_resize_scales = aug_config.get('crop_resize_scales', [400, 500, 600])
    crop_min_size = int(aug_config.get('crop_min_size', 384))
    crop_max_size = int(aug_config.get('crop_max_size', 600))
    train_max_size = int(aug_config.get('train_max_size', 1333))
    eval_short_edge = int(aug_config.get('eval_short_edge', 800))
    eval_max_size = int(aug_config.get('eval_max_size', train_max_size))

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=train_max_size),
                T.Compose([
                    T.RandomResize(crop_resize_scales),
                    T.RandomSizeCrop(crop_min_size, crop_max_size),
                    T.RandomResize(scales, max_size=train_max_size),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([eval_short_edge], max_size=eval_max_size),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class DeformableCOCODataset(torch.utils.data.Dataset):
    """
    COCO 数据集适配器（官方格式）
    生成 NestedTensor 和 targets，与官方 Deformable DETR 兼容
    """
    
    def __init__(self, img_folder, ann_file, transforms=None, return_masks=False, image_id_offset=0, dataset_name=None):
        """
        Args:
            img_folder: 图像文件夹路径
            ann_file: COCO 标注文件路径
            transforms: 数据增强
            return_masks: 是否返回分割 mask
        """
        from pycocotools.coco import COCO
        
        self.img_folder = Path(img_folder)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self._transforms = transforms
        self.return_masks = return_masks
        self.image_id_offset = int(image_id_offset)
        self.dataset_name = dataset_name or Path(img_folder).name
        self.max_image_id = max(self.ids) if self.ids else -1
        
        # 获取类别映射
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_continuous = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.continuous_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_continuous.items()}
        categories = sorted(self.coco.dataset.get('categories', []), key=lambda cat: cat.get('id', -1))
        self.categories_signature = tuple(
            (cat.get('id'), cat.get('name'))
            for cat in categories
        )
        
        print(f"✅ 加载 Deformable COCO 数据集:")
        print(f"   - 图像数量: {len(self.ids)}")
        print(f"   - 类别数量: {len(self.cat_ids)}")
        print(f"   - 类别 ID: {self.cat_ids}")
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        返回官方格式:
        - img: PIL Image 或 Tensor (经过 transform)
        - target: dict
            - boxes: [N, 4] 归一化的 cxcywh
            - labels: [N] 连续的类别索引 [0, num_classes-1]
            - image_id: tensor
            - area: [N]
            - iscrowd: [N]
            - orig_size: [2] (H, W)
            - size: [2] (H, W)
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_folder / img_info['file_name']
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # 加载标注
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # 过滤 crowd
        anns = [obj for obj in anns if obj.get('iscrowd', 0) == 0]
        
        # 转换 boxes: [x, y, w, h] -> [x1, y1, x2, y2]
        boxes = []
        labels = []
        areas = []
        iscrowds = []
        
        for obj in anns:
            xmin, ymin, box_w, box_h = obj['bbox']
            xmax = xmin + box_w
            ymax = ymin + box_h
            
            # 裁剪到图像边界
            xmin = max(0, min(xmin, w))
            ymin = max(0, min(ymin, h))
            xmax = max(0, min(xmax, w))
            ymax = max(0, min(ymax, h))
            
            # 过滤无效框
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                # 映射到连续索引
                cat_id = obj['category_id']
                labels.append(self.cat_id_to_continuous[cat_id])
                areas.append(obj.get('area', box_w * box_h))
                iscrowds.append(obj.get('iscrowd', 0))
        
        # 转为 tensor
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowds = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowds = torch.as_tensor(iscrowds, dtype=torch.int64)
        
        # 构建 target
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id + self.image_id_offset])
        target["area"] = areas
        target["iscrowd"] = iscrowds
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        
        # 应用 transform
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target


def build_deformable_dataloader(config, image_set='train'):
    """
    构建 Deformable DETR 数据加载器（官方格式）
    
    Args:
        config: 配置字典
        image_set: 'train' 或 'val'
    
    Returns:
        DataLoader
    """
    dataset_config = config['dataset']
    train_config = config['training']
    val_config = config.get('validation', {})
    
    # 创建 transforms
    transforms = make_deformable_transforms(image_set, config)
    
    # 创建数据集
    datasets_list = []
    
    if 'datasets' in dataset_config:
        category_signature = None
        image_id_offset = 0
        for ds_conf in dataset_config['datasets']:
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
            
            img_folder = ds_conf.get(f'{image_set}_img')
            if not img_folder:
                img_folder = str(root / 'images' / image_set)
            elif not Path(img_folder).is_absolute():
                img_folder = str(root / img_folder)
            
            ds = DeformableCOCODataset(
                img_folder=img_folder,
                ann_file=str(ann_file),
                transforms=transforms,
                return_masks=False,
                image_id_offset=image_id_offset,
                dataset_name=ds_conf.get('name'),
            )
            if category_signature is None:
                category_signature = ds.categories_signature
                merged_mapping = dict(ds.continuous_to_cat_id)
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
        dataset.continuous_to_cat_id = merged_mapping
        print(f"🔗 已合并 {len(datasets_list)} 个数据集，总样本数: {len(dataset)}")
    else:
        # 单一数据集模式
        if image_set == 'train':
            # 支持两种配置方式：train_img 或 root_dir + images/train
            img_folder = dataset_config.get('train_img')
            if not img_folder:
                root_dir = dataset_config.get('root_dir', 'data')
                img_folder = str(Path(root_dir) / 'images' / 'train')
            elif not Path(img_folder).is_absolute() and dataset_config.get('root_dir'):
                img_folder = str(Path(dataset_config['root_dir']) / img_folder)

            ann_file = dataset_config['train_ann']
            ann_file = Path(ann_file)
            if not ann_file.is_absolute() and 'root_dir' in dataset_config:
                ann_file = Path(dataset_config['root_dir']) / ann_file
            ann_file = str(ann_file)
        else:
            img_folder = dataset_config.get('val_img')
            if not img_folder:
                root_dir = dataset_config.get('root_dir', 'data')
                img_folder = str(Path(root_dir) / 'images' / 'val')
            elif not Path(img_folder).is_absolute() and dataset_config.get('root_dir'):
                img_folder = str(Path(dataset_config['root_dir']) / img_folder)

            ann_file = dataset_config['val_ann']
            ann_file = Path(ann_file)
            if not ann_file.is_absolute() and 'root_dir' in dataset_config:
                ann_file = Path(dataset_config['root_dir']) / ann_file
            ann_file = str(ann_file)

        dataset = DeformableCOCODataset(
            img_folder=img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=False,
            image_id_offset=0,
            dataset_name=dataset_config.get('name'),
        )

    subset_size = train_config.get('subset_size')
    overfit_mode = train_config.get('overfit', False)
    if image_set == 'train' and subset_size:
        subset_seed = train_config.get('subset_seed', 42)
        random.seed(subset_seed)
        sample_size = min(int(subset_size), len(dataset))

        if overfit_mode:
            indices = list(range(sample_size))
        else:
            indices = random.sample(range(len(dataset)), sample_size)

        dataset = Subset(dataset, indices)
        print(f"📌 Deformable {image_set} 子集: {len(dataset)} / {sample_size} (seed={subset_seed})")
    
    # 创建 DataLoader
    if image_set == 'train':
        batch_size = train_config['batch_size']
        num_workers = train_config.get('num_workers', 4)
        shuffle = not overfit_mode
    else:
        batch_size = val_config.get('batch_size', train_config['batch_size'])
        num_workers = val_config.get('num_workers', train_config.get('num_workers', 4))
        shuffle = False
    
    prefetch_factor = None
    if num_workers > 0:
        prefetch_factor = (
            train_config.get('prefetch_factor', 2)
            if image_set == 'train'
            else val_config.get('prefetch_factor', train_config.get('prefetch_factor', 2))
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  # 官方 collate_fn，生成 NestedTensor
        pin_memory=True,
        drop_last=(image_set == 'train' and not overfit_mode),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    print(f"✅ Deformable DataLoader 创建成功 ({image_set})")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Workers: {num_workers}")
    
    return dataloader, dataset
