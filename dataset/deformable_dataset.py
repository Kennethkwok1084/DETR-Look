#!/usr/bin/env python3
"""
Deformable DETR 数据集适配器
生成官方格式的 NestedTensor 和 targets
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
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

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


class DeformableCOCODataset(torch.utils.data.Dataset):
    """
    COCO 数据集适配器（官方格式）
    生成 NestedTensor 和 targets，与官方 Deformable DETR 兼容
    """
    
    def __init__(self, img_folder, ann_file, transforms=None, return_masks=False):
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
        
        # 获取类别映射
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_continuous = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.continuous_to_cat_id = {idx: cat_id for cat_id, idx in self.cat_id_to_continuous.items()}
        
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
        target["image_id"] = torch.tensor([img_id])
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
    
    if image_set == 'train':
        # 支持两种配置方式：train_img 或 root_dir + images/train
        img_folder = dataset_config.get('train_img')
        if not img_folder:
            root_dir = dataset_config.get('root_dir', 'data')
            img_folder = str(Path(root_dir) / 'images' / 'train')
        
        ann_file = dataset_config['train_ann']
        # 如果 ann_file 是相对路径，拼接 root_dir
        ann_file = Path(ann_file)
        if not ann_file.is_absolute() and 'root_dir' in dataset_config:
            ann_file = Path(dataset_config['root_dir']) / ann_file
        ann_file = str(ann_file)
    else:
        img_folder = dataset_config.get('val_img')
        if not img_folder:
            root_dir = dataset_config.get('root_dir', 'data')
            img_folder = str(Path(root_dir) / 'images' / 'val')
        
        ann_file = dataset_config['val_ann']
        ann_file = Path(ann_file)
        if not ann_file.is_absolute() and 'root_dir' in dataset_config:
            ann_file = Path(dataset_config['root_dir']) / ann_file
        ann_file = str(ann_file)
    
    # 创建 transforms
    transforms = make_deformable_transforms(image_set, config)
    
    # 创建数据集
    dataset = DeformableCOCODataset(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=transforms,
        return_masks=False
    )
    
    # 创建 DataLoader
    batch_size = train_config['batch_size'] if image_set == 'train' else 1
    num_workers = train_config.get('num_workers', 4)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,  # 官方 collate_fn，生成 NestedTensor
        pin_memory=True,
        drop_last=(image_set == 'train')
    )
    
    print(f"✅ Deformable DataLoader 创建成功 ({image_set})")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Workers: {num_workers}")
    
    return dataloader, dataset
