"""
数据增强参考实现

如需在训练时添加数据增强，可按以下方式修改coco_dataset.py：

1. 在make_transforms中返回transforms pipeline
2. 在__getitem__中对PIL图像应用transforms
3. transforms后的PIL图像传给DetrImageProcessor

关键点：
- DetrImageProcessor会处理resize/pad/normalize，所以transforms中不要包含这些
- bbox的转换由processor处理，transforms中只需要图像增强
- 使用torchvision.transforms中对PIL图像友好的操作
"""

from torchvision import transforms
from typing import Any


def make_transforms_with_augmentation(image_set: str, config: dict) -> Any:
    """
    带数据增强的transforms构建示例
    
    可以复制这个函数到coco_dataset.py中替换原来的make_transforms
    """
    if image_set == 'train':
        # 训练时的数据增强
        aug_config = config.get('dataset', {}).get('augmentation', {})
        
        transform_list = []
        
        # 随机水平翻转
        if aug_config.get('random_horizontal_flip', False):
            transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )
        
        # 颜色抖动
        if aug_config.get('color_jitter', False):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
        
        # 随机灰度化
        if aug_config.get('random_grayscale', False):
            transform_list.append(
                transforms.RandomGrayscale(p=0.1)
            )
        
        # 随机仿射变换（可选，需要同时转换bbox）
        # 注意：如果使用仿射/旋转等几何变换，需要同步更新annotations中的bbox
        # 建议使用albumentations库，它可以自动处理bbox转换
        
        if transform_list:
            return transforms.Compose(transform_list)
        else:
            return None
    else:
        # 验证/测试时不做增强
        return None


# ===================================================================
# 在coco_dataset.py的__getitem__中的使用方式：
# ===================================================================
"""
def __getitem__(self, idx: int) -> Tuple[Any, Dict]:
    # 1. 加载图像和标注
    img_id = self.ids[idx]
    img_info = self.coco.loadImgs(img_id)[0]
    image_path = self.root / self.img_folder / img_info['file_name']
    
    # 加载PIL图像
    image = Image.open(image_path).convert('RGB')
    
    # 2. 构建COCO格式target
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
    
    # 过滤并构建annotations
    annotations = []
    for ann in anns:
        if ann.get('iscrowd', 0) == 1:
            continue
        bbox = ann['bbox']  # [x, y, w, h]
        if bbox[2] > 0 and bbox[3] > 0:
            annotations.append({
                'bbox': bbox,
                'category_id': ann['category_id'],
                'area': ann.get('area', bbox[2] * bbox[3]),
                'iscrowd': 0,
            })
    
    target = {
        'image_id': img_id,
        'annotations': annotations,
    }
    
    # 3. 应用数据增强（如果有）
    if self._transforms is not None:
        # 注意：这里的transforms只做图像增强，不改变bbox
        # 如果需要几何变换（旋转、裁剪等），建议使用albumentations
        image = self._transforms(image)
    
    # 4. 返回PIL图像和COCO格式target
    # DetrImageProcessor会在训练循环中处理resize/pad/normalize和bbox转换
    return image, target
"""


# ===================================================================
# 使用albumentations进行几何变换的示例（推荐）
# ===================================================================
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

def make_transforms_with_albumentations(image_set: str, config: dict):
    '''使用albumentations的增强pipeline'''
    if image_set == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            # 注意：不要包含Resize和Normalize，由DetrImageProcessor处理
        ], bbox_params=A.BboxParams(
            format='coco',  # [x, y, w, h]
            label_fields=['category_ids'],
            min_visibility=0.3,  # 至少30%可见才保留
        ))
    else:
        return None

def __getitem__with_albumentations(self, idx: int):
    # ... 加载图像和标注 ...
    
    # 应用albumentations
    if self._transforms is not None:
        # 准备bbox和labels
        bboxes = [ann['bbox'] for ann in annotations]
        category_ids = [ann['category_id'] for ann in annotations]
        
        # 转为numpy数组用于albumentations
        import numpy as np
        image_np = np.array(image)
        
        # 应用增强
        transformed = self._transforms(
            image=image_np,
            bboxes=bboxes,
            category_ids=category_ids
        )
        
        # 转回PIL图像
        image = Image.fromarray(transformed['image'])
        
        # 更新annotations
        annotations = [
            {
                'bbox': list(bbox),
                'category_id': cat_id,
                'area': bbox[2] * bbox[3],
                'iscrowd': 0,
            }
            for bbox, cat_id in zip(transformed['bboxes'], transformed['category_ids'])
        ]
    
    target = {
        'image_id': img_id,
        'annotations': annotations,
    }
    
    return image, target
"""


# ===================================================================
# 配置文件示例（configs/detr_baseline.yaml）
# ===================================================================
"""
dataset:
  name: "bdd100k_traffic"
  root_dir: "data/traffic_coco/bdd100k_det"
  train_ann: "annotations/instances_train.json"
  val_ann: "annotations/instances_val.json"
  num_classes: 3
  
  # 数据增强配置
  augmentation:
    random_horizontal_flip: true
    color_jitter: true
    random_grayscale: false
    # 如果使用albumentations:
    # shift_scale_rotate: true
    # random_brightness_contrast: true
"""

print(__doc__)
