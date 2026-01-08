"""
DETR 和 Deformable DETR 模型封装
提供统一的模型构建接口
"""

from .detr_model import build_detr_model
from .deformable_detr_model import build_deformable_detr_model


def build_model(config: dict):
    """
    统一模型构建接口
    根据配置中的 model.type 选择对应的模型
    
    Args:
        config: 配置字典
    
    Returns:
        模型实例
    
    Raises:
        ValueError: 不支持的模型类型
    """
    model_type = config['model'].get('type', 'detr').lower()
    
    if model_type == 'detr':
        print(f"📦 构建 DETR 模型...")
        return build_detr_model(config)
    elif model_type == 'deformable_detr' or model_type == 'deformable-detr':
        print(f"📦 构建 Deformable DETR 模型...")
        return build_deformable_detr_model(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: detr, deformable_detr")


def build_image_processor(config: dict):
    """
    统一图像处理器构建接口
    根据配置中的 model.type 选择对应的处理器
    
    Args:
        config: 配置字典
    
    Returns:
        图像处理器实例
    """
    from transformers import DetrImageProcessor, DeformableDetrImageProcessor
    
    model_type = config['model'].get('type', 'detr').lower()
    model_name = config['model']['name']
    
    # 处理模型名称前缀
    if '/' not in model_name:
        # 如果没有包含 /，根据模型类型添加默认前缀
        if model_type == 'detr':
            model_name = f"facebook/{model_name}"
        elif model_type == 'deformable_detr' or model_type == 'deformable-detr':
            model_name = f"SenseTime/{model_name}"
    
    if model_type == 'detr':
        print(f"🖼️  加载 DETR 图像处理器: {model_name}")
        return DetrImageProcessor.from_pretrained(model_name)
    elif model_type == 'deformable_detr' or model_type == 'deformable-detr':
        print(f"🖼️  加载 Deformable DETR 图像处理器: {model_name}")
        return DeformableDetrImageProcessor.from_pretrained(model_name)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


__all__ = [
    'build_detr_model',
    'build_deformable_detr_model', 
    'build_model',
    'build_image_processor',
]
