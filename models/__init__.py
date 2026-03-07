"""
DETR 和 Deformable DETR 模型封装
提供统一的模型构建接口
"""

from .detr_model import build_detr_model


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
        print(f"📦 构建 DETR 模型（HuggingFace 数据流）...")
        return build_detr_model(config)
    elif model_type == 'deformable_detr' or model_type == 'deformable-detr':
        # 延迟导入，避免在不使用时加载 third_party
        from .deformable_detr_model import build_deformable_detr_model
        print(f"📦 构建 Deformable DETR 模型（官方数据流）...")
        return build_deformable_detr_model(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: detr, deformable_detr")


def build_image_processor(config: dict):
    """
    统一图像处理器构建接口
    根据配置中的 model.type 选择对应的处理器
    
    注意：
    - DETR: 使用 HuggingFace DetrImageProcessor
    - Deformable DETR: 不需要独立的处理器（使用官方数据流）
    
    Args:
        config: 配置字典
    
    Returns:
        图像处理器实例（仅 DETR）或 None（Deformable DETR）
    """
    from transformers import DetrImageProcessor
    
    model_type = config['model'].get('type', 'detr').lower()
    
    if model_type == 'detr':
        # DETR 使用 HuggingFace 处理器
        model_name = config['model']['name']
        
        # 处理模型名称前缀
        if '/' not in model_name:
            model_name = f"facebook/{model_name}"
        
        print(f"🖼️  加载 DETR 图像处理器: {model_name}")
        return DetrImageProcessor.from_pretrained(model_name)
    
    elif model_type == 'deformable_detr' or model_type == 'deformable-detr':
        # Deformable DETR 不需要独立处理器（数据集直接生成官方格式）
        print(f"🖼️  Deformable DETR 使用官方数据流，无需独立处理器")
        return None
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


__all__ = [
    'build_detr_model',
    'build_model',
    'build_image_processor',
]
