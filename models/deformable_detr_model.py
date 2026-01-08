#!/usr/bin/env python3
"""
Deformable DETR 模型封装
基于官方实现，适配项目训练流程
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn

# 添加 third_party 路径到 sys.path
THIRD_PARTY_PATH = Path(__file__).parent.parent / "third_party" / "deformable_detr"
if str(THIRD_PARTY_PATH) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_PATH))

# 延迟导入，避免在模块加载时就报错
def _lazy_import_deformable_detr():
    """延迟导入 Deformable DETR 模块"""
    import sys
    import importlib
    
    # 保存并临时修改 sys.path
    # 保留虚拟环境路径（site-packages）但移除项目根目录
    original_path = sys.path.copy()
    venv_paths = [p for p in original_path if 'site-packages' in p or 'lib' in p.lower()]
    sys.path = [str(THIRD_PARTY_PATH)] + venv_paths
    
    # 删除已加载的项目 models 包，强制从 third_party 重新导入
    modules_to_clear = [k for k in list(sys.modules.keys()) 
                       if (k == 'models' or k.startswith('models.')) 
                       and not k.startswith('models.deformable_detr_model')]
    for k in modules_to_clear:
        sys.modules.pop(k, None)
    
    try:
        # 现在可以导入 models.deformable_detr (来自third_party/deformable_detr/models/)
        import models.deformable_detr as deformable_detr_module
        import models.backbone as backbone_module
        import models.matcher as matcher_module  
        import models.deformable_transformer as transformer_module
        import util.misc as misc_module
        
        result = {
            'DeformableDETR': deformable_detr_module.DeformableDETR,
            'SetCriterion': deformable_detr_module.SetCriterion,
            'MLP': deformable_detr_module.MLP,
            'build_backbone': backbone_module.build_backbone,
            'build_matcher': matcher_module.build_matcher,
            'build_deforamble_transformer': transformer_module.build_deforamble_transformer,
            'NestedTensor': misc_module.NestedTensor,
            'nested_tensor_from_tensor_list': misc_module.nested_tensor_from_tensor_list,
        }
        return result
    except Exception as e:
        raise ImportError(
            f"无法导入 Deformable DETR 模块。\n"
            f"请确保：\n"
            f"1. 已将官方源码复制到 {THIRD_PARTY_PATH}\n"
            f"2. 已编译 CUDA 扩展（如果使用 GPU）\n"
            f"错误详情: {e}"
        )
    finally:
        # 恢复原始 sys.path
        sys.path = original_path


class DeformableDETRModel(nn.Module):
    """
    Deformable DETR 模型封装
    适配现有训练流程，提供统一接口
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，包含模型参数
        """
        super().__init__()
        
        # 延迟导入模块
        modules = _lazy_import_deformable_detr()
        DeformableDETR = modules['DeformableDETR']
        SetCriterion = modules['SetCriterion']
        build_backbone = modules['build_backbone']
        build_matcher = modules['build_matcher']
        build_deforamble_transformer = modules['build_deforamble_transformer']
        
        self.config = config
        model_config = config['model']
        
        # 获取类别数
        num_classes = config['dataset']['num_classes']
        
        # 构建模型参数
        args = self._build_args(model_config, num_classes)
        
        # 构建 backbone
        print(f"🔨 构建 Deformable DETR backbone...")
        backbone = build_backbone(args)
        
        # 构建 transformer
        print(f"🔨 构建 Deformable Transformer...")
        transformer = build_deforamble_transformer(args)
        
        # 构建 Deformable DETR 模型
        print(f"🔨 构建 Deformable DETR 模型: {num_classes} 个类别")
        self.model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
        )
        
        # 构建 matcher 和 criterion
        matcher = build_matcher(args)
        weight_dict = {
            'loss_ce': args.cls_loss_coef,
            'loss_bbox': args.bbox_loss_coef,
            'loss_giou': args.giou_loss_coef,
        }
        
        # 辅助损失权重
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        losses = ['labels', 'boxes', 'cardinality']
        
        self.criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            focal_alpha=args.focal_alpha
        )
        
        print(f"✅ Deformable DETR 模型创建成功")
        
    def _build_args(self, model_config, num_classes):
        """构建模型参数对象"""
        class Args:
            pass
        
        args = Args()
        
        # Backbone
        args.backbone = model_config.get('backbone', 'resnet50')
        args.dilation = model_config.get('dilation', False)
        args.position_embedding = model_config.get('position_embedding', 'sine')
        args.position_embedding_scale = model_config.get('position_embedding_scale', 2 * 3.14159265359)
        args.num_feature_levels = model_config.get('num_feature_levels', 4)
        args.lr_backbone = model_config.get('lr_backbone', 1e-5)  # 添加 lr_backbone
        args.masks = model_config.get('masks', False)  # 添加 masks (分割任务)
        
        # Transformer
        args.enc_layers = model_config.get('enc_layers', 6)
        args.dec_layers = model_config.get('dec_layers', 6)
        args.dim_feedforward = model_config.get('dim_feedforward', 1024)
        args.hidden_dim = model_config.get('hidden_dim', 256)
        args.dropout = model_config.get('dropout', 0.1)
        args.nheads = model_config.get('nheads', 8)
        args.num_queries = model_config.get('num_queries', 300)
        args.dec_n_points = model_config.get('dec_n_points', 4)
        args.enc_n_points = model_config.get('enc_n_points', 4)
        
        # Deformable DETR 特有
        args.two_stage = model_config.get('two_stage', False)
        args.with_box_refine = model_config.get('with_box_refine', False)
        
        # Loss
        args.aux_loss = model_config.get('aux_loss', True)
        loss_weights = model_config.get('loss_weights', {})
        args.cls_loss_coef = loss_weights.get('class_loss_coef', 2.0)
        args.bbox_loss_coef = loss_weights.get('bbox_loss_coef', 5.0)
        args.giou_loss_coef = loss_weights.get('giou_loss_coef', 2.0)
        args.focal_alpha = loss_weights.get('focal_alpha', 0.25)
        
        # Matcher
        args.set_cost_class = model_config.get('set_cost_class', 2.0)
        args.set_cost_bbox = model_config.get('set_cost_bbox', 5.0)
        args.set_cost_giou = model_config.get('set_cost_giou', 2.0)
        
        # 其他
        args.num_classes = num_classes
        
        return args
    
    def forward(self, samples, targets=None):
        """
        前向传播
        
        Args:
            samples: 图像 tensor 或 NestedTensor
            targets: 训练时提供，List[Dict]，包含 'labels' 和 'boxes'
        
        Returns:
            如果 targets 不为 None，返回 loss dict
            否则返回预测结果
        """
        # 获取必要的类
        modules = _lazy_import_deformable_detr()
        NestedTensor = modules['NestedTensor']
        nested_tensor_from_tensor_list = modules['nested_tensor_from_tensor_list']
        
        # 确保输入是 NestedTensor 格式
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        # 模型前向传播
        outputs = self.model(samples)
        
        if targets is not None:
            # 训练模式：计算损失
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            
            # 加权损失
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            # 返回格式与 HF DETR 一致
            return type('Outputs', (), {
                'loss': losses,
                'loss_dict': loss_dict,
                'logits': outputs['pred_logits'],
                'pred_boxes': outputs['pred_boxes'],
            })()
        else:
            # 推理模式
            return outputs


def build_deformable_detr_model(config: dict) -> nn.Module:
    """
    构建 Deformable DETR 模型
    
    Args:
        config: 配置字典
    
    Returns:
        DeformableDETRModel 实例
    """
    return DeformableDETRModel(config)
