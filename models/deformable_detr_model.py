#!/usr/bin/env python3
"""
Deformable DETR 模型构建（官方数据流）
基于官方实现，保持与 SetCriterion / PostProcess 的完全兼容性

导入策略：模块级缓存 + 平衡隔离
- 官方模块通过模块级变量缓存供本文件使用
- 保留第三方子模块（models.*/util.*）支持 pickle 序列化
- 仅恢复主模块（models/util）确保后续导入指向本地
- sys.path 完全恢复，避免全局污染
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

# === 模块级缓存：避免重复导入和 sys.modules 污染 ===
_official_modules_cache = {}

def _import_official_modules():
    """
    隔离导入官方 Deformable DETR 模块
    返回模块引用的字典，保留子模块支持 pickle/torch.save
    
    策略调整：
    1. 保存本地 models/util 主模块
    2. 临时删除本地主模块（保留子模块）
    3. 导入官方模块
    4. 缓存到模块变量
    5. 恢复本地主模块
    6. 保留第三方子模块在 sys.modules（支持序列化）
    """
    if _official_modules_cache:
        return _official_modules_cache
    
    # 1. 保存当前状态
    _local_models = sys.modules.get('models')
    _local_util = sys.modules.get('util')
    _original_sys_path = sys.path.copy()
    
    # 2. 临时移除本地 models/util 主模块（保留子模块）
    if 'models' in sys.modules:
        del sys.modules['models']
    if 'util' in sys.modules:
        del sys.modules['util']
    
    # 3. 添加官方代码路径
    _third_party_detr = Path(__file__).parent.parent / "third_party" / "deformable_detr"
    sys.path.insert(0, str(_third_party_detr))
    
    try:
        # 4. 导入官方模块
        from models.deformable_detr import DeformableDETR, SetCriterion, PostProcess, MLP
        from models.backbone import build_backbone
        from models.matcher import build_matcher
        from models.deformable_transformer import build_deforamble_transformer
        from util.misc import NestedTensor, nested_tensor_from_tensor_list
        
        # 5. 缓存到模块级变量
        _official_modules_cache.update({
            'DeformableDETR': DeformableDETR,
            'SetCriterion': SetCriterion,
            'PostProcess': PostProcess,
            'MLP': MLP,
            'build_backbone': build_backbone,
            'build_matcher': build_matcher,
            'build_deforamble_transformer': build_deforamble_transformer,
            'NestedTensor': NestedTensor,
            'nested_tensor_from_tensor_list': nested_tensor_from_tensor_list,
        })
        
    finally:
        # 6. 恢复本地 models/util 主模块
        if _local_models is not None:
            sys.modules['models'] = _local_models
        if _local_util is not None:
            sys.modules['util'] = _local_util
        
        # 7. 恢复 sys.path
        sys.path[:] = _original_sys_path
        
        # 注意：保留第三方子模块在 sys.modules（如 models.deformable_detr）
        # 这样 torch.save(model) 时 pickle 可以找到类定义
    
    return _official_modules_cache

# 导入并缓存官方模块
_modules = _import_official_modules()
DeformableDETR = _modules['DeformableDETR']
SetCriterion = _modules['SetCriterion']
PostProcess = _modules['PostProcess']
MLP = _modules['MLP']
build_backbone = _modules['build_backbone']
build_matcher = _modules['build_matcher']
build_deforamble_transformer = _modules['build_deforamble_transformer']
NestedTensor = _modules['NestedTensor']
nested_tensor_from_tensor_list = _modules['nested_tensor_from_tensor_list']


class DeformableDETRModelWrapper(nn.Module):
    """
    Deformable DETR 模型封装
    使用官方 build() 函数的逻辑，但适配我们的配置格式
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: 配置字典，包含模型参数
        """
        super().__init__()
        self.config = config
        model_config = config['model']
        
        # 构建 args 对象（模拟官方 argparse）
        args = self._build_args(config)
        
        # 使用官方 build 逻辑
        self.model, self.criterion, self.postprocessors = self._build_official(args)
        
        print(f"✅ Deformable DETR 模型创建成功")
        print(f"   - 类别数: {args.num_classes}")
        print(f"   - 查询数: {args.num_queries}")
        print(f"   - 特征层级: {args.num_feature_levels}")
        print(f"   - 两阶段: {args.two_stage}")
        print(f"   - Box Refine: {args.with_box_refine}")
        
    def _build_args(self, config):
        """构建官方格式的 args 对象"""
        class Args:
            pass
        
        args = Args()
        model_config = config['model']
        
        # 基础参数
        args.num_classes = config['dataset']['num_classes']
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 模型参数
        args.num_queries = model_config.get('num_queries', 300)
        args.num_feature_levels = model_config.get('num_feature_levels', 4)
        args.aux_loss = model_config.get('aux_loss', True)
        args.with_box_refine = model_config.get('with_box_refine', False)
        args.two_stage = model_config.get('two_stage', False)
        args.masks = False  # 暂不支持分割
        args.frozen_weights = None
        
        # Backbone 参数
        args.backbone = model_config.get('backbone', 'resnet50')
        args.dilation = model_config.get('dilation', False)
        args.position_embedding = model_config.get('position_embedding', 'sine')
        args.position_embedding_scale = model_config.get('position_embedding_scale', 2 * 3.141592653589793)
        args.num_feature_levels = model_config.get('num_feature_levels', 4)
        
        # Transformer 参数（支持多种配置键名）
        args.enc_layers = model_config.get('enc_layers', model_config.get('num_encoder_layers', 6))
        args.dec_layers = model_config.get('dec_layers', model_config.get('num_decoder_layers', 6))
        args.dim_feedforward = model_config.get('dim_feedforward', 1024)
        args.hidden_dim = model_config.get('hidden_dim', 256)
        args.dropout = model_config.get('dropout', 0.1)
        args.nheads = model_config.get('nheads', 8)
        args.num_queries = args.num_queries
        args.dec_n_points = model_config.get('dec_n_points', 4)
        args.enc_n_points = model_config.get('enc_n_points', 4)
        
        # Matcher 参数
        args.set_cost_class = model_config.get('set_cost_class', 2.0)
        args.set_cost_bbox = model_config.get('set_cost_bbox', 5.0)
        args.set_cost_giou = model_config.get('set_cost_giou', 2.0)
        
        # Loss 参数
        loss_weights = model_config.get('loss_weights', {})
        args.cls_loss_coef = loss_weights.get('class_loss_coef', 2.0)
        args.bbox_loss_coef = loss_weights.get('bbox_loss_coef', 5.0)
        args.giou_loss_coef = loss_weights.get('giou_loss_coef', 2.0)
        args.mask_loss_coef = loss_weights.get('mask_loss_coef', 1.0)
        args.dice_loss_coef = loss_weights.get('dice_loss_coef', 1.0)
        # focal_alpha 可能在 model_config 或 loss_weights 中
        args.focal_alpha = loss_weights.get('focal_alpha', model_config.get('focal_alpha', 0.25))
        
        # 其他参数
        args.dataset_file = 'coco'
        
        return args
    
    def _build_official(self, args):
        """使用官方 build 函数构建模型"""
        device = torch.device(args.device)

        # 构建 Backbone
        backbone = build_backbone(args)

        # 构建 Transformer
        transformer = build_deforamble_transformer(args)
        
        # 构建模型
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            num_feature_levels=args.num_feature_levels,
            aux_loss=args.aux_loss,
            with_box_refine=args.with_box_refine,
            two_stage=args.two_stage,
        )
        
        # 构建 Matcher
        matcher = build_matcher(args)
        
        # 构建 Loss 权重
        weight_dict = {
            'loss_ce': args.cls_loss_coef,
            'loss_bbox': args.bbox_loss_coef,
            'loss_giou': args.giou_loss_coef
        }
        
        if args.masks:
            weight_dict["loss_mask"] = args.mask_loss_coef
            weight_dict["loss_dice"] = args.dice_loss_coef
        
        # 辅助损失权重
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        # 损失类型
        losses = ['labels', 'boxes', 'cardinality']
        if args.masks:
            losses += ["masks"]
        
        # 构建 Criterion
        criterion = SetCriterion(
            args.num_classes,
            matcher,
            weight_dict,
            losses,
            focal_alpha=args.focal_alpha
        )
        criterion.to(device)
        
        # 构建 PostProcessors
        postprocessors = {'bbox': PostProcess()}
        
        return model, criterion, postprocessors
    
    def forward(self, samples, targets=None):
        """
        前向传播（官方格式）
        
        Args:
            samples: NestedTensor 或 tensor list
                - samples.tensor: [B, 3, H, W]
                - samples.mask: [B, H, W] (True for padding)
            targets: List[Dict] (训练时提供)
                每个 dict 包含:
                - boxes: [N, 4] tensor (normalized cxcywh)
                - labels: [N] tensor (class indices)
                - orig_size: [2] tensor (H, W)
                - size: [2] tensor (H, W)
        
        Returns:
            训练模式: loss_dict
            推理模式: outputs (pred_logits, pred_boxes)
        """
        # 确保输入是 NestedTensor
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        
        # 前向传播
        outputs = self.model(samples)
        
        if targets is not None:
            # 训练模式：计算损失
            loss_dict = self.criterion(outputs, targets)
            return loss_dict
        else:
            # 推理模式：返回原始输出
            return outputs
    
    def postprocess(self, outputs, target_sizes):
        """
        后处理（使用官方 PostProcess）
        
        Args:
            outputs: 模型输出 dict
            target_sizes: [B, 2] tensor (H, W)
        
        Returns:
            List[Dict]: 每个包含 scores, labels, boxes
        """
        return self.postprocessors['bbox'](outputs, target_sizes)


def build_deformable_detr_model(config: dict) -> nn.Module:
    """
    构建 Deformable DETR 模型
    
    Args:
        config: 配置字典
    
    Returns:
        DeformableDETRModelWrapper 实例
    """
    return DeformableDETRModelWrapper(config)
