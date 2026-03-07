#!/usr/bin/env python3
"""
DETR 训练脚本（使用 transformers 库 + 优化的数据加载）
- 使用 torchvision.io.read_image（C++ 解码）
- 优化 DataLoader 参数
- 支持 AMP、checkpoint、评估
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.functional as F
from transformers import DetrForObjectDetection, DetrConfig, DetrImageProcessor

# DETR 标准归一化参数（ImageNet）
DETR_MEAN = [0.485, 0.456, 0.406]
DETR_STD = [0.229, 0.224, 0.225]


class CocoDetrDataset(torch.utils.data.Dataset):
    """COCO 格式数据集（C++ 解码 + transformers DETR 格式）"""
    
    def __init__(
        self,
        img_root: str,
        ann_file: str,
        min_size: int = 800,
        max_size: int = 1333,
        is_train: bool = True,
        blacklist_file: str = None,
    ):
        self.root = Path(img_root)
        self.coco = COCO(str(ann_file))
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # 加载黑名单并过滤
        if blacklist_file and Path(blacklist_file).exists():
            with open(blacklist_file) as f:
                blacklist_data = json.load(f)
            corrupted_paths = {item["path"] for item in blacklist_data.get("corrupted_images", [])}
            
            # 过滤损坏图像
            original_count = len(self.ids)
            self.ids = [
                img_id for img_id in self.ids
                if str(self.root / self.coco.loadImgs(img_id)[0]["file_name"]) not in corrupted_paths
            ]
            filtered_count = original_count - len(self.ids)
            if filtered_count > 0:
                print(f"📋 黑名单过滤: {filtered_count} 张损坏图像已跳过")
        
        # 类别映射到连续 [0..N-1]
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        self.reverse_cat_id_map = {i: cat_id for i, cat_id in enumerate(cat_ids)}  # 反向映射
        self.num_classes = len(cat_ids)
        
        self.min_size = min_size
        self.max_size = max_size
        self.is_train = is_train

    def __len__(self):
        return len(self.ids)

    def _resize(self, image: torch.Tensor, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """保持纵横比的 resize 并转换 bbox 为归一化 cxcywh"""
        c, h, w = image.shape
        
        min_original_size = float(min(h, w))
        max_original_size = float(max(h, w))
        
        if max_original_size / min_original_size * self.min_size > self.max_size:
            size = int(round(self.max_size * min_original_size / max_original_size))
        else:
            size = self.min_size
        
        scale_factor = size / min_original_size
        
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        image = F.resize(image, [new_h, new_w])
        
        # 转换 bbox：xyxy 像素 -> 归一化 cxcywh（DETR 标准格式）
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"] * scale_factor  # 先缩放到 resize 后的像素坐标
            # xyxy -> cxcywh
            boxes_cxcywh = torch.zeros_like(boxes)
            boxes_cxcywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # cx
            boxes_cxcywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # cy
            boxes_cxcywh[:, 2] = boxes[:, 2] - boxes[:, 0]        # w
            boxes_cxcywh[:, 3] = boxes[:, 3] - boxes[:, 1]        # h
            # 归一化
            boxes_cxcywh[:, [0, 2]] /= new_w
            boxes_cxcywh[:, [1, 3]] /= new_h
            # Clamp 到 [0, 1] 防止越界标注导致 loss 异常
            boxes_cxcywh = torch.clamp(boxes_cxcywh, min=0.0, max=1.0)
            
            # 过滤零宽/零高框（clamp 后可能出现，会导致 loss/匹配不稳定）
            valid_mask = (boxes_cxcywh[:, 2] > 0) & (boxes_cxcywh[:, 3] > 0)
            boxes_cxcywh = boxes_cxcywh[valid_mask]
            target["class_labels"] = target["class_labels"][valid_mask]
            
            target["boxes"] = boxes_cxcywh
        
        # 删除 area 字段（clamp 后 area 会不一致，且 DETR 训练不依赖 area）
        if "area" in target:
            del target["area"]
        
        target["size"] = torch.tensor([new_h, new_w])  # resize 后尺寸
        target["orig_size"] = torch.tensor([h, w])    # 原始图像尺寸（用于评估）
        
        return image, target

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root / img_info["file_name"]

        # C++ 解码（黑名单过滤 + 后备跳过）
        try:
            img = read_image(str(img_path), mode=ImageReadMode.RGB).float() / 255.0
        except Exception as e:
            # 损坏图像后备处理（黑名单缺失/不完整时）
            print(f"\n⚠️  跳过损坏图像: {img_path} ({e})")
            print(f"   建议运行: python tools/scan_corrupted_images.py --ann {self.coco.dataset.get('info', {}).get('description', 'annotation')} --img-dir {self.root}\n")
            # 跳过到下一个（避免无限递归）
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                # 最后一张图损坏，返回第一张
                return self.__getitem__(0)
        
        # DETR 标准归一化（ImageNet）
        for c in range(3):
            img[c] = (img[c] - DETR_MEAN[c]) / DETR_STD[c]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_map[ann["category_id"]])
            areas.append(ann.get("area", w * h))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "class_labels": labels,  # transformers 使用 class_labels
            "image_id": torch.tensor([img_id]),
            "area": areas,
        }

        img, target = self._resize(img, target)
        return img, target


def collate_fn(batch):
    """自定义 collate（transformers DETR 格式）"""
    # 获取batch中的最大尺寸
    max_h = max([img.shape[1] for img, _ in batch])
    max_w = max([img.shape[2] for img, _ in batch])
    
    # Pad到相同尺寸
    pixel_values = []
    pixel_mask = []
    labels = []
    
    for img, target in batch:
        c, h, w = img.shape
        padded_img = torch.zeros(c, max_h, max_w)
        padded_img[:, :h, :w] = img
        pixel_values.append(padded_img)
        
        # mask: 1表示真实像素，0表示padding
        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        pixel_mask.append(mask)
        
        labels.append(target)
    
    return {
        "pixel_values": torch.stack(pixel_values),
        "pixel_mask": torch.stack(pixel_mask),
        "labels": labels,
    }


def build_model(num_classes: int, pretrained: bool = True, offline_mode: bool = False):
    """构建 DETR 模型"""
    if pretrained:
        try:
            # 优先尝试本地缓存
            model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                local_files_only=offline_mode,
            )
        except Exception as e:
            if offline_mode:
                print(f"⚠️  离线模式下无法加载预训练模型: {e}")
                print("⚠️  将使用随机初始化模型")
                config = DetrConfig(num_labels=num_classes, num_queries=100)
                model = DetrForObjectDetection(config)
            else:
                # 非离线模式，允许下载
                model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50",
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
    else:
        config = DetrConfig(num_labels=num_classes, num_queries=100)
        model = DetrForObjectDetection(config)
    
    return model


def save_checkpoint(model, optimizer, epoch, iteration, best_map, output_dir, is_best=False):
    """保存 checkpoint"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "iteration": iteration,
        "best_map": best_map,
    }
    
    torch.save(checkpoint, output_dir / "last.pth")
    
    if is_best:
        torch.save(checkpoint, output_dir / "best.pth")
        print(f"✅ 保存最佳模型: mAP={best_map:.4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载 checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("iteration", 0), checkpoint.get("best_map", 0.0)


@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt, reverse_cat_id_map=None, processor=None, score_threshold=0.05, offline_mode=False):
    """COCO 评估（使用 DetrImageProcessor.post_process_object_detection）"""
    model.eval()
    
    try:
        # 创建 processor（用于 post_process）
        if processor is None:
            try:
                processor = DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50",
                    local_files_only=True  # 优先使用缓存
                )
            except Exception as e:
                if offline_mode:
                    print(f"⚠️  离线模式下无法加载 DetrImageProcessor 缓存，跳过评估: {e}")
                    return None  # 跳过评估
                # 缓存不存在时才下载
                processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        
        results = []
        
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
            labels = batch["labels"]
            
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            
            # 使用官方 post_process（正确处理 padding 和归一化）
            # target_sizes: 原始图像尺寸（未 resize，用于将预测框坐标转换回原始坐标系）
            target_sizes = torch.stack([l["orig_size"] for l in labels]).to(device)
            
            # post_process_object_detection 返回 [{'scores', 'labels', 'boxes'}, ...]
            results_per_image = processor.post_process_object_detection(
                outputs, 
                threshold=score_threshold,  # 使用可调置信度阈值
                target_sizes=target_sizes
            )
            
            for i, result in enumerate(results_per_image):
                image_id = labels[i]["image_id"].item()
                
                boxes = result["boxes"].cpu()  # xyxy 像素坐标（原始图像坐标系）
                scores = result["scores"].cpu()
                pred_labels = result["labels"].cpu()
                
                for box, score, label in zip(boxes, scores, pred_labels):
                    x1, y1, x2, y2 = box.tolist()
                    
                    # 反向映射到原始 category_id（如果提供）
                    original_cat_id = label.item()
                    if reverse_cat_id_map is not None:
                        original_cat_id = reverse_cat_id_map.get(label.item(), label.item())
                    
                    results.append({
                        "image_id": image_id,
                        "category_id": original_cat_id,  # 使用原始 category_id
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO 格式 xywh
                        "score": score.item(),
                    })
        
        if not results:
            return {"mAP": 0.0, "AP_small": 0.0}
        
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return {
            "mAP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
        }
    finally:
        # 确保所有返回路径都恢复训练模式（包括异常/提前返回）
        model.train()


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp, amp_dtype, 
                    scaler=None, print_freq=50, grad_accum=1, clip_max_norm=None):
    """训练一个 epoch
    
    Args:
        grad_accum: 梯度累积步数（有效batch = batch_size * grad_accum）
        clip_max_norm: 梯度裁剪最大范数（None=不裁剪）
    """
    model.train()
    total_loss = 0.0
    start_time = time.time()
    
    # 分段计时统计
    total_t_load = 0.0
    total_t_step = 0.0
    t_batch_start = time.time()
    
    for step, batch in enumerate(data_loader, start=1):
        # 数据加载耗时
        t_load = time.time() - t_batch_start
        total_t_load += t_load
        
        t_step_start = time.time()
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = batch["labels"]
        
        # 转换 labels 格式（boxes 已经是归一化 cxcywh）
        batch_labels = []
        for target in labels:
            batch_labels.append({
                "class_labels": target["class_labels"].to(device, non_blocking=True),
                "boxes": target["boxes"].to(device, non_blocking=True),  # 归一化 cxcywh
            })
        
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=batch_labels)
            loss = outputs.loss
            
            # 梯度累积：loss缩放，防止累积后梯度放大
            if grad_accum > 1:
                loss = loss / grad_accum
        
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # 梯度累积：每grad_accum步更新一次
            if step % grad_accum == 0:
                # 梯度裁剪（在unscale后、step前）
                if clip_max_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            
            # 梯度累积：每grad_accum步更新一次
            if step % grad_accum == 0:
                # 梯度裁剪
                if clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # 记录原始loss（未缩放）
        total_loss += loss.item() * (grad_accum if grad_accum > 1 else 1)
        
        # 训练计算耗时
        torch.cuda.synchronize() if device.type == "cuda" else None
        t_step = time.time() - t_step_start
        total_t_step += t_step
        
        if step % print_freq == 0:
            elapsed = time.time() - start_time
            it_s = step / elapsed
            avg_loss = total_loss / step
            
            # 计算耗时占比
            avg_t_load = total_t_load / step
            avg_t_step = total_t_step / step
            pct_load = 100.0 * total_t_load / elapsed
            pct_step = 100.0 * total_t_step / elapsed
            
            # 显示实际loss值（已还原）
            actual_loss = loss.item() * (grad_accum if grad_accum > 1 else 1)
            print(f"Epoch [{epoch}] Step [{step}/{len(data_loader)}] "
                  f"Loss: {actual_loss:.4f} (avg: {avg_loss:.4f}) | Speed: {it_s:.2f} it/s")
            print(f"  ⏱️  t_load: {avg_t_load:.3f}s ({pct_load:.1f}%) | t_step: {avg_t_step:.3f}s ({pct_step:.1f}%)")
        
        t_batch_start = time.time()
    
    # 刷新残留梯度（当batch数不能被grad_accum整除时）
    if grad_accum > 1 and len(data_loader) % grad_accum != 0:
        if scaler is not None:
            if clip_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"  🔄 刷新残留梯度（{len(data_loader)} % {grad_accum} = {len(data_loader) % grad_accum} 个 batch）")
    
    return {"loss": total_loss / len(data_loader)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-img", required=True)
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-img")
    parser.add_argument("--val-ann")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--subset", type=int)
    parser.add_argument("--output-dir", default="outputs/detr_optimized")
    parser.add_argument("--resume")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--score-threshold", type=float, default=0.05, help="评估时的置信度阈值")
    parser.add_argument("--offline", action="store_true", help="离线模式，不下载预训练模型")
    parser.add_argument("--no-eval", action="store_true", help="跳过评估（离线无缓存时自动跳过）")
    parser.add_argument("--blacklist", help="损坏图像黑名单文件")
    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数（有效batch=batch_size*grad_accum）")
    parser.add_argument("--clip-max-norm", type=float, help="梯度裁剪最大范数（推荐0.1，None=不裁剪）")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # 全局变量用于信号处理
    interrupted = False
    checkpoint_data = {}
    
    def save_interrupt_checkpoint(model, optimizer, scheduler, epoch, output_dir):
        """保存中断时的checkpoint"""
        ckpt_path = output_dir / "interrupted.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        }, ckpt_path)
        print(f"\n💾 已保存中断checkpoint: {ckpt_path}")
    
    def signal_handler(signum, frame):
        """处理Ctrl+C信号"""
        nonlocal interrupted
        print(f"\n\n⚠️  收到中断信号 (Ctrl+C)，正在保存checkpoint...")
        interrupted = True
        if checkpoint_data:
            save_interrupt_checkpoint(**checkpoint_data)
        print("✅ Checkpoint已保存，训练将在当前epoch结束后停止")
        print("   可使用 --resume interrupted.pth 恢复训练\n")
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    print("=" * 80)
    print("🚀 DETR 训练（transformers + 优化数据加载）")
    print("=" * 80)
    print(f"输出目录: {output_dir}")
    print(f"设备: {device}")
    print(f"Batch Size: {args.batch_size} | Workers: {args.num_workers}")
    print(f"图像尺寸: min={args.min_size}, max={args.max_size}")
    
    # 梯度累积和裁剪信息
    if args.grad_accum > 1:
        effective_batch = args.batch_size * args.grad_accum
        print(f"梯度累积: {args.grad_accum} 步 | 有效Batch: {effective_batch}")
    if args.clip_max_norm is not None:
        print(f"梯度裁剪: clip_max_norm={args.clip_max_norm}")
    
    print("=" * 80)
    
    # 数据集
    train_dataset = CocoDetrDataset(
        args.train_img, args.train_ann, args.min_size, args.max_size, 
        blacklist_file=args.blacklist
    )
    
    # 自动推断类别数（从数据集）
    actual_num_classes = train_dataset.num_classes
    if args.num_classes != actual_num_classes:
        print(f"⚠️  命令行指定 --num-classes={args.num_classes}，但数据集有 {actual_num_classes} 个类别")
        print(f"    自动使用数据集类别数: {actual_num_classes}")
        args.num_classes = actual_num_classes
    
    # 写入配置（在 num_classes 校验后，确保配置记录准确）
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    if args.subset:
        train_dataset = Subset(train_dataset, range(min(args.subset, len(train_dataset))))
        print(f"📊 使用训练子集: {len(train_dataset)} 张图像")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    
    val_loader = None
    coco_gt = None
    temp_ann_file = None  # 记录临时文件路径用于清理
    if args.val_img and args.val_ann:
        val_dataset_base = CocoDetrDataset(args.val_img, args.val_ann, args.min_size, args.max_size, is_train=False)
        val_dataset = val_dataset_base
        
        if args.subset:
            subset_size = min(args.subset // 4, len(val_dataset_base))
            
            # 防止 subset 验证集为空（args.subset < 4 时）
            if subset_size == 0:
                print("⚠️  subset 太小，验证集为空，跳过评估")
                val_loader = None
            else:
                val_dataset = Subset(val_dataset_base, range(subset_size))
                
                # 创建只包含 subset 图像的临时 COCO（用于准确评估）
                subset_img_ids = [val_dataset_base.ids[i] for i in range(subset_size)]
                coco_full = COCO(args.val_ann)
                
                # 构建子集标注
                subset_anns = {
                    "images": [img for img in coco_full.dataset["images"] if img["id"] in subset_img_ids],
                    "annotations": [ann for ann in coco_full.dataset["annotations"] if ann["image_id"] in subset_img_ids],
                    "categories": coco_full.dataset["categories"]
                }
                
                # 创建临时 COCO 对象
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(subset_anns, f)
                    temp_ann_file = f.name  # 保存路径用于清理
                
                coco_gt = COCO(temp_ann_file)
                print(f"📊 验证集: {len(val_dataset)} 张图像（subset 模式，使用子集标注）")
        else:
            coco_gt = COCO(args.val_ann)
            print(f"📊 验证集: {len(val_dataset)} 张图像")
        
        if val_loader is None and args.subset and subset_size == 0:
            pass  # 已跳过
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
    
    # 模型
    print(f"📦 构建模型: num_classes={args.num_classes}")
    model = build_model(args.num_classes, args.pretrained, args.offline)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    amp_dtype = torch.bfloat16 if args.amp and torch.cuda.is_bf16_supported() else torch.float16
    use_amp = args.amp and device.type == "cuda"
    
    # 初始化 GradScaler（fp16 时需要）
    scaler = None
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()
        print(f"⚡ 使用 AMP (fp16) + GradScaler")
    elif use_amp:
        print(f"⚡ 使用 AMP (bf16)")
    
    start_epoch = 0
    best_map = 0.0
    
    if args.resume:
        start_epoch, _, best_map = load_checkpoint(args.resume, model, optimizer)
    
    print("\n" + "=" * 80)
    print("🎯 开始训练")
    print("=" * 80)
    
    metrics_log = []
    
    for epoch in range(start_epoch, args.num_epochs):
        # 更新checkpoint数据供信号处理器使用
        checkpoint_data.update({
            'model': model,
            'optimizer': optimizer,
            'scheduler': None,  # 如果有scheduler可以在这里更新
            'epoch': epoch,
            'output_dir': output_dir
        })
        
        # 检查是否被中断
        if interrupted:
            print("\n🛑 训练已被用户中断")
            break
        
        epoch_start = time.time()
        
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch + 1,
            use_amp, amp_dtype, scaler, args.print_freq,
            grad_accum=args.grad_accum,
            clip_max_norm=args.clip_max_norm
        )
        
        epoch_time = time.time() - epoch_start
        
        eval_metrics = {}
        is_best = False
        
        if val_loader and ((epoch + 1) % args.eval_interval == 0 or epoch == args.num_epochs - 1) and not args.no_eval:
            print(f"\n📊 评估 Epoch {epoch + 1}...")
            # 传递反向映射字典
            reverse_map = getattr(val_loader.dataset.dataset if hasattr(val_loader.dataset, 'dataset') else val_loader.dataset, 'reverse_cat_id_map', None)
            eval_metrics = evaluate(
                model, val_loader, device, coco_gt, 
                reverse_cat_id_map=reverse_map,
                score_threshold=args.score_threshold,
                offline_mode=args.offline
            )
            
            # 离线模式无缓存时 evaluate 返回 None
            if eval_metrics is None:
                print("⚠️  评估已跳过（离线模式无 processor 缓存）")
                eval_metrics = {}
            elif eval_metrics.get("mAP", 0) > best_map:
                best_map = eval_metrics["mAP"]
                is_best = True
        
        save_checkpoint(model, optimizer, epoch + 1, 0, best_map, output_dir, is_best)
        
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "epoch_time": epoch_time,
            **eval_metrics,
        }
        metrics_log.append(log_entry)
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=2)
        
        print(f"\n{'=' * 80}")
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] 完成 | 耗时: {epoch_time:.1f}s")
        print(f"训练 Loss: {train_metrics['loss']:.4f}")
        if eval_metrics:
            print(f"验证 mAP: {eval_metrics['mAP']:.4f} | AP_small: {eval_metrics['AP_small']:.4f}")
            print(f"最佳 mAP: {best_map:.4f}")
        print(f"{'=' * 80}\n")
    
    # 清理临时文件
    if temp_ann_file and Path(temp_ann_file).exists():
        Path(temp_ann_file).unlink()
        print(f"🧹 已清理临时标注文件: {temp_ann_file}")
    
    print("✅ 训练完成！")


if __name__ == "__main__":
    main()
