#!/usr/bin/env python3
"""
数据加载验证脚本
测试 C++ 图像解码 + DataLoader 性能
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_detr_optimized import CocoDetrDataset, collate_fn


def benchmark_dataloader(
    img_root: str,
    ann_file: str,
    batch_size: int = 16,
    num_workers: int = 12,
    prefetch_factor: int = 2,
    num_batches: int = 100,
):
    """Benchmark DataLoader 性能"""
    
    print("=" * 80)
    print("📊 DataLoader 性能测试")
    print("=" * 80)
    
    # 创建数据集
    dataset = CocoDetrDataset(img_root, ann_file, min_size=800, max_size=1333)
    
    # 使用子集（避免测试时间过长）
    subset_size = min(1000, len(dataset))
    subset = Subset(dataset, range(subset_size))
    
    print(f"数据集大小: {subset_size}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print(f"Prefetch factor: {prefetch_factor}")
    print("")
    
    # 创建 DataLoader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    # Warmup
    print("🔥 Warmup...")
    for i, batch in enumerate(loader):
        if i >= 5:
            break
    
    # Benchmark
    print("⏱️  Benchmark...")
    start_time = time.time()
    
    total_images = 0
    actual_batches = 0
    for i, batch in enumerate(loader):
        total_images += len(batch["pixel_values"])
        actual_batches += 1
        
        if i >= num_batches - 1:
            break
    
    elapsed = time.time() - start_time
    throughput = total_images / elapsed
    iter_speed = actual_batches / elapsed
    
    print("")
    print("=" * 80)
    print(f"📈 结果")
    print("=" * 80)
    print(f"总图像数: {total_images}")
    print(f"实际批次数: {actual_batches}")
    print(f"总耗时: {elapsed:.2f}s")
    print(f"吞吐量: {throughput:.2f} images/s")
    print(f"批次速度: {iter_speed:.2f} it/s")
    print("=" * 80)
    
    # 检查数据格式
    print("\n📦 数据格式检查")
    print("-" * 80)
    batch = next(iter(loader))
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]
    print(f"Batch 图像数: {len(pixel_values)}")
    print(f"图像形状: {pixel_values[0].shape} (C, H, W)")
    print(f"图像类型: {pixel_values[0].dtype}")
    print(f"图像范围: [{pixel_values[0].min():.3f}, {pixel_values[0].max():.3f}]")
    print(f"Labels[0] 键: {list(labels[0].keys())}")
    print(f"Boxes 形状: {labels[0]['boxes'].shape}")
    print(f"Class labels: {labels[0]['class_labels'][:5].tolist() if len(labels[0]['class_labels']) > 0 else []}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str,
                       default="data/traffic_coco/bdd100k_det/images/train")
    parser.add_argument("--ann-file", type=str,
                       default="data/traffic_coco/bdd100k_det/annotations/instances_train.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=100)
    
    args = parser.parse_args()
    
    benchmark_dataloader(
        args.img_root,
        args.ann_file,
        args.batch_size,
        args.num_workers,
        args.prefetch_factor,
        args.num_batches,
    )


if __name__ == "__main__":
    main()
