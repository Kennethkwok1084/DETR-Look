#!/usr/bin/env python3
"""
预扫描数据集中的损坏图像，生成黑名单文件
使用: python tools/scan_corrupted_images.py --ann <annotation_file> --img-dir <image_dir>
"""

import argparse
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pycocotools.coco import COCO
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm


def check_image(img_path):
    """检查单张图像是否损坏"""
    try:
        read_image(str(img_path), mode=ImageReadMode.RGB)
        return None  # 正常
    except Exception as e:
        return (str(img_path), str(e))  # 损坏


def scan_dataset(ann_file, img_dir, blacklist_file, num_workers=8):
    """扫描数据集并生成黑名单"""
    print(f"🔍 扫描数据集: {ann_file}")
    print(f"   图像目录: {img_dir}")
    
    # 加载 COCO 标注
    coco = COCO(ann_file)
    img_root = Path(img_dir)
    
    # 获取所有图像路径
    img_paths = []
    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        img_path = img_root / img_info["file_name"]
        img_paths.append(img_path)
    
    print(f"   总图像数: {len(img_paths):,}")
    print(f"   并发线程: {num_workers}")
    print()
    
    # 并发检查
    corrupted = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_image, p): p for p in img_paths}
        
        with tqdm(total=len(img_paths), desc="检查图像") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    corrupted.append(result)
                pbar.update(1)
    
    # 保存黑名单
    blacklist_path = Path(blacklist_file)
    blacklist_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(blacklist_path, 'w') as f:
        json.dump({
            "annotation_file": str(ann_file),
            "image_dir": str(img_dir),
            "total_images": len(img_paths),
            "corrupted_count": len(corrupted),
            "corrupted_images": [
                {"path": path, "error": error} 
                for path, error in corrupted
            ]
        }, f, indent=2)
    
    # 打印结果
    print()
    if len(corrupted) == 0:
        print("✅ 所有图像正常")
    else:
        print(f"❌ 发现 {len(corrupted)} 张损坏图像:")
        for path, error in corrupted[:10]:  # 只显示前10个
            print(f"   {path}: {error}")
        if len(corrupted) > 10:
            print(f"   ... ({len(corrupted) - 10} 更多)")
    
    print()
    print(f"📝 黑名单已保存: {blacklist_path}")
    return len(corrupted)


def main():
    parser = argparse.ArgumentParser(description="扫描损坏图像")
    parser.add_argument("--ann", required=True, help="COCO标注文件")
    parser.add_argument("--img-dir", required=True, help="图像目录")
    parser.add_argument("--output", help="黑名单输出文件")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    args = parser.parse_args()
    
    # 默认输出路径
    if args.output is None:
        ann_name = Path(args.ann).stem
        args.output = f"outputs/blacklist_{ann_name}.json"
    
    corrupted_count = scan_dataset(args.ann, args.img_dir, args.output, args.workers)
    sys.exit(1 if corrupted_count > 0 else 0)


if __name__ == "__main__":
    main()
