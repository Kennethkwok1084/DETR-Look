#!/usr/bin/env python3
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional


def extract_zip(zip_path: Path, dst_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"压缩包不存在: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def move_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            print(f"⚠️  跳过已存在: {target}")
            continue
        shutil.move(str(item), str(target))


def find_single_root(root: Path, marker: str) -> Optional[Path]:
    if (root / marker).exists():
        return root
    children = [p for p in root.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / marker).exists():
        return children[0]
    return None


def deploy_bdd100k(src_dir: Path, dst_root: Path, use_det20: bool) -> None:
    print("\n== 部署 BDD100K ==")
    dst_dir = dst_root / "bdd100k"
    dst_dir.mkdir(parents=True, exist_ok=True)

    images_zip = src_dir / "bdd100k_images_100k.zip"
    labels_zip = src_dir / "bdd100k_labels.zip"
    det20_zip = src_dir / "bdd100k_det_20_labels.zip"

    print(f"解压: {images_zip}")
    extract_zip(images_zip, dst_dir)

    if labels_zip.exists():
        print(f"解压: {labels_zip}")
        extract_zip(labels_zip, dst_dir)
    else:
        raise FileNotFoundError(f"缺少逐图 JSON 标注包: {labels_zip}")

    if use_det20 and det20_zip.exists():
        print(f"解压: {det20_zip}")
        extract_zip(det20_zip, dst_dir)


def deploy_cctsdb(src_dir: Path, dst_root: Path) -> None:
    print("\n== 部署 CCTSDB ==")
    dst_dir = dst_root / "cctsdb"
    dst_dir.mkdir(parents=True, exist_ok=True)

    train_zip = src_dir / "train_img.zip"
    test_zip = src_dir / "test_img.zip"
    xml_zip = src_dir / "xml.zip"

    if not train_zip.exists() or not test_zip.exists() or not xml_zip.exists():
        raise FileNotFoundError("CCTSDB 缺少 train_img.zip/test_img.zip/xml.zip")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        extract_zip(train_zip, tmp_path)
        extract_zip(test_zip, tmp_path)
        extract_zip(xml_zip, tmp_path)

        train_src = next((p for p in [tmp_path / "train_img", tmp_path / "train"] if p.exists()), None)
        test_src = next((p for p in [tmp_path / "test_img", tmp_path / "test"] if p.exists()), None)
        xml_src = next((p for p in [tmp_path / "xml", tmp_path / "labels" / "xml"] if p.exists()), None)

        if train_src is None or test_src is None or xml_src is None:
            raise FileNotFoundError("CCTSDB 解压结构不符合预期（未找到 train/test/xml）")

        move_tree(train_src, dst_dir / "images" / "train")
        move_tree(test_src, dst_dir / "images" / "test")
        move_tree(xml_src, dst_dir / "labels" / "xml")


def deploy_tt100k(src_dir: Path, dst_root: Path) -> None:
    print("\n== 部署 TT100K ==")
    dst_dir = dst_root / "tt100k"
    dst_dir.mkdir(parents=True, exist_ok=True)

    tt_zip = src_dir / "tt100k_2021.zip"
    if not tt_zip.exists():
        raise FileNotFoundError(f"TT100K 压缩包不存在: {tt_zip}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        extract_zip(tt_zip, tmp_path)

        root = find_single_root(tmp_path, "annotations_all.json")
        if root is None:
            raise FileNotFoundError("TT100K 解压结构不符合预期（未找到 annotations_all.json）")

        move_tree(root, dst_dir)


def normalize_datasets(value: str) -> Iterable[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    return [v.lower() for v in items]


def main() -> None:
    parser = argparse.ArgumentParser(description="快速部署数据集（解压+落盘）")
    parser.add_argument("--src-dir", required=True, help="压缩包所在目录")
    parser.add_argument("--dst-root", default="data/raw", help="输出根目录")
    parser.add_argument(
        "--datasets",
        default="bdd100k,cctsdb,tt100k",
        help="要部署的数据集列表（逗号分隔）：bdd100k,cctsdb,tt100k",
    )
    parser.add_argument(
        "--with-det20",
        action="store_true",
        help="同时解压 bdd100k_det_20_labels.zip（可选）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="如目标目录已存在则先删除",
    )

    args = parser.parse_args()
    src_dir = Path(args.src_dir)
    dst_root = Path(args.dst_root)

    if not src_dir.exists():
        print(f"❌ 源目录不存在: {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.force and dst_root.exists():
        print(f"⚠️  清理输出目录: {dst_root}")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    datasets = normalize_datasets(args.datasets)

    if "bdd100k" in datasets:
        deploy_bdd100k(src_dir, dst_root, args.with_det20)
    if "cctsdb" in datasets:
        deploy_cctsdb(src_dir, dst_root)
    if "tt100k" in datasets:
        deploy_tt100k(src_dir, dst_root)

    print("\n✅ 数据集部署完成")


if __name__ == "__main__":
    main()
