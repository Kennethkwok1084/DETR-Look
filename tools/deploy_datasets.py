#!/usr/bin/env python3
import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


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
    
    images_zip = src_dir / "bdd100k_images_100k.zip"
    labels_zip = src_dir / "bdd100k_labels.zip"
    det20_zip = src_dir / "bdd100k_det_20_labels.zip"

    # 解压到临时目录以便重新组织结构
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 解压图片
        print(f"解压: {images_zip}")
        img_tmp = tmp_path / "images"
        img_tmp.mkdir()
        extract_zip(images_zip, img_tmp)
        
        # 解压标注
        if labels_zip.exists():
            print(f"解压: {labels_zip}")
            label_tmp = tmp_path / "labels"
            label_tmp.mkdir()
            extract_zip(labels_zip, label_tmp)
        else:
            raise FileNotFoundError(f"缺少逐图 JSON 标注包: {labels_zip}")
        
        # 解压 det_20（可选）
        if use_det20 and det20_zip.exists():
            print(f"解压: {det20_zip}")
            det20_tmp = tmp_path / "det20"
            det20_tmp.mkdir()
            extract_zip(det20_zip, det20_tmp)
        
        # 查找解压后的实际根目录（可能有多层嵌套）
        img_root = find_single_root(img_tmp, "train") or img_tmp
        label_root = find_single_root(label_tmp, "labels") or label_tmp
        
        # 移动图片到规范路径: images/100k/{split}
        for split in ["train", "val", "test"]:
            src_img = img_root / split
            if src_img.exists():
                dst_img = dst_dir / "images" / "100k" / split
                print(f"📁 移动图片: {split} -> {dst_img.relative_to(dst_root)}")
                move_tree(src_img, dst_img)
        
        # 移动标注到规范路径: labels/
        labels_src = label_root / "labels" if (label_root / "labels").exists() else label_root
        if labels_src.exists():
            print(f"📁 移动标注: labels/ -> {(dst_dir / 'labels').relative_to(dst_root)}")
            move_tree(labels_src, dst_dir / "labels")
        
        # 移动 det_20 标注（如果有）
        if use_det20 and det20_zip.exists():
            det20_root = find_single_root(det20_tmp, "labels") or det20_tmp
            det20_labels = det20_root / "labels" / "det_20" if (det20_root / "labels" / "det_20").exists() else det20_root
            if det20_labels.exists():
                print(f"📁 移动 det_20 标注")
                move_tree(det20_labels, dst_dir / "labels" / "det_20")


def find_dir_with_suffix(root: Path, suffixes: Sequence[str]) -> Optional[Path]:
    suffixes = tuple(s.lower() for s in suffixes)
    if root.exists():
        if any(p.is_file() and p.suffix.lower() in suffixes for p in root.iterdir()):
            return root
    candidates: list[Tuple[int, int, Path]] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
        if not files:
            continue
        depth = len(path.relative_to(root).parts)
        candidates.append((len(files), depth, path))
    if not candidates:
        return None
    # 选取文件数最多且层级最浅的目录
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][2]


def deploy_cctsdb(src_dir: Path, dst_root: Path, tmp_base: Path) -> None:
    print("\n== 部署 CCTSDB ==")
    dst_dir = dst_root / "cctsdb"
    dst_dir.mkdir(parents=True, exist_ok=True)

    train_zip = src_dir / "train_img.zip"
    test_zip = src_dir / "test_img.zip"
    xml_zip = src_dir / "xml.zip"

    if not train_zip.exists() or not test_zip.exists() or not xml_zip.exists():
        raise FileNotFoundError("CCTSDB 缺少 train_img.zip/test_img.zip/xml.zip")

    tmp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_tmp = tmp_path / "train_zip"
        test_tmp = tmp_path / "test_zip"
        xml_tmp = tmp_path / "xml_zip"
        train_tmp.mkdir(parents=True, exist_ok=True)
        test_tmp.mkdir(parents=True, exist_ok=True)
        xml_tmp.mkdir(parents=True, exist_ok=True)

        extract_zip(train_zip, train_tmp)
        extract_zip(test_zip, test_tmp)
        extract_zip(xml_zip, xml_tmp)

        train_src = find_dir_with_suffix(train_tmp, [".jpg", ".jpeg", ".png"])
        test_src = find_dir_with_suffix(test_tmp, [".jpg", ".jpeg", ".png"])
        xml_src = find_dir_with_suffix(xml_tmp, [".xml"])

        if train_src is None or test_src is None or xml_src is None:
            raise FileNotFoundError("CCTSDB 解压结构不符合预期（未找到 train/test/xml）")

        move_tree(train_src, dst_dir / "images" / "train")
        move_tree(test_src, dst_dir / "images" / "test")
        move_tree(xml_src, dst_dir / "labels" / "xml")


def deploy_tt100k(src_dir: Path, dst_root: Path, tmp_base: Path) -> None:
    print("\n== 部署 TT100K ==")
    dst_dir = dst_root / "tt100k"
    dst_dir.mkdir(parents=True, exist_ok=True)

    tt_zip = src_dir / "tt100k_2021.zip"
    if not tt_zip.exists():
        raise FileNotFoundError(f"TT100K 压缩包不存在: {tt_zip}")

    tmp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp_dir:
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
        "--tmp-dir",
        default=None,
        help="临时解压目录（默认使用 dst-root 下的 .tmp）",
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
    tmp_base = Path(args.tmp_dir) if args.tmp_dir else (dst_root / ".tmp")
    datasets = normalize_datasets(args.datasets)

    if "bdd100k" in datasets:
        deploy_bdd100k(src_dir, dst_root, args.with_det20)
    if "cctsdb" in datasets:
        deploy_cctsdb(src_dir, dst_root, tmp_base)
    if "tt100k" in datasets:
        deploy_tt100k(src_dir, dst_root, tmp_base)

    print("\n✅ 数据集部署完成")


if __name__ == "__main__":
    main()
