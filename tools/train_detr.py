#!/usr/bin/env python3
"""
兼容入口：将旧的 train_detr.py 调用收敛到 train_unified.py。
"""

import argparse
import sys
import tempfile
from pathlib import Path

import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.train_unified import main as unified_main


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_temp_config(config: dict) -> Path:
    """将兼容层覆写后的配置写入临时文件。"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(config, tmp, allow_unicode=True, sort_keys=False)
        return Path(tmp.name)


def build_parser() -> argparse.ArgumentParser:
    """保留旧命令行参数，内部委托给统一训练入口。"""
    parser = argparse.ArgumentParser(description="训练 Deformable DETR 主线模型用于交通场景目标检测")
    parser.add_argument("--config", type=str, default="configs/deformable_detr_baseline.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（覆盖配置文件中的设置）")
    parser.add_argument("--max-iter", type=int, default=None, help="最大迭代次数（用于冒烟测试）")
    parser.add_argument("--eval-interval", type=int, default=None, help="评估间隔")
    parser.add_argument("--save-interval", type=int, default=None, help="保存间隔")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    parser.add_argument("--subset-size", type=int, default=None, help="子集大小（用于快速验证或小样本过拟合）")
    parser.add_argument("--overfit", action="store_true", help="过拟合模式（关闭数据增强，固定随机种子）")
    return parser


def main(argv=None):
    """兼容旧入口，实际训练逻辑交由 train_unified.py。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    print("⚠️  train_detr.py 已收敛为兼容入口，实际执行 tools/train_unified.py")

    config = load_config(args.config)
    training = config.setdefault("training", {})

    if args.max_iter is not None:
        training["max_iters"] = args.max_iter
    if args.eval_interval is not None:
        training["eval_interval"] = args.eval_interval
    if args.save_interval is not None:
        training["save_interval"] = args.save_interval
    if args.subset_size is not None:
        training["subset_size"] = args.subset_size
    if args.overfit:
        training["overfit"] = True

    temp_config = dump_temp_config(config)
    try:
        unified_args = argparse.Namespace(
            config=str(temp_config),
            output_dir=args.output_dir,
            resume=args.resume,
        )
        return unified_main(unified_args)
    finally:
        temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
