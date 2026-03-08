#!/usr/bin/env python3
"""
兼容入口：将旧的 eval_detr.py 调用收敛到 eval_unified.py。
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools import eval_unified

load_config = eval_unified.load_config


def evaluate(model, dataloader, device, coco_gt, logger, score_threshold=0.05, image_processor=None, config=None):
    """
    保留旧函数签名，内部转发到统一评估入口。
    """
    del image_processor
    if config is None:
        raise ValueError("兼容入口需要显式传入 config 参数")
    return eval_unified.evaluate(model, dataloader, device, coco_gt, logger, config, score_threshold)


def build_parser() -> argparse.ArgumentParser:
    """保留旧命令行接口。"""
    parser = argparse.ArgumentParser(description="评估 DETR 模型")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--eval-set", type=str, default="val", choices=["train", "val", "test"], help="评估数据集")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="检测置信度阈值（默认 0.05）")
    return parser


def main(argv=None):
    """兼容旧入口，实际评估逻辑交由 eval_unified.py。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.eval_set == "test":
        parser.error("统一评估入口当前仅支持 train/val；如需 test，请先扩展 tools/eval_unified.py")

    print("⚠️  eval_detr.py 已收敛为兼容入口，实际执行 tools/eval_unified.py")
    unified_args = argparse.Namespace(
        config=args.config,
        checkpoint=args.checkpoint,
        eval_set=args.eval_set,
        output=args.output,
        score_threshold=args.score_threshold,
    )
    return eval_unified.main(unified_args)


if __name__ == "__main__":
    main()
