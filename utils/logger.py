#!/usr/bin/env python3
"""
日志工具
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    """
    设置logger
    
    Args:
        name: logger名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    
    Returns:
        Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """
    指标记录器
    支持JSON和CSV格式输出，支持Resume模式续写
    """
    
    def __init__(self, output_dir: Path, experiment_name: str = "metrics", resume: bool = False):
        """
        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
            resume: 是否Resume模式（加载已有指标）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.output_dir / f"{experiment_name}.json"
        self.csv_path = self.output_dir / f"{experiment_name}.csv"
        
        # 固定 CSV 列顺序（避免字段时有时无导致列漂移）
        # 注意：与训练实际产出对齐（mAP_50 而非 AP_50，并包含 lr）
        self.csv_fieldnames = ['step', 'epoch', 'loss', 'lr', 'mAP', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large']
        
        # Resume模式：加载已有指标
        self.metrics = []
        json_loaded = False
        if resume and self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    self.metrics = json.load(f)
                json_loaded = True
                print(f"📂 Resume: 已加载 {len(self.metrics)} 条历史指标")
            except Exception as e:
                print(f"⚠️  无法加载历史指标: {e}，从空列表开始")
                self.metrics = []
        
        # CSV 状态：Resume 时检查是否已有 CSV
        self.csv_header_written = False
        csv_exists = resume and self.csv_path.exists()
        if csv_exists:
            # 已有 CSV，设置为已写入 header（后续用 append 模式）
            self.csv_header_written = True
            print(f"📂 Resume: 将续写 CSV 文件")
        
        # 一致性检查：Resume 时 CSV 存在但 JSON 不存在（或加载失败）
        if resume and csv_exists and not json_loaded:
            print(f"⚠️  警告: CSV 存在但 JSON 缺失/损坏")
            print(f"    → CSV 将继续追加，但历史指标无法在 JSON 中体现")
            print(f"    → 建议检查 {self.json_path} 或手动恢复")
    
    def log(self, metrics: Dict[str, Any], step: int, epoch: int):
        """
        记录一组指标
        
        Args:
            metrics: 指标字典
            step: 当前步数/迭代数
            epoch: 当前epoch
        """
        record = {
            'step': step,
            'epoch': epoch,
            **metrics
        }
        self.metrics.append(record)
        
        # 保存JSON（完整覆盖）
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
        
        # 保存CSV
        self._write_csv(record)
    
    def _write_csv(self, record: Dict[str, Any]):
        """写入CSV文件（使用固定列顺序）"""
        import csv
        
        mode = 'w' if not self.csv_header_written else 'a'
        
        # 使用固定字段，缺失字段填充空字符串
        row = {field: record.get(field, '') for field in self.csv_fieldnames}
        
        with open(self.csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            
            if not self.csv_header_written:
                writer.writeheader()
                self.csv_header_written = True
            
            writer.writerow(row)
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Optional[Dict[str, Any]]:
        """
        获取最佳指标记录
        
        Args:
            metric_name: 指标名称
            mode: 'max' 或 'min'
        
        Returns:
            最佳记录字典，或 None
        """
        if not self.metrics:
            return None
        
        # 过滤出包含该指标的记录
        valid_records = [r for r in self.metrics if metric_name in r]
        if not valid_records:
            return None
        
        if mode == 'max':
            return max(valid_records, key=lambda x: x[metric_name])
        else:
            return min(valid_records, key=lambda x: x[metric_name])
