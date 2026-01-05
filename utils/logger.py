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
    支持JSON和CSV格式输出
    """
    
    def __init__(self, output_dir: Path, experiment_name: str = "metrics"):
        """
        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.json_path = self.output_dir / f"{experiment_name}.json"
        self.csv_path = self.output_dir / f"{experiment_name}.csv"
        
        self.metrics_history = []
        self.csv_header_written = False
    
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
        self.metrics_history.append(record)
        
        # 保存JSON（追加模式）
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # 保存CSV
        self._write_csv(record)
    
    def _write_csv(self, record: Dict[str, Any]):
        """写入CSV文件"""
        import csv
        
        mode = 'w' if not self.csv_header_written else 'a'
        
        with open(self.csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            
            if not self.csv_header_written:
                writer.writeheader()
                self.csv_header_written = True
            
            writer.writerow(record)
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Dict[str, Any]:
        """
        获取最佳指标记录
        
        Args:
            metric_name: 指标名称
            mode: 'max' 或 'min'
        
        Returns:
            最佳记录字典
        """
        if not self.metrics_history:
            return {}
        
        if mode == 'max':
            best_record = max(self.metrics_history, key=lambda x: x.get(metric_name, float('-inf')))
        else:
            best_record = min(self.metrics_history, key=lambda x: x.get(metric_name, float('inf')))
        
        return best_record
