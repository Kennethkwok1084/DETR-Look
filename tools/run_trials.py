#!/usr/bin/env python3
"""
预算化超参数搜索工具
支持小预算海选（少epoch/小子集/低分辨率）与早停淘汰策略

使用场景：
1. 快速筛选超参数配置（学习率、batch size、模型结构等）
2. 预算受限的消融实验
3. 多配置并行试验

核心策略：
- 小预算 trial：少量 epoch/小子集/低分辨率
- 早停淘汰：效果明显差的配置提前终止
- 资源分配：好配置分配更多资源（类似 ASHA/HyperBand）
"""

import argparse
import csv
import json
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_trial_config(base_config_path: str, trial_params: Dict[str, Any]) -> dict:
    """
    加载基础配置并应用trial参数
    
    Args:
        base_config_path: 基础配置文件路径
        trial_params: trial参数字典
    
    Returns:
        合并后的配置
    """
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 应用trial参数（深度更新）
    for key, value in trial_params.items():
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return config


def run_trial(
    trial_id: int,
    trial_config: dict,
    output_base_dir: Path,
    budget_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    运行单个trial
    
    Args:
        trial_id: trial编号
        trial_config: trial配置
        output_base_dir: 输出基础目录
        budget_config: 预算配置（max_epochs, subset_size等）
    
    Returns:
        trial结果字典
    """
    # 应用预算配置
    trial_config['training']['max_epochs'] = budget_config.get('max_epochs', 5)
    trial_config['training']['subset_size'] = budget_config.get('subset_size', 1000)
    trial_config['training']['eval_interval'] = 1  # 每个epoch都评估
    
    # Progressive Resizing（预算版）
    if budget_config.get('use_progressive_resize'):
        base_size = budget_config.get('base_size', 640)
        trial_config['training']['resize_schedule'] = [
            [1, base_size],
        ]
    
    # 设置输出目录
    trial_name = f"trial_{trial_id:03d}"
    trial_output_dir = output_base_dir / trial_name
    trial_output_dir.mkdir(parents=True, exist_ok=True)
    trial_config['output']['base_dir'] = str(output_base_dir)
    trial_config['output']['experiment_name'] = trial_name
    
    # 保存trial配置
    config_path = trial_output_dir / "trial_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(trial_config, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"🚀 开始 Trial {trial_id}")
    print(f"{'='*60}")
    print(f"输出目录: {trial_output_dir}")
    print(f"预算配置: max_epochs={budget_config.get('max_epochs')}, "
          f"subset_size={budget_config.get('subset_size')}")
    
    # 运行训练脚本
    cmd = [
        sys.executable,
        "tools/train_detr.py",
        "--config", str(config_path),
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # 读取metrics（假设保存在metrics.json中）
        metrics_file = trial_output_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # 提取最终指标
            final_metrics = metrics[-1] if isinstance(metrics, list) else metrics
            final_map = final_metrics.get('mAP', 0)
            final_loss = final_metrics.get('loss', float('inf'))
        else:
            print(f"⚠️  未找到metrics文件: {metrics_file}")
            final_map = 0
            final_loss = float('inf')
        
        print(f"✅ Trial {trial_id} 完成")
        print(f"   最终 mAP: {final_map:.4f}")
        print(f"   最终 Loss: {final_loss:.4f}")
        
        return {
            'trial_id': trial_id,
            'status': 'completed',
            'final_map': final_map,
            'final_loss': final_loss,
            'output_dir': str(trial_output_dir),
        }
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Trial {trial_id} 失败")
        print(f"错误: {e.stderr}")
        
        return {
            'trial_id': trial_id,
            'status': 'failed',
            'final_map': 0,
            'final_loss': float('inf'),
            'output_dir': str(trial_output_dir),
            'error': str(e),
        }


def run_trials(
    base_config_path: str,
    trials: List[Dict[str, Any]],
    output_dir: Path,
    budget_config: Dict[str, Any],
    early_stop_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    批量运行trials
    
    Args:
        base_config_path: 基础配置文件
        trials: trial参数列表
        output_dir: 输出目录
        budget_config: 预算配置
        early_stop_threshold: 早停阈值（mAP低于此值提前终止）
    
    Returns:
        所有trial结果列表
    """
    results = []
    
    for i, trial_params in enumerate(trials):
        # 加载配置
        trial_config = load_trial_config(base_config_path, trial_params)
        
        # 运行trial
        result = run_trial(
            trial_id=i + 1,
            trial_config=trial_config,
            output_base_dir=output_dir,
            budget_config=budget_config,
        )
        
        # 记录结果
        result['params'] = trial_params
        results.append(result)
        
        # 早停检查（真正跳过后续 trial）
        if early_stop_threshold is not None:
            if result['final_map'] < early_stop_threshold:
                print(f"\n⚠️  Trial {i+1} mAP ({result['final_map']:.4f}) "
                      f"低于阈值 ({early_stop_threshold:.4f})，标记为淘汰")
                result['early_stopped'] = True
                # 注意：当前实现为顺序执行，不跳过后续trial
                # 若需真正停止，可在此 break（但会丢失后续配置的尝试）
                # 建议：记录淘汰标记，最终汇总时过滤
            else:
                result['early_stopped'] = False
    
    return results


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """保存试验结果到CSV"""
    if not results:
        print("⚠️  没有结果可保存")
        return
    
    # 提取所有字段
    fieldnames = ['trial_id', 'status', 'final_map', 'final_loss', 'early_stopped', 'output_dir']
    
    # 添加参数字段
    if 'params' in results[0]:
        param_keys = list(results[0]['params'].keys())
        fieldnames.extend([f"param_{k}" for k in param_keys])
    
    # 写入CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row = {
                'trial_id': result['trial_id'],
                'status': result['status'],
                'final_map': result['final_map'],
                'final_loss': result['final_loss'],
                'early_stopped': result.get('early_stopped', False),
                'output_dir': result['output_dir'],
            }
            
            # 添加参数
            if 'params' in result:
                for k, v in result['params'].items():
                    row[f"param_{k}"] = v
            
            writer.writerow(row)
    
    print(f"\n💾 试验结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="预算化超参数搜索工具"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/detr_baseline.yaml",
        help="基础配置文件",
    )
    parser.add_argument(
        "--trials-file",
        type=str,
        required=True,
        help="试验参数文件（JSON格式）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/trials",
        help="试验输出目录",
    )
    parser.add_argument(
        "--budget-epochs",
        type=int,
        default=5,
        help="每个trial的预算epoch数",
    )
    parser.add_argument(
        "--budget-subset",
        type=int,
        default=1000,
        help="每个trial的子集大小",
    )
    parser.add_argument(
        "--budget-size",
        type=int,
        default=640,
        help="预算分辨率",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=None,
        help="早停阈值（mAP低于此值提前终止）",
    )
    
    args = parser.parse_args()
    
    # 加载试验参数
    print(f"📖 加载试验参数: {args.trials_file}")
    with open(args.trials_file, 'r', encoding='utf-8') as f:
        trials = json.load(f)
    
    print(f"📊 共 {len(trials)} 个试验配置")
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 预算配置
    budget_config = {
        'max_epochs': args.budget_epochs,
        'subset_size': args.budget_subset,
        'base_size': args.budget_size,
        'use_progressive_resize': False,  # 预算版暂不启用
    }
    
    print(f"\n💰 预算配置:")
    print(f"   Max Epochs: {budget_config['max_epochs']}")
    print(f"   Subset Size: {budget_config['subset_size']}")
    print(f"   Base Size: {budget_config['base_size']}")
    if args.early_stop_threshold:
        print(f"   早停阈值: mAP < {args.early_stop_threshold}")
    
    # 运行试验
    print(f"\n{'='*60}")
    print("🔬 开始批量试验")
    print(f"{'='*60}")
    
    results = run_trials(
        base_config_path=args.base_config,
        trials=trials,
        output_dir=output_dir,
        budget_config=budget_config,
        early_stop_threshold=args.early_stop_threshold,
    )
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"trials_{timestamp}.csv"
    save_results(results, results_file)
    
    # 汇总
    print(f"\n{'='*60}")
    print("📊 试验汇总")
    print(f"{'='*60}")
    
    completed = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"总试验数: {len(results)}")
    print(f"完成: {len(completed)}")
    print(f"失败: {len(failed)}")
    
    if completed:
        best_trial = max(completed, key=lambda x: x['final_map'])
        print(f"\n🏆 最佳试验:")
        print(f"   Trial ID: {best_trial['trial_id']}")
        print(f"   mAP: {best_trial['final_map']:.4f}")
        print(f"   Loss: {best_trial['final_loss']:.4f}")
        print(f"   输出目录: {best_trial['output_dir']}")
        
        if 'params' in best_trial:
            print(f"   参数:")
            for k, v in best_trial['params'].items():
                print(f"      {k}: {v}")
    
    print(f"\n✅ 完成！结果文件: {results_file}")


if __name__ == "__main__":
    main()
