"""
数据集统计分析脚本
生成论文所需的各类统计表、分布图和数据文件

功能：
1. train/val/test 划分统计（图像数、标注数、三类数量与占比）
2. 目标尺度分布（Small/Medium/Large，按COCO标准）
3. 每张图平均标注数、标注数分布
4. 导出 CSV/JSON 格式的统计数据
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple
import yaml

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class DatasetStatistics:
    """数据集统计分析类"""
    
    # COCO 标准的目标尺度划分（按 bbox 面积）
    SMALL_THRESHOLD = 32 * 32    # < 1024 pixels²
    MEDIUM_THRESHOLD = 96 * 96   # < 9216 pixels²
    
    def __init__(self, data_root: str, output_dir: str):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载类别配置
        config_path = self.data_root.parent.parent / "configs" / "classes.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        self.class_names = [config['COARSE_CLASSES'][i] for i in sorted(config['COARSE_CLASSES'].keys())]
        self.num_classes = len(self.class_names)
        
        print(f"类别配置: {self.class_names}")
        
    def load_coco_annotations(self, ann_file: Path) -> Dict:
        """加载 COCO 格式的标注文件"""
        print(f"加载标注文件: {ann_file}")
        with open(ann_file, 'r') as f:
            data = json.load(f)
        return data
    
    def compute_bbox_area(self, bbox: List[float]) -> float:
        """计算 bbox 面积（COCO 格式：[x, y, w, h]）"""
        return bbox[2] * bbox[3]
    
    def classify_size(self, area: float) -> str:
        """根据面积分类目标尺度"""
        if area < self.SMALL_THRESHOLD:
            return 'small'
        elif area < self.MEDIUM_THRESHOLD:
            return 'medium'
        else:
            return 'large'
    
    def analyze_split(self, ann_file: Path, split_name: str) -> Dict:
        """分析单个数据划分的统计信息"""
        data = self.load_coco_annotations(ann_file)
        
        # 基础统计
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        
        # 类别统计
        category_counts = Counter()
        size_counts = defaultdict(lambda: {'small': 0, 'medium': 0, 'large': 0})
        annotations_per_image = defaultdict(int)
        bbox_areas = []
        
        # 遍历所有标注
        for ann in data['annotations']:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            bbox = ann['bbox']
            
            # 统计类别
            category_counts[cat_id] += 1
            
            # 统计尺度
            area = self.compute_bbox_area(bbox)
            bbox_areas.append(area)
            size_class = self.classify_size(area)
            size_counts[cat_id][size_class] += 1
            
            # 统计每张图的标注数
            annotations_per_image[img_id] += 1
        
        # 计算类别占比
        category_stats = []
        for cat_id in range(self.num_classes):
            count = category_counts.get(cat_id, 0)
            ratio = count / num_annotations if num_annotations > 0 else 0
            category_stats.append({
                'category_id': cat_id,
                'category_name': self.class_names[cat_id],
                'count': count,
                'ratio': ratio
            })
        
        # 尺度分布统计
        size_dist = {'small': 0, 'medium': 0, 'large': 0}
        for cat_id in size_counts:
            for size in ['small', 'medium', 'large']:
                size_dist[size] += size_counts[cat_id][size]
        
        # 每张图标注数统计
        ann_counts = list(annotations_per_image.values())
        if not ann_counts:
            ann_counts = [0]
        
        # 构造统计结果
        stats = {
            'split': split_name,
            'num_images': num_images,
            'num_annotations': num_annotations,
            'avg_annotations_per_image': np.mean(ann_counts),
            'median_annotations_per_image': np.median(ann_counts),
            'max_annotations_per_image': max(ann_counts),
            'min_annotations_per_image': min(ann_counts),
            'category_stats': category_stats,
            'size_distribution': {
                'small': size_dist['small'],
                'medium': size_dist['medium'],
                'large': size_dist['large'],
                'small_ratio': size_dist['small'] / num_annotations if num_annotations > 0 else 0,
                'medium_ratio': size_dist['medium'] / num_annotations if num_annotations > 0 else 0,
                'large_ratio': size_dist['large'] / num_annotations if num_annotations > 0 else 0,
            },
            'bbox_areas': bbox_areas,
            'annotations_per_image': ann_counts
        }
        
        return stats
    
    def create_summary_table(self, stats_dict: Dict[str, Dict]) -> pd.DataFrame:
        """创建汇总统计表"""
        rows = []
        for split, stats in stats_dict.items():
            row = {
                '划分': split,
                '图像数': stats['num_images'],
                '标注数': stats['num_annotations'],
                '平均标注/图': f"{stats['avg_annotations_per_image']:.2f}",
            }
            # 添加各类别统计
            for cat_stat in stats['category_stats']:
                cat_name = cat_stat['category_name']
                row[f'{cat_name}_数量'] = cat_stat['count']
                row[f'{cat_name}_占比'] = f"{cat_stat['ratio']*100:.1f}%"
            
            # 添加尺度统计
            size_dist = stats['size_distribution']
            row['Small数量'] = size_dist['small']
            row['Small占比'] = f"{size_dist['small_ratio']*100:.1f}%"
            row['Medium数量'] = size_dist['medium']
            row['Medium占比'] = f"{size_dist['medium_ratio']*100:.1f}%"
            row['Large数量'] = size_dist['large']
            row['Large占比'] = f"{size_dist['large_ratio']*100:.1f}%"
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def plot_category_distribution(self, stats_dict: Dict[str, Dict], save_path: Path):
        """绘制类别分布对比图"""
        fig, axes = plt.subplots(1, len(stats_dict), figsize=(5*len(stats_dict), 5))
        if len(stats_dict) == 1:
            axes = [axes]
        
        for idx, (split, stats) in enumerate(stats_dict.items()):
            cat_names = [s['category_name'] for s in stats['category_stats']]
            cat_counts = [s['count'] for s in stats['category_stats']]
            
            axes[idx].bar(cat_names, cat_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            axes[idx].set_title(f'{split} - Category Distribution')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for i, v in enumerate(cat_counts):
                axes[idx].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存类别分布图: {save_path}")
    
    def plot_size_distribution(self, stats_dict: Dict[str, Dict], save_path: Path):
        """绘制尺度分布对比图"""
        fig, axes = plt.subplots(1, len(stats_dict), figsize=(5*len(stats_dict), 5))
        if len(stats_dict) == 1:
            axes = [axes]
        
        for idx, (split, stats) in enumerate(stats_dict.items()):
            size_dist = stats['size_distribution']
            sizes = ['Small\n(<32²)', 'Medium\n(32²-96²)', 'Large\n(>96²)']
            counts = [size_dist['small'], size_dist['medium'], size_dist['large']]
            ratios = [size_dist['small_ratio'], size_dist['medium_ratio'], size_dist['large_ratio']]
            
            bars = axes[idx].bar(sizes, counts, color=['#d62728', '#ff9896', '#2ca02c'])
            axes[idx].set_title(f'{split} - Size Distribution')
            axes[idx].set_ylabel('Count')
            
            # 添加百分比标签
            for i, (bar, ratio) in enumerate(zip(bars, ratios)):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{counts[i]}\n({ratio*100:.1f}%)',
                             ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存尺度分布图: {save_path}")
    
    def plot_bbox_area_histogram(self, stats_dict: Dict[str, Dict], save_path: Path):
        """绘制 bbox 面积分布直方图（对数坐标）"""
        fig, axes = plt.subplots(1, len(stats_dict), figsize=(6*len(stats_dict), 5))
        if len(stats_dict) == 1:
            axes = [axes]
        
        for idx, (split, stats) in enumerate(stats_dict.items()):
            areas = np.array(stats['bbox_areas'])
            log_areas = np.log10(areas + 1)  # 避免log(0)
            
            axes[idx].hist(log_areas, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].axvline(np.log10(self.SMALL_THRESHOLD), color='red', 
                            linestyle='--', label=f'Small threshold (32²)')
            axes[idx].axvline(np.log10(self.MEDIUM_THRESHOLD), color='orange', 
                            linestyle='--', label=f'Medium threshold (96²)')
            axes[idx].set_xlabel('log10(BBox Area + 1)')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{split} - BBox Area Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存bbox面积分布图: {save_path}")
    
    def plot_annotations_per_image(self, stats_dict: Dict[str, Dict], save_path: Path):
        """绘制每张图标注数分布（直方图+箱线图）"""
        n_splits = len(stats_dict)
        fig, axes = plt.subplots(2, n_splits, figsize=(5*n_splits, 10))
        if n_splits == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (split, stats) in enumerate(stats_dict.items()):
            ann_counts = stats['annotations_per_image']
            
            # 直方图
            axes[0, idx].hist(ann_counts, bins=30, color='skyblue', edgecolor='black')
            axes[0, idx].set_xlabel('Annotations per Image')
            axes[0, idx].set_ylabel('Frequency')
            axes[0, idx].set_title(f'{split} - Annotations per Image (Histogram)')
            axes[0, idx].axvline(np.mean(ann_counts), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(ann_counts):.2f}')
            axes[0, idx].legend()
            axes[0, idx].grid(True, alpha=0.3)
            
            # 箱线图
            bp = axes[1, idx].boxplot([ann_counts], labels=[split], patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgreen')
            axes[1, idx].set_ylabel('Annotations per Image')
            axes[1, idx].set_title(f'{split} - Annotations per Image (Box Plot)')
            axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存每图标注数分布图: {save_path}")
    
    def export_statistics_json(self, stats_dict: Dict[str, Dict], save_path: Path):
        """导出完整统计数据为 JSON"""
        # 移除 numpy array，转换为 list
        export_data = {}
        for split, stats in stats_dict.items():
            export_stats = stats.copy()
            export_stats['bbox_areas'] = [float(x) for x in stats['bbox_areas'][:100]]  # 只导出前100个示例
            export_stats['annotations_per_image'] = [int(x) for x in stats['annotations_per_image'][:100]]
            export_data[split] = export_stats
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 导出统计数据JSON: {save_path}")
    
    def run_analysis(self, dataset_name: str = 'bdd100k_det'):
        """运行完整的数据集分析"""
        dataset_path = self.data_root / dataset_name
        ann_dir = dataset_path / 'annotations'
        
        print(f"\n{'='*60}")
        print(f"开始分析数据集: {dataset_name}")
        print(f"{'='*60}\n")
        
        # 分析各个划分
        stats_dict = {}
        for split in ['train', 'val']:
            ann_file = ann_dir / f'instances_{split}.json'
            if ann_file.exists():
                stats_dict[split] = self.analyze_split(ann_file, split)
            else:
                print(f"⚠ 警告: 未找到 {ann_file}")
        
        if not stats_dict:
            print("❌ 错误: 没有找到任何标注文件")
            return
        
        # 创建输出目录
        output_subdir = self.output_dir / f'{dataset_name}_statistics'
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("生成统计表格和图表")
        print(f"{'='*60}\n")
        
        # 1. 创建汇总表
        summary_df = self.create_summary_table(stats_dict)
        csv_path = output_subdir / 'summary_statistics.csv'
        summary_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存汇总统计CSV: {csv_path}")
        print("\n汇总统计表:")
        print(summary_df.to_string(index=False))
        
        # 2. 绘制类别分布图
        self.plot_category_distribution(stats_dict, output_subdir / 'category_distribution.png')
        
        # 3. 绘制尺度分布图
        self.plot_size_distribution(stats_dict, output_subdir / 'size_distribution.png')
        
        # 4. 绘制 bbox 面积分布
        self.plot_bbox_area_histogram(stats_dict, output_subdir / 'bbox_area_histogram.png')
        
        # 5. 绘制每图标注数分布
        self.plot_annotations_per_image(stats_dict, output_subdir / 'annotations_per_image.png')
        
        # 6. 导出 JSON
        self.export_statistics_json(stats_dict, output_subdir / 'statistics.json')
        
        print(f"\n{'='*60}")
        print(f"✓ 分析完成！所有结果保存至: {output_subdir}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    data_root = project_root / 'data' / 'traffic_coco'
    output_dir = project_root / 'outputs' / 'dataset_analysis'
    
    # 创建统计分析器
    analyzer = DatasetStatistics(data_root=str(data_root), output_dir=str(output_dir))
    
    # 分析 BDD100K 数据集
    analyzer.run_analysis(dataset_name='bdd100k_det')
    
    # 如果有其他数据集，也可以分析
    # analyzer.run_analysis(dataset_name='cctsdb_det')
    # analyzer.run_analysis(dataset_name='tt100k_det')


if __name__ == '__main__':
    main()
