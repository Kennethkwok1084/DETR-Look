"""
模型性能测试脚本
测试预训练模型在不同配置下的性能指标

功能：
1. FPS 和单帧延迟测试（mean/p50/p95/p99）
2. 显存峰值测试
3. 多分辨率对比（720p vs 1080p）
4. 模型加载时间测试（首次加载 vs 缓存加载）
5. 导出性能数据为 CSV/JSON
"""

import torch
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import gc
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict

class PerformanceBenchmark:
    """模型性能基准测试"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
    
    def create_dummy_image(self, height: int, width: int) -> torch.Tensor:
        """创建虚拟测试图像"""
        # 创建随机图像 [3, H, W]
        img = torch.rand(3, height, width)
        return img
    
    def load_model(self, model_name: str = 'detr-resnet50'):
        """加载预训练模型"""
        print(f"\n加载模型: {model_name}")
        start_time = time.time()
        
        try:
            # 尝试加载 torchvision DETR
            from torchvision.models.detection import detr_resnet50
            model = detr_resnet50(pretrained=True, num_classes=91)  # COCO 91类
            model = model.to(self.device)
            model.eval()
            load_time = time.time() - start_time
            print(f"✓ 模型加载成功，耗时: {load_time:.2f}s")
            return model, load_time
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return None, 0
    
    def measure_inference_time(self, model, image: torch.Tensor, 
                              warmup: int = 10, iterations: int = 100) -> Dict:
        """测量推理时间"""
        image = image.unsqueeze(0).to(self.device)  # [1, 3, H, W]
        
        # Warmup
        print(f"  Warmup {warmup} 次...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(image)
        
        # 同步GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量推理时间
        print(f"  测量 {iterations} 次推理...")
        latencies = []
        
        with torch.no_grad():
            for i in range(iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.time()
                _ = model(image)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                latency = (time.time() - start) * 1000  # 转换为毫秒
                latencies.append(latency)
                
                if (i + 1) % 20 == 0:
                    print(f"    进度: {i+1}/{iterations}")
        
        # 计算统计指标
        latencies = np.array(latencies)
        stats = {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'std_ms': float(np.std(latencies)),
            'fps': float(1000.0 / np.mean(latencies))
        }
        
        return stats
    
    def measure_memory(self, model, image: torch.Tensor) -> Dict:
        """测量显存使用"""
        if not torch.cuda.is_available():
            return {'peak_memory_mb': 0, 'allocated_memory_mb': 0}
        
        image = image.unsqueeze(0).to(self.device)
        
        # 重置内存统计
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # 推理
        with torch.no_grad():
            _ = model(image)
        
        torch.cuda.synchronize()
        
        # 获取内存统计
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        return {
            'peak_memory_mb': float(peak_memory),
            'allocated_memory_mb': float(allocated_memory)
        }
    
    def benchmark_resolution(self, model, resolution_name: str, 
                           height: int, width: int) -> Dict:
        """对特定分辨率进行基准测试"""
        print(f"\n{'='*60}")
        print(f"测试分辨率: {resolution_name} ({width}x{height})")
        print(f"{'='*60}")
        
        # 创建测试图像
        image = self.create_dummy_image(height, width)
        print(f"图像尺寸: {image.shape}")
        
        # 测量推理时间
        time_stats = self.measure_inference_time(model, image)
        print(f"\n推理时间统计:")
        print(f"  平均: {time_stats['mean_ms']:.2f} ms")
        print(f"  中位数: {time_stats['median_ms']:.2f} ms")
        print(f"  P95: {time_stats['p95_ms']:.2f} ms")
        print(f"  P99: {time_stats['p99_ms']:.2f} ms")
        print(f"  FPS: {time_stats['fps']:.2f}")
        
        # 测量显存
        memory_stats = self.measure_memory(model, image)
        print(f"\n显存统计:")
        print(f"  峰值: {memory_stats['peak_memory_mb']:.2f} MB")
        print(f"  当前分配: {memory_stats['allocated_memory_mb']:.2f} MB")
        
        # 合并结果
        result = {
            'resolution': resolution_name,
            'width': width,
            'height': height,
            **time_stats,
            **memory_stats
        }
        
        return result
    
    def test_model_loading_cache(self, model_name: str = 'detr-resnet50', 
                                 num_loads: int = 3) -> Dict:
        """测试模型加载时间（首次 vs 缓存）"""
        print(f"\n{'='*60}")
        print(f"测试模型加载时间（{num_loads} 次加载）")
        print(f"{'='*60}\n")
        
        load_times = []
        
        for i in range(num_loads):
            # 清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"第 {i+1} 次加载...")
            _, load_time = self.load_model(model_name)
            load_times.append(load_time)
            
            # 第一次加载后等待一下
            if i == 0:
                time.sleep(2)
        
        return {
            'first_load_time_s': load_times[0],
            'subsequent_load_times_s': load_times[1:],
            'avg_subsequent_load_time_s': np.mean(load_times[1:]) if len(load_times) > 1 else 0,
            'speedup_ratio': load_times[0] / np.mean(load_times[1:]) if len(load_times) > 1 else 1.0
        }
    
    def run_full_benchmark(self):
        """运行完整的性能基准测试"""
        print(f"\n{'='*60}")
        print(f"DETR 模型性能基准测试")
        print(f"{'='*60}\n")
        
        all_results = {}
        
        # 1. 测试模型加载时间
        loading_stats = self.test_model_loading_cache()
        all_results['loading_times'] = loading_stats
        
        print(f"\n模型加载时间:")
        print(f"  首次加载: {loading_stats['first_load_time_s']:.2f}s")
        print(f"  后续平均: {loading_stats['avg_subsequent_load_time_s']:.2f}s")
        print(f"  加速比: {loading_stats['speedup_ratio']:.2f}x")
        
        # 2. 加载模型进行推理测试
        model, _ = self.load_model()
        if model is None:
            print("❌ 无法加载模型，跳过推理测试")
            return
        
        # 3. 测试不同分辨率
        resolutions = [
            ('720p', 720, 1280),
            ('1080p', 1080, 1920),
        ]
        
        resolution_results = []
        for res_name, height, width in resolutions:
            result = self.benchmark_resolution(model, res_name, height, width)
            resolution_results.append(result)
        
        all_results['resolution_benchmarks'] = resolution_results
        
        # 4. 保存结果
        self.save_results(all_results)
        
        # 5. 生成报告
        self.generate_report(all_results)
        
        print(f"\n{'='*60}")
        print(f"✓ 基准测试完成！结果保存至: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def save_results(self, results: Dict):
        """保存测试结果"""
        # 保存 JSON
        json_path = self.output_dir / 'performance_benchmark.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 保存 JSON: {json_path}")
        
        # 保存 CSV（分辨率对比表）
        if 'resolution_benchmarks' in results:
            df = pd.DataFrame(results['resolution_benchmarks'])
            csv_path = self.output_dir / 'resolution_comparison.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ 保存 CSV: {csv_path}")
    
    def generate_report(self, results: Dict):
        """生成性能报告"""
        report_path = self.output_dir / 'performance_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# DETR 模型性能基准测试报告\n\n")
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 设备信息
            f.write("## 测试环境\n\n")
            f.write(f"- 设备: {self.device}\n")
            if torch.cuda.is_available():
                f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
                f.write(f"- CUDA: {torch.version.cuda}\n")
            f.write(f"- PyTorch: {torch.__version__}\n\n")
            
            # 模型加载时间
            if 'loading_times' in results:
                f.write("## 模型加载时间\n\n")
                loading = results['loading_times']
                f.write(f"- 首次加载: **{loading['first_load_time_s']:.2f}s**\n")
                f.write(f"- 后续平均加载: **{loading['avg_subsequent_load_time_s']:.2f}s**\n")
                f.write(f"- 缓存加速比: **{loading['speedup_ratio']:.2f}x**\n\n")
                f.write("> 说明：后续加载受益于系统缓存和PyTorch模型权重缓存\n\n")
            
            # 分辨率对比
            if 'resolution_benchmarks' in results:
                f.write("## 不同分辨率性能对比\n\n")
                f.write("| 分辨率 | 尺寸 | 平均延迟(ms) | P95延迟(ms) | FPS | 显存峰值(MB) |\n")
                f.write("|--------|------|--------------|-------------|-----|-------------|\n")
                
                for res in results['resolution_benchmarks']:
                    f.write(f"| {res['resolution']} | "
                          f"{res['width']}x{res['height']} | "
                          f"{res['mean_ms']:.2f} | "
                          f"{res['p95_ms']:.2f} | "
                          f"{res['fps']:.2f} | "
                          f"{res['peak_memory_mb']:.2f} |\n")
                
                f.write("\n### 详细延迟统计\n\n")
                f.write("| 分辨率 | Mean | Median | P95 | P99 | Min | Max | Std |\n")
                f.write("|--------|------|--------|-----|-----|-----|-----|-----|\n")
                
                for res in results['resolution_benchmarks']:
                    f.write(f"| {res['resolution']} | "
                          f"{res['mean_ms']:.2f} | "
                          f"{res['median_ms']:.2f} | "
                          f"{res['p95_ms']:.2f} | "
                          f"{res['p99_ms']:.2f} | "
                          f"{res['min_ms']:.2f} | "
                          f"{res['max_ms']:.2f} | "
                          f"{res['std_ms']:.2f} |\n")
                
                f.write("\n")
            
            f.write("## 结论\n\n")
            f.write("1. **模型加载优化**：缓存机制显著减少了重复加载时间\n")
            f.write("2. **分辨率影响**：更高分辨率导致推理时间增加和显存消耗增大\n")
            f.write("3. **实时性能**：需要根据应用场景选择合适的输入分辨率\n")
        
        print(f"✓ 生成报告: {report_path}")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs' / 'performance_benchmark'
    
    benchmark = PerformanceBenchmark(output_dir=str(output_dir))
    benchmark.run_full_benchmark()


if __name__ == '__main__':
    main()
