"""
性能数据模板生成器
当无法运行实际性能测试时，生成预期的性能数据格式和模板
用于论文撰写时的占位符数据
"""

import json
import pandas as pd
from pathlib import Path
import time


def generate_performance_template():
    """生成性能数据模板"""
    
    # 模拟的性能数据（基于 DETR-ResNet50 的典型性能）
    template_data = {
        "test_info": {
            "note": "这是模板数据，实际数值需要运行 performance_benchmark.py 获取",
            "model": "DETR-ResNet50",
            "device": "GPU (需实测)",
            "test_date": time.strftime('%Y-%m-%d'),
        },
        "loading_times": {
            "first_load_time_s": 8.5,
            "subsequent_load_times_s": [2.3, 2.1, 2.2],
            "avg_subsequent_load_time_s": 2.2,
            "speedup_ratio": 3.86,
            "note": "首次加载包含模型下载和初始化，后续加载受益于缓存"
        },
        "resolution_benchmarks": [
            {
                "resolution": "720p",
                "width": 1280,
                "height": 720,
                "mean_ms": 85.3,
                "median_ms": 84.7,
                "p95_ms": 92.1,
                "p99_ms": 98.5,
                "min_ms": 78.2,
                "max_ms": 105.3,
                "std_ms": 4.8,
                "fps": 11.72,
                "peak_memory_mb": 1180.5,
                "allocated_memory_mb": 1050.2
            },
            {
                "resolution": "1080p",
                "width": 1920,
                "height": 1080,
                "mean_ms": 178.6,
                "median_ms": 176.3,
                "p95_ms": 195.8,
                "p99_ms": 208.2,
                "min_ms": 165.1,
                "max_ms": 225.6,
                "std_ms": 9.2,
                "fps": 5.60,
                "peak_memory_mb": 2385.7,
                "allocated_memory_mb": 2150.3
            }
        ]
    }
    
    return template_data


def create_performance_tables(data, output_dir):
    """创建性能对比表格"""
    
    # 1. 分辨率对比表
    res_data = []
    for res in data['resolution_benchmarks']:
        res_data.append({
            '分辨率': res['resolution'],
            '尺寸': f"{res['width']}×{res['height']}",
            '平均延迟(ms)': f"{res['mean_ms']:.1f}",
            'P95延迟(ms)': f"{res['p95_ms']:.1f}",
            'FPS': f"{res['fps']:.2f}",
            '显存峰值(MB)': f"{res['peak_memory_mb']:.1f}",
            '推理耗时比': f"{res['mean_ms'] / data['resolution_benchmarks'][0]['mean_ms']:.2f}x"
        })
    
    df_res = pd.DataFrame(res_data)
    csv_path = output_dir / 'resolution_comparison.csv'
    df_res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存分辨率对比表: {csv_path}")
    
    # 2. 详细延迟统计表
    latency_data = []
    for res in data['resolution_benchmarks']:
        latency_data.append({
            '分辨率': res['resolution'],
            'Mean(ms)': f"{res['mean_ms']:.2f}",
            'Median(ms)': f"{res['median_ms']:.2f}",
            'P95(ms)': f"{res['p95_ms']:.2f}",
            'P99(ms)': f"{res['p99_ms']:.2f}",
            'Min(ms)': f"{res['min_ms']:.2f}",
            'Max(ms)': f"{res['max_ms']:.2f}",
            'Std(ms)': f"{res['std_ms']:.2f}"
        })
    
    df_latency = pd.DataFrame(latency_data)
    csv_path = output_dir / 'latency_statistics.csv'
    df_latency.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存延迟统计表: {csv_path}")
    
    # 3. 模型加载时间对比表
    loading_data = [{
        '指标': '首次加载',
        '耗时(s)': data['loading_times']['first_load_time_s'],
        '备注': '包含模型下载和初始化'
    }, {
        '指标': '后续平均加载',
        '耗时(s)': data['loading_times']['avg_subsequent_load_time_s'],
        '备注': '受益于系统缓存'
    }, {
        '指标': '缓存加速比',
        '耗时(s)': data['loading_times']['speedup_ratio'],
        '备注': '首次/后续的比值'
    }]
    
    df_loading = pd.DataFrame(loading_data)
    csv_path = output_dir / 'model_loading_time.csv'
    df_loading.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 保存模型加载时间表: {csv_path}")
    
    return df_res, df_latency, df_loading


def create_performance_report(data, output_dir):
    """生成性能报告 Markdown"""
    
    report_path = output_dir / 'performance_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# DETR 模型性能基准测试报告\n\n")
        f.write("> ⚠️ **注意**: 这是基于典型DETR-ResNet50性能的模板数据\n")
        f.write("> 实际数值需要在 GPU 环境运行 `tools/performance_benchmark.py` 获取\n\n")
        
        f.write(f"**测试日期**: {data['test_info']['test_date']}\n")
        f.write(f"**模型**: {data['test_info']['model']}\n\n")
        
        f.write("---\n\n")
        
        # 1. 模型加载时间
        f.write("## 1. 模型加载时间\n\n")
        f.write("| 指标 | 耗时 | 说明 |\n")
        f.write("|------|------|------|\n")
        loading = data['loading_times']
        f.write(f"| 首次加载 | **{loading['first_load_time_s']:.2f}s** | 包含模型权重下载和初始化 |\n")
        f.write(f"| 后续加载（缓存） | **{loading['avg_subsequent_load_time_s']:.2f}s** | 受益于系统缓存和PyTorch缓存 |\n")
        f.write(f"| 加速比 | **{loading['speedup_ratio']:.2f}x** | 缓存优化效果 |\n\n")
        
        f.write("**论文意义**: 证明系统优化中的缓存策略有效性，支持 4.6 节的优化设计论述。\n\n")
        
        # 2. 分辨率性能对比
        f.write("## 2. 不同分辨率性能对比\n\n")
        f.write("### 2.1 核心指标对比\n\n")
        f.write("| 分辨率 | 尺寸 | 平均延迟 | P95延迟 | FPS | 显存峰值 |\n")
        f.write("|--------|------|----------|---------|-----|----------|\n")
        
        for res in data['resolution_benchmarks']:
            f.write(f"| **{res['resolution']}** | "
                   f"{res['width']}×{res['height']} | "
                   f"{res['mean_ms']:.1f} ms | "
                   f"{res['p95_ms']:.1f} ms | "
                   f"{res['fps']:.2f} | "
                   f"{res['peak_memory_mb']:.0f} MB |\n")
        
        f.write("\n**关键发现**:\n")
        ratio = data['resolution_benchmarks'][1]['mean_ms'] / data['resolution_benchmarks'][0]['mean_ms']
        f.write(f"- 1080p 推理时间是 720p 的 **{ratio:.2f}x**\n")
        fps_720 = data['resolution_benchmarks'][0]['fps']
        fps_1080 = data['resolution_benchmarks'][1]['fps']
        f.write(f"- 720p 可达 **{fps_720:.1f} FPS**，接近实时（>10 FPS）\n")
        f.write(f"- 1080p 仅 **{fps_1080:.1f} FPS**，难以满足实时性要求\n")
        mem_ratio = data['resolution_benchmarks'][1]['peak_memory_mb'] / data['resolution_benchmarks'][0]['peak_memory_mb']
        f.write(f"- 显存消耗随分辨率增加 **{mem_ratio:.2f}x**\n\n")
        
        # 3. 详细延迟统计
        f.write("### 2.2 延迟详细统计\n\n")
        f.write("| 分辨率 | Mean | Median | P95 | P99 | Min | Max | Std |\n")
        f.write("|--------|------|--------|-----|-----|-----|-----|-----|\n")
        
        for res in data['resolution_benchmarks']:
            f.write(f"| {res['resolution']} | "
                   f"{res['mean_ms']:.2f} | "
                   f"{res['median_ms']:.2f} | "
                   f"{res['p95_ms']:.2f} | "
                   f"{res['p99_ms']:.2f} | "
                   f"{res['min_ms']:.2f} | "
                   f"{res['max_ms']:.2f} | "
                   f"{res['std_ms']:.2f} |\n")
        
        f.write("\n**论文意义**: P95/P99 指标反映尾延迟，对实时系统的用户体验至关重要。\n\n")
        
        # 4. 结论与建议
        f.write("## 3. 结论与建议\n\n")
        f.write("### 性能分析\n\n")
        f.write("1. **分辨率权衡**:\n")
        f.write("   - 720p 在保证检测精度的同时可实现准实时推理\n")
        f.write("   - 1080p 虽然细节更丰富，但推理速度降低约50%\n\n")
        
        f.write("2. **实时性评估**:\n")
        f.write(f"   - 720p @ {fps_720:.1f} FPS：适合实时视频流处理\n")
        f.write(f"   - 1080p @ {fps_1080:.1f} FPS：适合离线分析或低帧率场景\n\n")
        
        f.write("3. **资源消耗**:\n")
        f.write(f"   - 720p 显存 ~{data['resolution_benchmarks'][0]['peak_memory_mb']:.0f} MB：适合消费级 GPU\n")
        f.write(f"   - 1080p 显存 ~{data['resolution_benchmarks'][1]['peak_memory_mb']:.0f} MB：需要中高端 GPU\n\n")
        
        f.write("4. **优化策略**:\n")
        f.write(f"   - 模型缓存可减少 {loading['speedup_ratio']:.1f}x 加载时间\n")
        f.write("   - 批处理推理可进一步提升吞吐量\n")
        f.write("   - 考虑使用 TensorRT/ONNX 优化推理速度\n\n")
        
        f.write("### 论文写作建议\n\n")
        f.write("**第4.6节 系统优化设计** 可引用：\n")
        f.write("- 分辨率对比表（720p vs 1080p）\n")
        f.write("- FPS 和延迟数据\n")
        f.write("- 显存消耗对比\n")
        f.write("- 模型加载优化的加速比\n\n")
        
        f.write("**第5章 实验与结果** 可讨论：\n")
        f.write("- 性能瓶颈分析（计算 vs 内存）\n")
        f.write("- 实时性与精度的权衡\n")
        f.write("- 不同硬件配置的适用场景\n\n")
        
        f.write("---\n\n")
        f.write("## 附录：如何获取实际数据\n\n")
        f.write("```bash\n")
        f.write("# 1. 安装 PyTorch（需要 CUDA 支持）\n")
        f.write("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\n\n")
        f.write("# 2. 运行性能测试\n")
        f.write("python tools/performance_benchmark.py\n\n")
        f.write("# 3. 查看结果\n")
        f.write("cat outputs/performance_benchmark/performance_report.md\n")
        f.write("```\n\n")
        
        f.write("测试完成后会生成：\n")
        f.write("- `performance_benchmark.json` - 完整性能数据（JSON格式）\n")
        f.write("- `resolution_comparison.csv` - 分辨率对比表（CSV格式）\n")
        f.write("- `performance_report.md` - 本报告（实际测试数据版本）\n")
    
    print(f"✓ 生成性能报告: {report_path}")


def main():
    """主函数"""
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs' / 'performance_benchmark_template'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("生成性能数据模板（占位符数据）")
    print("="*60)
    print()
    print("⚠️  注意：这是模板数据，实际数值需要在GPU环境运行")
    print("   tools/performance_benchmark.py 获取")
    print()
    
    # 1. 生成模板数据
    data = generate_performance_template()
    
    # 2. 保存 JSON
    json_path = output_dir / 'performance_benchmark_template.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存模板数据 JSON: {json_path}")
    
    # 3. 生成 CSV 表格
    print("\n生成对比表格...")
    df_res, df_latency, df_loading = create_performance_tables(data, output_dir)
    
    # 4. 生成报告
    print("\n生成性能报告...")
    create_performance_report(data, output_dir)
    
    # 5. 打印预览
    print("\n" + "="*60)
    print("分辨率对比预览")
    print("="*60)
    print(df_res.to_string(index=False))
    
    print("\n" + "="*60)
    print("模型加载时间预览")
    print("="*60)
    print(df_loading.to_string(index=False))
    
    print("\n" + "="*60)
    print(f"✓ 完成！模板文件保存至: {output_dir}")
    print("="*60)
    print("\n提示：要获取实际性能数据，请运行：")
    print("  python tools/performance_benchmark.py")


if __name__ == '__main__':
    main()
