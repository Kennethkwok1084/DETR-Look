"""
检测结果接口数据结构和样例生成

生成标准化的检测结果 Schema 和示例数据文件
用于系统接口设计和前后端交互
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import random


class DetectionResultSchema:
    """检测结果数据结构定义"""
    
    @staticmethod
    def get_schema_definition() -> Dict:
        """获取检测结果 Schema 定义"""
        schema = {
            "schema_version": "1.0",
            "description": "交通场景目标检测结果数据结构",
            "detection_result": {
                "type": "object",
                "properties": {
                    "frame_id": {
                        "type": "string",
                        "description": "帧ID或图像ID，唯一标识",
                        "example": "frame_000123"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "ISO8601",
                        "description": "检测时间戳",
                        "example": "2026-01-12T10:30:45.123Z"
                    },
                    "image_info": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "integer", "description": "图像宽度(像素)"},
                            "height": {"type": "integer", "description": "图像高度(像素)"},
                            "file_name": {"type": "string", "description": "图像文件名"}
                        }
                    },
                    "detections": {
                        "type": "array",
                        "description": "检测到的目标列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "detection_id": {
                                    "type": "integer",
                                    "description": "检测框ID（当前帧内唯一）"
                                },
                                "track_id": {
                                    "type": "integer",
                                    "description": "跟踪ID（跨帧唯一，-1表示未跟踪）"
                                },
                                "class_id": {
                                    "type": "integer",
                                    "description": "类别ID（0:vehicle, 1:traffic_sign, 2:traffic_light）"
                                },
                                "class_name": {
                                    "type": "string",
                                    "description": "类别名称",
                                    "enum": ["vehicle", "traffic_sign", "traffic_light"]
                                },
                                "confidence": {
                                    "type": "float",
                                    "description": "置信度分数 [0.0, 1.0]",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "bbox": {
                                    "type": "object",
                                    "description": "边界框坐标",
                                    "properties": {
                                        "x": {"type": "float", "description": "左上角x坐标"},
                                        "y": {"type": "float", "description": "左上角y坐标"},
                                        "width": {"type": "float", "description": "框宽度"},
                                        "height": {"type": "float", "description": "框高度"}
                                    }
                                },
                                "bbox_normalized": {
                                    "type": "object",
                                    "description": "归一化边界框坐标 [0.0, 1.0]",
                                    "properties": {
                                        "x": {"type": "float", "minimum": 0.0, "maximum": 1.0},
                                        "y": {"type": "float", "minimum": 0.0, "maximum": 1.0},
                                        "width": {"type": "float", "minimum": 0.0, "maximum": 1.0},
                                        "height": {"type": "float", "minimum": 0.0, "maximum": 1.0}
                                    }
                                }
                            },
                            "required": ["detection_id", "class_id", "class_name", "confidence", "bbox"]
                        }
                    },
                    "inference_info": {
                        "type": "object",
                        "description": "推理元信息",
                        "properties": {
                            "model_name": {"type": "string", "description": "模型名称"},
                            "inference_time_ms": {"type": "float", "description": "推理耗时(毫秒)"},
                            "num_detections": {"type": "integer", "description": "检测数量"},
                            "confidence_threshold": {"type": "float", "description": "置信度阈值"}
                        }
                    }
                },
                "required": ["frame_id", "timestamp", "image_info", "detections"]
            }
        }
        return schema
    
    @staticmethod
    def generate_sample_detection(frame_id: str, num_detections: int = 5) -> Dict:
        """生成单帧检测结果样例"""
        class_names = ["vehicle", "traffic_sign", "traffic_light"]
        
        detections = []
        for i in range(num_detections):
            class_id = random.randint(0, 2)
            x = random.uniform(0, 1280)
            y = random.uniform(0, 720)
            w = random.uniform(30, 200)
            h = random.uniform(30, 200)
            
            detection = {
                "detection_id": i,
                "track_id": random.randint(1, 100) if random.random() > 0.3 else -1,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "confidence": round(random.uniform(0.5, 0.99), 3),
                "bbox": {
                    "x": round(x, 2),
                    "y": round(y, 2),
                    "width": round(w, 2),
                    "height": round(h, 2)
                },
                "bbox_normalized": {
                    "x": round(x / 1280, 4),
                    "y": round(y / 720, 4),
                    "width": round(w / 1280, 4),
                    "height": round(h / 720, 4)
                }
            }
            detections.append(detection)
        
        result = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat() + "Z",
            "image_info": {
                "width": 1280,
                "height": 720,
                "file_name": f"{frame_id}.jpg"
            },
            "detections": detections,
            "inference_info": {
                "model_name": "DETR-ResNet50",
                "inference_time_ms": round(random.uniform(80, 150), 2),
                "num_detections": num_detections,
                "confidence_threshold": 0.5
            }
        }
        
        return result
    
    @staticmethod
    def generate_batch_samples(num_frames: int = 20) -> List[Dict]:
        """生成多帧检测结果样例"""
        samples = []
        for i in range(num_frames):
            frame_id = f"frame_{i:06d}"
            num_detections = random.randint(3, 10)
            sample = DetectionResultSchema.generate_sample_detection(frame_id, num_detections)
            samples.append(sample)
        return samples
    
    @staticmethod
    def export_to_csv(samples: List[Dict], output_path: Path):
        """导出为 CSV 格式（扁平化结构）"""
        rows = []
        for sample in samples:
            frame_id = sample['frame_id']
            timestamp = sample['timestamp']
            inference_time = sample['inference_info']['inference_time_ms']
            
            for det in sample['detections']:
                row = {
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'detection_id': det['detection_id'],
                    'track_id': det.get('track_id', -1),
                    'class_id': det['class_id'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'bbox_x': det['bbox']['x'],
                    'bbox_y': det['bbox']['y'],
                    'bbox_width': det['bbox']['width'],
                    'bbox_height': det['bbox']['height'],
                    'bbox_norm_x': det['bbox_normalized']['x'],
                    'bbox_norm_y': det['bbox_normalized']['y'],
                    'bbox_norm_width': det['bbox_normalized']['width'],
                    'bbox_norm_height': det['bbox_normalized']['height'],
                    'inference_time_ms': inference_time
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✓ 导出 CSV: {output_path} ({len(df)} 条检测记录)")
        return df
    
    @staticmethod
    def export_to_json(samples: List[Dict], output_path: Path, pretty: bool = True):
        """导出为 JSON 格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            else:
                json.dump(samples, f, ensure_ascii=False)
        print(f"✓ 导出 JSON: {output_path} ({len(samples)} 帧)")


def generate_api_documentation(output_dir: Path):
    """生成 API 文档"""
    doc_path = output_dir / 'API_SCHEMA.md'
    
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# 检测结果 API 数据结构文档\n\n")
        f.write("## 概述\n\n")
        f.write("本文档定义了交通场景目标检测系统的结果数据格式。\n\n")
        
        f.write("## 数据结构\n\n")
        f.write("### DetectionResult\n\n")
        f.write("检测结果的顶层结构。\n\n")
        
        f.write("```json\n")
        schema = DetectionResultSchema.get_schema_definition()
        f.write(json.dumps(schema, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
        
        f.write("## 字段说明\n\n")
        f.write("### frame_id\n")
        f.write("- **类型**: String\n")
        f.write("- **描述**: 帧的唯一标识符，格式为 `frame_XXXXXX`\n")
        f.write("- **示例**: `frame_000123`\n\n")
        
        f.write("### timestamp\n")
        f.write("- **类型**: String (ISO8601)\n")
        f.write("- **描述**: 检测时间戳，使用 ISO8601 格式\n")
        f.write("- **示例**: `2026-01-12T10:30:45.123Z`\n\n")
        
        f.write("### detections\n")
        f.write("- **类型**: Array<Detection>\n")
        f.write("- **描述**: 当前帧检测到的所有目标列表\n\n")
        
        f.write("#### Detection 对象\n\n")
        f.write("| 字段 | 类型 | 描述 | 示例 |\n")
        f.write("|------|------|------|------|\n")
        f.write("| detection_id | Integer | 检测框ID（帧内唯一） | 0 |\n")
        f.write("| track_id | Integer | 跟踪ID（跨帧唯一，-1表示未跟踪） | 42 |\n")
        f.write("| class_id | Integer | 类别ID (0/1/2) | 0 |\n")
        f.write("| class_name | String | 类别名称 | \"vehicle\" |\n")
        f.write("| confidence | Float | 置信度 [0.0, 1.0] | 0.95 |\n")
        f.write("| bbox | BBox | 边界框坐标 | {x, y, width, height} |\n")
        f.write("| bbox_normalized | BBox | 归一化边界框 [0.0, 1.0] | {x, y, width, height} |\n\n")
        
        f.write("### 类别映射\n\n")
        f.write("| class_id | class_name | 说明 |\n")
        f.write("|----------|------------|------|\n")
        f.write("| 0 | vehicle | 交通工具（车、公交、卡车等） |\n")
        f.write("| 1 | traffic_sign | 交通标志/路牌 |\n")
        f.write("| 2 | traffic_light | 红绿灯 |\n\n")
        
        f.write("## 示例数据\n\n")
        f.write("### 单帧检测结果\n\n")
        f.write("```json\n")
        sample = DetectionResultSchema.generate_sample_detection("frame_000001", 3)
        f.write(json.dumps(sample, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
        
        f.write("## 数据格式\n\n")
        f.write("### JSON 格式\n")
        f.write("用于 RESTful API 响应和实时数据交换。\n\n")
        
        f.write("### CSV 格式\n")
        f.write("用于批量分析和数据统计，采用扁平化结构。\n\n")
        f.write("**CSV 列定义**:\n\n")
        f.write("```\n")
        f.write("frame_id, timestamp, detection_id, track_id, class_id, class_name, confidence,\n")
        f.write("bbox_x, bbox_y, bbox_width, bbox_height,\n")
        f.write("bbox_norm_x, bbox_norm_y, bbox_norm_width, bbox_norm_height,\n")
        f.write("inference_time_ms\n")
        f.write("```\n\n")
        
        f.write("## 使用场景\n\n")
        f.write("1. **实时检测**：WebSocket/REST API 返回 JSON 格式\n")
        f.write("2. **批量处理**：离线分析使用 CSV 格式\n")
        f.write("3. **前端可视化**：解析 JSON 绘制检测框\n")
        f.write("4. **数据分析**：使用 Pandas 加载 CSV 进行统计\n")
    
    print(f"✓ 生成 API 文档: {doc_path}")


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'outputs' / 'api_schema'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("生成检测结果数据结构和样例")
    print("="*60)
    print()
    
    # 1. 保存 Schema 定义
    schema = DetectionResultSchema.get_schema_definition()
    schema_path = output_dir / 'detection_result_schema.json'
    with open(schema_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存 Schema 定义: {schema_path}")
    
    # 2. 生成样例数据（20帧）
    print("\n生成样例数据 (20 帧)...")
    samples = DetectionResultSchema.generate_batch_samples(num_frames=20)
    
    # 3. 导出为 JSON
    json_path = output_dir / 'detection_samples.json'
    DetectionResultSchema.export_to_json(samples, json_path)
    
    # 4. 导出为 CSV
    csv_path = output_dir / 'detection_samples.csv'
    df = DetectionResultSchema.export_to_csv(samples, csv_path)
    
    # 5. 生成 CSV 预览（前20行）
    preview_path = output_dir / 'detection_samples_preview.csv'
    df.head(20).to_csv(preview_path, index=False, encoding='utf-8-sig')
    print(f"✓ 导出 CSV 预览（前20行）: {preview_path}")
    
    # 6. 生成 API 文档
    generate_api_documentation(output_dir)
    
    # 7. 打印统计信息
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    print(f"总帧数: {len(samples)}")
    print(f"总检测数: {len(df)}")
    print(f"\n类别分布:")
    print(df['class_name'].value_counts())
    print(f"\n平均置信度: {df['confidence'].mean():.3f}")
    print(f"平均推理时间: {df.groupby('frame_id')['inference_time_ms'].first().mean():.2f} ms")
    
    print("\n" + "="*60)
    print(f"✓ 完成！所有文件保存至: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
