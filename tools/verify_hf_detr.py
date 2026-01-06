#!/usr/bin/env python3
"""
验证HF DETR数据流修复
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("\n" + "="*60)
    print("🔍 验证HF DETR数据流修复")
    print("="*60 + "\n")
    
    all_pass = True
    
    # 1. 检查Dataset返回PIL图像
    print("1. 检查Dataset返回格式")
    print("-" * 60)
    
    dataset_file = project_root / 'dataset' / 'coco_dataset.py'
    with open(dataset_file) as f:
        dataset_content = f.read()
    
    dataset_checks = [
        ('def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:', '✓ Dataset返回PIL.Image'),
        ("'image_id': img_id,", '✓ 返回image_id'),
        ("'annotations': annotations,", '✓ 返回COCO格式annotations'),
        ("'bbox': ann['bbox'],", '✓ bbox保持xywh格式'),
        ("'category_id': ann['category_id'],", '✓ 包含category_id'),
        ('return None', '✓ make_transforms返回None（不做归一化）'),
        ('def collate_fn(batch: List[Tuple[Image.Image, Dict]])', '✓ collate_fn接受PIL图像'),
    ]
    
    for check, msg in dataset_checks:
        if check in dataset_content:
            print(msg)
        else:
            print(f"❌ 缺少: {msg}")
            all_pass = False
    
    print()
    
    # 2. 检查DETRModel.forward参数
    print("2. 检查DETRModel.forward")
    print("-" * 60)
    
    model_file = project_root / 'models' / 'detr_model.py'
    with open(model_file) as f:
        model_content = f.read()
    
    model_checks = [
        ('def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor = None, labels: list = None):', 'forward接受pixel_values和pixel_mask'),
        ('outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)', '训练时传递labels'),
        ('outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)', '推理时不传递labels'),
        ("f\"facebook/{model_config['name']}\"", '__init__中添加facebook/前缀'),
    ]
    
    for check, msg in model_checks:
        if check in model_content:
            print(msg)
        else:
            print(f"❌ 缺少: {msg}")
            all_pass = False
    
    print()
    
    # 3. 检查train_detr.py数据处理
    print("3. 检查train_detr.py数据流")
    print("-" * 60)
    
    train_file = project_root / 'tools' / 'train_detr.py'
    with open(train_file) as f:
        train_content = f.read()
    
    train_checks = [
        ("if not model_name.startswith('facebook/'):", '✓ ImageProcessor使用完整路径'),
        ('annotations = [t[\'annotations\'] for t in targets]', '✓ 提取COCO annotations'),
        ('encoding = image_processor(\n            images=images,\n            annotations=annotations,', '✓ 传递PIL images和annotations给processor'),
        ('labels = encoding[\'labels\']', '✓ 从encoding获取labels'),
        ('outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)', '✓ 使用正确的参数调用model'),
    ]
    
    for check, msg in train_checks:
        if check in train_content:
            print(msg)
        else:
            print(f"❌ 缺少: {msg}")
            all_pass = False
    
    print()
    
    # 4. 检查eval_detr.py使用post_process
    print("4. 检查eval_detr.py预测还原")
    print("-" * 60)
    
    eval_file = project_root / 'tools' / 'eval_detr.py'
    with open(eval_file) as f:
        eval_content = f.read()
    
    eval_checks = [
        ('encoding = image_processor(images=images, return_tensors=\'pt\')', 'processor处理PIL images'),
        ('target_sizes = torch.tensor([img.size[::-1] for img in images])', '获取原图尺寸'),
        ('processed_outputs = image_processor.post_process_object_detection(', '使用post_process还原预测'),
        ('threshold=score_threshold,', '传递score_threshold'),
        ('target_sizes=target_sizes', '传递target_sizes'),
        ('boxes = output[\'boxes\']', '从processed output获取boxes'),
        ('[x1, y1, x2 - x1, y2 - y1]', '转换为COCO xywh格式'),
    ]
    
    for check, msg in eval_checks:
        if check in eval_content:
            print(f"✓ {msg}")
        else:
            print(f"❌ 缺少: {msg}")
            all_pass = False
    
    print()
    
    # 5. 检查不应该存在的错误代码
    print("5. 检查已移除的错误代码")
    print("-" * 60)
    
    bad_patterns = [
        ('T.ToTensor()', 'dataset/coco_dataset.py', '❌ 仍在Dataset中做ToTensor'),
        ('T.Normalize(', 'dataset/coco_dataset.py', '❌ 仍在Dataset中做Normalize'),
        ('images_pil = [img.cpu().numpy()', 'tools/train_detr.py', '❌ 仍将tensor转numpy'),
        ('images_pil = [img.cpu().numpy()', 'tools/eval_detr.py', '❌ 仍将tensor转numpy'),
        ('def forward(self, images: torch.Tensor, targets: list = None):', 'models/detr_model.py', '❌ forward仍接受images参数'),
        ("'boxes': t['boxes']", 'tools/train_detr.py', '❌ 仍手动处理boxes'),
        ('logits = outputs.logits', 'tools/eval_detr.py', '❌ 仍手动解析logits'),
    ]
    
    files_content = {
        'dataset/coco_dataset.py': dataset_content,
        'tools/train_detr.py': train_content,
        'tools/eval_detr.py': eval_content,
        'models/detr_model.py': model_content,
    }
    
    for pattern, filename, msg in bad_patterns:
        content = files_content.get(filename, '')
        if pattern in content:
            print(msg)
            all_pass = False
        else:
            print(f"✓ 已移除: {pattern[:50]}...")
    
    print()
    
    print("="*60)
    if all_pass:
        print("🎉 所有HF DETR数据流修复验证通过!")
        print("="*60)
        print("\n关键改进:")
        print("1. ✅ Dataset返回PIL图像和COCO原始标注（未归一化）")
        print("2. ✅ DetrImageProcessor负责resize/pad/normalize")
        print("3. ✅ DETRModel.forward使用pixel_values和pixel_mask")
        print("4. ✅ 训练时processor将COCO annotations转为HF labels格式")
        print("5. ✅ 评估时使用post_process_object_detection还原预测")
        print("6. ✅ 所有路径使用完整的facebook/前缀")
        print("\n数据流:")
        print("  Dataset → PIL Image + COCO annotations")
        print("  ↓")
        print("  DetrImageProcessor → pixel_values + pixel_mask + labels")
        print("  ↓")
        print("  Model.forward(pixel_values, pixel_mask, labels)")
        print("  ↓")
        print("  post_process_object_detection → 还原到原图尺寸")
        print("\n下一步:")
        print("  python tools/syntax_check.py  # 语法检查")
    else:
        print("❌ 部分验证失败，请检查上述错误")
        print("="*60)
    print()
    
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
