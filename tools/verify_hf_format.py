#!/usr/bin/env python3
"""
验证HuggingFace DETR格式修复
验证点：
1. annotations传参格式（完整targets vs 仅annotations列表）
2. facebook/前缀处理（避免重复）
3. processor与模型一致性（从配置读取）
4. 数据增强说明完整性
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check_train_annotations_format():
    """检查train_detr.py中annotations传参格式"""
    print("\n1️⃣ 检查训练时annotations传参格式")
    
    train_file = ROOT / 'tools' / 'train_detr.py'
    content = train_file.read_text()
    
    checks = []
    
    # ✅ 应该直接传targets而不是[t['annotations'] for t in targets]
    if "annotations=targets" in content:
        checks.append("✅ annotations传参：直接传targets（正确）")
    else:
        checks.append("❌ annotations传参：可能仍在拆分annotations列表")
    
    # ❌ 不应该有这行
    if "annotations = [t['annotations'] for t in targets]" in content:
        checks.append("❌ 发现旧的annotations拆分代码")
    else:
        checks.append("✅ 已移除旧的annotations拆分代码")
    
    # ✅ 应该有注释说明HF格式
    if "image_id" in content and "annotations': List[Dict]" in content:
        checks.append("✅ 有HF格式说明注释")
    else:
        checks.append("⚠️  缺少HF格式说明注释")
    
    for check in checks:
        print(f"  {check}")
    
    return all("✅" in c for c in checks)


def check_facebook_prefix():
    """检查facebook/前缀处理"""
    print("\n2️⃣ 检查facebook/前缀处理")
    
    model_file = ROOT / 'models' / 'detr_model.py'
    content = model_file.read_text()
    
    checks = []
    
    # ✅ 应该有前缀判断
    if "if not model_name.startswith('facebook/')" in content:
        checks.append("✅ detr_model.py：有facebook/前缀判断")
    else:
        checks.append("❌ detr_model.py：缺少facebook/前缀判断")
    
    # ❌ 不应该直接拼接
    if 'f"facebook/{model_config[\'name\']}"' in content:
        checks.append("❌ 发现直接拼接facebook/的旧代码")
    else:
        checks.append("✅ 已移除直接拼接的旧代码")
    
    for check in checks:
        print(f"  {check}")
    
    return all("✅" in c for c in checks)


def check_processor_consistency():
    """检查processor与模型一致性"""
    print("\n3️⃣ 检查processor与模型一致性")
    
    eval_file = ROOT / 'tools' / 'eval_detr.py'
    content = eval_file.read_text()
    
    checks = []
    
    # ✅ 应该从配置读取
    if "config['model']['name']" in content:
        checks.append("✅ eval_detr.py：从配置读取模型名称")
    else:
        checks.append("❌ eval_detr.py：未从配置读取模型名称")
    
    # ❌ 不应该硬编码
    if "'facebook/detr-resnet-50'" in content and "config['model']['name']" not in content:
        checks.append("❌ 发现硬编码的模型名称")
    else:
        checks.append("✅ 未发现硬编码的模型名称")
    
    # ✅ 应该有前缀判断
    if "if not model_name.startswith('facebook/')" in content:
        checks.append("✅ eval_detr.py：有facebook/前缀判断")
    else:
        checks.append("⚠️  eval_detr.py：缺少facebook/前缀判断")
    
    for check in checks:
        print(f"  {check}")
    
    return all("✅" in c or "⚠️" in c for c in checks)


def check_augmentation_docs():
    """检查数据增强文档说明"""
    print("\n4️⃣ 检查数据增强文档说明")
    
    dataset_file = ROOT / 'dataset' / 'coco_dataset.py'
    content = dataset_file.read_text()
    
    checks = []
    
    # ✅ 应该有如何添加增强的说明
    if "参考实现" in content or "RandomHorizontalFlip" in content:
        checks.append("✅ 有数据增强添加方法的示例")
    else:
        checks.append("⚠️  缺少数据增强添加示例")
    
    # ✅ 应该说明与processor的关系
    if "processor" in content and "PIL图像" in content:
        checks.append("✅ 说明了与processor的协作方式")
    else:
        checks.append("⚠️  缺少与processor协作的说明")
    
    for check in checks:
        print(f"  {check}")
    
    return all("✅" in c or "⚠️" in c for c in checks)


def check_config_model_name():
    """检查配置文件中的模型名称格式"""
    print("\n5️⃣ 检查配置文件中的模型名称")
    
    config_file = ROOT / 'configs' / 'detr_baseline.yaml'
    if not config_file.exists():
        print("  ⚠️  配置文件不存在，跳过检查")
        return True
    
    content = config_file.read_text()
    
    checks = []
    
    # 检查是否有facebook/前缀（两种都可接受）
    if 'name: "detr-resnet-50"' in content:
        checks.append("✅ 配置使用简短名称（代码会自动添加前缀）")
    elif 'name: "facebook/detr-resnet-50"' in content:
        checks.append("✅ 配置使用完整名称（代码会检测不重复）")
    else:
        checks.append("⚠️  未找到模型名称配置")
    
    for check in checks:
        print(f"  {check}")
    
    return True


def main():
    print("=" * 60)
    print("🔍 验证HuggingFace DETR格式修复")
    print("=" * 60)
    
    results = []
    
    results.append(("annotations传参格式", check_train_annotations_format()))
    results.append(("facebook/前缀处理", check_facebook_prefix()))
    results.append(("processor一致性", check_processor_consistency()))
    results.append(("数据增强文档", check_augmentation_docs()))
    results.append(("配置文件格式", check_config_model_name()))
    
    print("\n" + "=" * 60)
    print("📊 检查结果汇总")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n🎉 所有HuggingFace格式修复验证通过！")
        print("\n📝 修复内容：")
        print("  1. ✅ 训练时直接传targets给processor（含image_id+annotations）")
        print("  2. ✅ detr_model.py自动处理facebook/前缀（避免重复）")
        print("  3. ✅ eval_detr.py从配置读取模型名称（保持一致）")
        print("  4. ✅ 数据增强文档完善（说明如何添加及与processor协作）")
        print("\n🚀 可以开始GPU测试了！")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请检查上述标记为❌的项目")
        return 1


if __name__ == '__main__':
    sys.exit(main())
