#!/usr/bin/env python3
"""
快速语法检查 - 不运行代码，只检查Python语法和导入结构
"""
import ast
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent

def check_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def main():
    """检查所有Python文件的语法"""
    print("="*60)
    print("🔍 Python语法检查")
    print("="*60)
    
    files_to_check = [
        'tools/train_detr.py',
        'tools/eval_detr.py',
        'tools/test_framework.py',
        'tools/verify_fixes.py',
        'dataset/coco_dataset.py',
        'models/detr_model.py',
        'utils/logger.py',
        'utils/checkpoint.py',
        'utils/metrics_logger.py',
    ]
    
    all_pass = True
    for file_path in files_to_check:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"⚠️  文件不存在: {file_path}")
            continue
            
        success, error = check_syntax(full_path)
        if success:
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path}: {error}")
            all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("🎉 所有文件语法检查通过!")
        print("="*60)
        print("\n准备在GPU服务器上运行:")
        print("1. pip install -r requirements.txt")
        print("2. python tools/verify_fixes.py")
        print("3. python tools/train_detr.py --config configs/detr_smoke.yaml")
    else:
        print("❌ 部分文件存在语法错误")
        print("="*60)
    print()
    
    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
