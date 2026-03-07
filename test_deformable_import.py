#!/usr/bin/env python3
"""测试 Deformable DETR 导入"""

import sys
from pathlib import Path

# 添加 third_party/deformable_detr 到路径
deformable_path = Path(__file__).parent / "third_party" / "deformable_detr"
if str(deformable_path) not in sys.path:
    sys.path.insert(0, str(deformable_path))

print(f"Python: {sys.executable}")
print(f"sys.path[0]: {sys.path[0]}")

try:
    print("\n1. 测试导入 util.misc...")
    from util import misc
    print("   ✅ util.misc 成功")
    
    print("\n2. 测试导入 models.backbone...")
    from models import backbone
    print("   ✅ models.backbone 成功")
    
    print("\n3. 测试导入 models.matcher...")
    from models import matcher
    print("   ✅ models.matcher 成功")
    
    print("\n4. 测试导入 models.deformable_transformer...")
    from models import deformable_transformer
    print("   ✅ models.deformable_transformer 成功")
    
    print("\n5. 测试导入 models.deformable_detr...")
    from models import deformable_detr
    print("   ✅ models.deformable_detr 成功")
    
    print("\n✅ 所有模块导入成功！")
    
except Exception as e:
    print(f"\n❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
