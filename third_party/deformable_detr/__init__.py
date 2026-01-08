"""
Deformable DETR 第三方库
"""

# 将官方代码路径加入可导入范围
import sys
from pathlib import Path

_third_party_root = Path(__file__).parent
if str(_third_party_root) not in sys.path:
    sys.path.insert(0, str(_third_party_root))
