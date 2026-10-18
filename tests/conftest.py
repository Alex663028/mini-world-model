"""pytest 共享配置：把仓库根目录加入 sys.path，让 tests 能 import 项目模块。

项目是"脚本集合"而不是安装包（没有 setup.py / pyproject.toml），
所以测试需要显式补路径——和所有入口脚本的做法一致。
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
