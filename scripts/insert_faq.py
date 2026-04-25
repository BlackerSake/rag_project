"""
兼容入口：将 faq_data.json 同步到 MySQL。

实际实现已统一收敛到项目根目录的 init.py，避免连接配置和同步逻辑重复。
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from init import sync_mysql_faq_only


if __name__ == "__main__":
    sync_mysql_faq_only()
