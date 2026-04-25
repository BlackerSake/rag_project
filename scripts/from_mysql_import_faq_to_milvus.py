"""
兼容入口：从 MySQL FAQ 导入 Milvus/Elasticsearch。

实际实现已统一收敛到项目根目录的 init.py。
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from init import import_mysql_faq_to_knowledge_base_only


if __name__ == "__main__":
    import_mysql_faq_to_knowledge_base_only()
