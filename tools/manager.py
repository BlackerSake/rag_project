import os
import sys

# 添加当前目录到 Python 路径，解决导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logistics import LogisticsTool

# TOOL_MAP: 根据意图ID映射到对应的工具
# 目前只包括物流查询工具（A3）
TOOL_MAP = {
    "A3": LogisticsTool()
}


def get_tool_by_intent(intent_id: str):
    """根据意图ID获取对应的工具
    
    Args:
        intent_id: 意图ID
        
    Returns:
        BaseTool: 对应的工具实例，如果没有找到返回None
    """
    return TOOL_MAP.get(intent_id)
