
import sys
import io

# 确保标准输出支持中文，防止 print 时崩溃
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 导入日志配置
from utils.logging_config import get_logger

# 导入核心模块
from core import State, compiled_graph, model
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, Any

# 获取日志记录器
logger = get_logger(__name__)

async def run_chat(user_input: str, thread_id: str = "default") -> Dict[str, Any]:
    """运行聊天"""
    logger.info(f"开始聊天，用户输入: '{user_input}', thread_id: '{thread_id}'")
    
    # 创建用户消息
    user_message = HumanMessage(content=user_input)
    
    # 执行状态图，使用thread_id来标识对话线程
    result = await compiled_graph.ainvoke(
        {"messages": [user_message]},
        config={"configurable": {"thread_id": thread_id}}
    )
    
    # 返回最新的助手响应
    assistant_response = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            assistant_response = msg.content
            break
    
    logger.info(f"聊天结束，响应: '{assistant_response[:50]}...', 主题: '{result['current_topic']}', thread_id: '{thread_id}'")
    
    return {
        "response": assistant_response,
        "topic": result["current_topic"],
        "thread_id": thread_id
    }