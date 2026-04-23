import os
import sys
from .state import State

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger

# 获取日志记录器
logger = get_logger(__name__)

async def decide_next_node(state: State) -> str:
    """根据相似度分数决定下一个节点"""
    score = state.get("highest_score", 0.0)
    intent_id = state.get("intent_id")
    
    logger.info(f"决定下一个节点，最高分数: {score}, 意图ID: {intent_id}")
    
    # 特殊意图处理
    if intent_id == "D1":
        logger.info("特殊意图D1(寒暄闲聊)，跳转到闲聊型回复")
        return "chat_response"  # 跳转到闲聊回复节点
    elif intent_id == "D2":
        logger.info("特殊意图D2(人工介入)，跳转到兜底型回复")
        return "fallback_response"
    
    # 设定阈值
    high_threshold = 0.6
    low_threshold = 0.3
    
    # 根据分数进行分流决策
    if score >= high_threshold:
        logger.info(f"分数 {score} >= {high_threshold}，跳转到直接回答")
        return "direct_answer"
    elif high_threshold > score >= low_threshold:
        logger.info(f"分数 {low_threshold} <= {score} < {high_threshold}，跳转到澄清区间")
        return "clarify_question"
    else:
        logger.info(f"分数 {score} <= 0.4，跳转到兜底回复")
        return "fallback_response"

def should_end_conversation(state: State) -> bool:
    """判断是否结束对话"""
    # 简单判断：如果用户提到"结束"、"再见"等关键词，结束对话
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    end_keywords = ["结束", "再见", "谢谢", "不用了", "拜拜"]
    
    for keyword in end_keywords:
        if keyword in latest_message.content:
            logger.info(f"检测到结束关键词: {keyword}")
            return True
    
    return False

async def check_conversation_rounds(state: State) -> str:
    """检查对话轮数，决定是否需要总结"""
    # 计算对话轮数（每轮包含一个用户消息和一个助手消息）
    conversation_rounds = state.get("conversation_rounds", 0)
    
    logger.info(f"当前对话轮数: {conversation_rounds}")
    
    if conversation_rounds >= 15:
        logger.info("对话轮数超过15，触发总结")
        return "summarize"
    else:
        logger.info("对话轮数未达到15，继续对话")
        return "end"

async def route_after_decompose(state: State) -> str:
    """查询拆解后的路由函数
    
    根据查询拆解结果决定是进行单问题意图检测还是多意图检索
    
    Args:
        state: 当前状态
        
    Returns:
        str: "single_intent" 或 "multi_intent"
    """
    is_complex = state.get("is_complex_query", False)
    sub_questions = state.get("sub_questions", [])
    skipped = state.get("decompose_skipped", False)  # 是否跳过了查询拆解
    
    logger.info(f"查询拆解后路由: is_complex_query={is_complex}, 子问题数量={len(sub_questions)}, 跳过拆解={skipped}")
    
    # 如果跳过了查询拆解（单问题）或者只有一个子问题，使用单意图流程
    if skipped or len(sub_questions) <= 1:
        logger.info("单问题查询，进入意图检测节点")
        return "single_intent"
    else:
        # 多个子问题，使用多意图检索
        logger.info(f"复杂查询，共 {len(sub_questions)} 个子问题，进入多意图检索")
        return "multi_intent"

async def should_decompose(state: State) -> str:
    """判断是否为复杂查询，决定是否使用多意图检索（兼容旧版本）"""
    is_complex = state.get("is_complex_query", False)
    sub_questions = state.get("sub_questions", [])
    
    logger.info(f"判断查询类型: is_complex_query={is_complex}, 子问题数量={len(sub_questions)}")
    
    # 如果拆解出多个子问题，认为是复杂查询
    if is_complex and len(sub_questions) > 1:
        logger.info(f"检测到复杂查询，使用多意图检索，共 {len(sub_questions)} 个子问题")
        return "multi_intent"
    else:
        logger.info("简单查询，使用单意图检索")
        return "single_intent"

# 导入HumanMessage以避免循环导入
from langchain_core.messages import HumanMessage
