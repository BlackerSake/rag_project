import os
import sys
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from .state import State
from .schema import direct_answer_prompt, clarify_question_prompt, fallback_response_prompt, summarization_prompt, chat_response_prompt
from .models import model
from .intent_manager import get_intent_manager

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.knowledge_base import KnowledgeBase
from utils.logging_config import get_logger
from tools.manager import get_tool_by_intent

# 获取日志记录器
logger = get_logger(__name__)

# 初始化知识库
knowledge_base = KnowledgeBase()

# 获取意图管理器
intent_manager = None

async def get_manager():
    global intent_manager
    if intent_manager is None:
        intent_manager = await get_intent_manager()
    return intent_manager

async def detect_topic(state: State) -> State:
    """检测对话主题"""
    # 获取最新消息
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    
    logger.info(f"检测对话主题: {latest_message.content}")
    
    # 获取异步的意图管理器
    manager = await get_manager()
    
    # 意图匹配
    intent_id, score = await manager.match_intent(latest_message.content)
    logger.info(f"意图匹配: {intent_id} (相似度: {score:.4f})")
    # 阈值兜底逻辑
    if score < 0.35:
        intent_id = "D1" 
        logger.info(f"匹配分数 {score:.4f} 过低，归类为闲聊")
    
    return {
        "current_topic": intent_id,
        "intent_id": intent_id,
        "intent_score": score
    }

import re

def split_user_input(text: str) -> list[str]:
    """
    对用户原始输入做轻量句子拆分
    核心逻辑：切句 → 过滤补充句 → 合并残句
    """
    # 1. 按标点切句
    pattern = r'[。？！\?!；;]+'
    parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
    
    # 没切出多句，直接返回原始输入
    if len(parts) <= 1:
        return [text]
    
    # 2. 识别"补充句"模式（不能独立表达意图的句子）
    dependent_patterns = [
        r'^帮我.{0,4}(查|看|处理|确认|一下)',  # "帮我查一下" 无宾语
        r'^(麻烦|劳烦|请帮).{0,6}$',            # 纯礼貌用语
        r'^(好的|谢谢|知道了|明白|嗯|哦)',
        r'^\d{5,}$',                              # 纯数字（单号）
        r'^(我的|这个|那个).{0,4}$',             # 纯指代
        r'^.{1,4}$',                              # 极短片段
    ]
    
    result = []
    for part in parts:
        is_dependent = any(re.search(p, part) for p in dependent_patterns)
        if is_dependent:
            # 补充句合并到上一句
            if result:
                result[-1] = result[-1] + "，" + part
            # 如果是第一句就是补充句，合并到整体（罕见情况）
        else:
            result.append(part)
    
    return result if result else [text]


async def decompose_query(state: State) -> State:
    """查询拆解节点 - 直接对用户输入做句子拆分"""
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    user_query = latest_message.content
    
    logger.info(f"Decomposer Node: 分析查询 - {user_query}")
    
    try:
        # 第一步：直接对用户输入切句，不再检索向量库
        sentences = split_user_input(user_query)
        logger.info(f"句子拆分结果: {sentences}")
        
        # 第二步：对每个句子做意图匹配
        manager = await get_manager()
        sub_question_intents = []
        
        for i, sent in enumerate(sentences, 1):
            intent_id, score = await manager.match_intent(sent)
            sub_question_intents.append((sent, intent_id, score))
            logger.info(f"  子问题{i}: {sent} -> 意图: {intent_id} (相似度: {score:.4f})")
        
        # 第三步：按意图去重，同一意图只保留得分最高的那个句子
        best_per_intent = {}
        for sq, intent_id, score in sub_question_intents:
            if intent_id not in best_per_intent or score > best_per_intent[intent_id][2]:
                best_per_intent[intent_id] = (sq, intent_id, score)
        
        final_intents = list(best_per_intent.values())
        sub_questions = [item[0] for item in final_intents]
        
        is_complex = len(sub_questions) > 1
        logger.info(f"查询拆解结果: {len(sub_questions)} 个子问题")
        for i, sq in enumerate(sub_questions, 1):
            logger.info(f"  最终子问题{i}: {sq}")
        
        return {
            "sub_questions": sub_questions,
            "is_complex_query": is_complex,
            "multi_intent_results": final_intents,
            "decompose_skipped": False
        }
    
    except Exception as e:
        logger.error(f"查询拆解失败: {str(e)}")
        return {
            "sub_questions": [user_query],
            "is_complex_query": False,
            "multi_intent_results": [],
            "decompose_skipped": True
        }

async def retrieve_knowledge_multi(state: State) -> State:
    """多意图知识库检索节点
    
    对拆解后的每个子问题分别进行意图匹配和知识库检索
    """
    # 获取子问题和多意图结果
    sub_questions = state.get("sub_questions", [])
    multi_intent_results = state.get("multi_intent_results", [])
    
    if not sub_questions:
        # 如果没有子问题，直接返回空结果
        return {
            "knowledge_results": [],
            "highest_score": 0.0,
            "intent_id": None,
            "multi_intent_results": []
        }
    
    logger.info(f"多意图检索: 处理 {len(sub_questions)} 个子问题")
    
    multi_results = []
    
    # 使用从state中获取的多意图结果
    for i, (sub_q, intent_id, score) in enumerate(multi_intent_results):
        logger.info(f"处理子问题 {i+1}/{len(multi_intent_results)}: {sub_q}")
        logger.info(f"子问题意图匹配: {intent_id} (相似度: {score:.4f})")
        
        # 跳过D1意图的检索
        if intent_id == "D1":
            logger.info(f"子问题{i+1}为闲聊意图，跳过检索")
            continue
        
        # 知识库检索
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: knowledge_base.search(sub_q, k=5)
        )
        
        # 过滤与当前意图相关的结果
        filtered_results = []
        for doc, doc_score in results:
            metadata = doc.metadata
            if ('intent' in metadata and metadata['intent'] == intent_id) or \
               ('intent_id' in metadata and metadata['intent_id'] == intent_id):
                filtered_results.append((doc, doc_score))
        
        if not filtered_results:
            filtered_results = results[:3]  # 取前3个
        
        # 排序并取Top3
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:3]
        
        # 记录结果
        sub_result = {
            "sub_question": sub_q,
            "intent_id": intent_id,
            "intent_score": score,
            "knowledge_results": [(doc.page_content, doc_score) for doc, doc_score in sorted_results],
            "highest_score": sorted_results[0][1] if sorted_results else 0.0
        }
        multi_results.append(sub_result)
        
        logger.info(f"子问题{i+1}检索结果: {len(sorted_results)} 条记录，最高分数: {sub_result['highest_score']:.4f}")
    
    # 聚合所有结果
    all_knowledge = []
    highest_score = 0.0
    primary_intent = None
    
    for result in multi_results:
        all_knowledge.extend(result["knowledge_results"])
        if result["highest_score"] > highest_score:
            highest_score = result["highest_score"]
            primary_intent = result["intent_id"]
    
    # 去重并保持顺序
    seen = set()
    unique_knowledge = []
    for content, score in all_knowledge:
        if content not in seen:
            seen.add(content)
            unique_knowledge.append((content, score))
    
    # 取Top3
    final_knowledge = unique_knowledge[:3]
    
    logger.info(f"多意图检索完成: 共 {len(final_knowledge)} 条唯一记录，最高分数: {highest_score:.4f}")
    
    return {
        "knowledge_results": final_knowledge,
        "highest_score": highest_score,
        "intent_id": primary_intent,
        "current_topic": primary_intent,  # 添加current_topic字段，与intent_id保持一致
        "multi_intent_results": multi_results
    }

async def retrieve_knowledge(state: State) -> State:
    """检索知识库"""
    # 获取最新消息
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    
    logger.info(f"检索知识库: {latest_message.content}")
    
    # 获取意图ID
    intent_id = state.get("intent_id")
    logger.info(f"使用意图ID: {intent_id} 进行检索")
    
    # 获取需要查询知识库的子问题
    processed_results = state.get("processed_results", [])
    knowledge_questions = [result["sub_question"] for result in processed_results if not result.get("is_tool", False)]
    
    # 如果有需要查询知识库的子问题，使用第一个子问题进行检索
    query = knowledge_questions[0] if knowledge_questions else latest_message.content
    
    # 异步搜索知识库（不使用过滤表达式，避免字段不存在的问题）
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: knowledge_base.search(query, k=10)  # 搜索更多结果，以便后续过滤
    )
    
    # 根据意图ID进行过滤
    if intent_id == "D1":
        logger.info("意图为D1(闲聊)，跳过检索")
        # 直接返回空结果
        knowledge_results = []
        highest_score = 0.0
        logger.info("意图为D1,知识库不检索")
    else:
        # 过滤结果：只保留与当前意图相关的结果
        filtered_results = []
        for doc, score in results:
            # 检查文档的metadata中是否包含intent或intent_id字段，并且与当前意图匹配
            metadata = doc.metadata
            if ('intent' in metadata and metadata['intent'] == intent_id) or \
               ('intent_id' in metadata and metadata['intent_id'] == intent_id):
                filtered_results.append((doc, score))
        
        # 如果没有过滤结果，使用原始结果（避免无结果的情况）
        if not filtered_results:
            filtered_results = results
        
        # 按分数排序并提取前3条
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:3]
        # 保存文档内容和分数
        knowledge_results = [(result[0].page_content, result[1]) for result in sorted_results]
        # 提取最高分数
        highest_score = sorted_results[0][1] if sorted_results else 0.0
        
        # 记录检索结果的具体内容
        logger.info(f"知识库检索结果: {len(sorted_results)} 条记录，最高分数: {highest_score}")
        for i, (doc, score) in enumerate(sorted_results, 1):
            # 截取前100个字符作为预览
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"  Top{i} (分数: {score:.4f}): {content_preview}")
    
    return {
        "knowledge_results": knowledge_results,
        "highest_score": highest_score,
        "intent_id": intent_id
    }

async def direct_answer(state: State) -> State:
    """直接回答节点"""
    # 构建提示
    knowledge_context = "\n".join([item[0] for item in state["knowledge_results"]]) if state.get("knowledge_results") else "无相关知识"
    summary = state.get("summary", "无")
    
    # 获取任务分发的处理结果
    processed_results = state.get("processed_results", [])
    
    # 构建工具执行结果的上下文
    tool_results_context = ""
    if processed_results:
        tool_results_context = "\n工具执行结果:\n"
        for result in processed_results:
            if result.get("is_tool"):
                if result.get("result"):
                    tool_results_context += f"- 子问题: {result['sub_question']}\n  结果: {result['result']}\n"
                elif result.get("error"):
                    tool_results_context += f"- 子问题: {result['sub_question']}\n  错误: {result['error']}\n"
    
    logger.info(f"进入直接回答节点，主题: {state.get('current_topic', '未知')}")
    logger.info(f"处理结果数量: {len(processed_results)}")
    
    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break

    try:
        # 使用 prompt.invoke 调用 LLM，使用流式输出
        chain = direct_answer_prompt | model
        content = ""
        async for chunk in chain.astream({
            "messages": state["messages"],
            "knowledge_context": knowledge_context,
            "tool_results_context": tool_results_context,
            "summary": summary
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content
        
        # 添加到消息历史
        ai_message = AIMessage(content=content)
        
        logger.info(f"======直接回答生成: {content}...")
        
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": content, "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"直接回答生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": error_message, "topic": state.get("current_topic", "")}]
        }

async def clarify_question(state: State) -> State:
    """澄清提问节点"""
    # 构建提示
    if state["knowledge_results"]:
        top_knowledge = state["knowledge_results"][0][0]
        # 提取问题部分
        question_part = top_knowledge.split("\n")[0].replace("问题: ", "")
    else:
        question_part = "相关问题"
    
    # 获取用户问题
    user_question = state["messages"][-1].content if state["messages"] else ""
    summary = state.get("summary", "无")
    
    logger.info(f"进入澄清提问节点，用户问题: {user_question}, 相关问题: {question_part}")
    
    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break
    
    try:
        # 使用 prompt.invoke 调用 LLM，使用流式输出
        chain = clarify_question_prompt | model
        content = ""
        async for chunk in chain.astream({
            "user_question": user_question,
            "related_question": question_part,
            "summary": summary
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content
        
        # 添加到消息历史
        ai_message = AIMessage(content=content)
        
        logger.info(f"======澄清提问生成: {content}")
        
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": content, "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"澄清提问生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": error_message, "topic": state.get("current_topic", "")}]
        }

async def chat_response(state: State) -> State:
    """闲聊回复节点"""
    # 获取最新消息
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")
    
    logger.info(f"进入闲聊回复节点，用户消息: {latest_message.content}")
    
    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break
    
    try:
        # 使用从配置文件加载的闲聊提示模板
        chain = chat_response_prompt | model
        content = ""
        async for chunk in chain.astream({
            "user_question": latest_message.content
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content
        
        # 添加到消息历史
        ai_message = AIMessage(content=content)
        
        logger.info(f"======闲聊回复生成: {content}...")
        
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": content, "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"闲聊回复生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": error_message, "topic": state.get("current_topic", "")}]
        }

async def fallback_response(state: State) -> State:
    """兜底回复节点"""
    # 获取用户问题
    user_question = state["messages"][-1].content if state["messages"] else ""
    summary = state.get("summary", "无")
    
    logger.info(f"进入兜底回复节点，用户问题: {user_question}")
    
    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break
    
    try:
        # 使用 prompt.invoke 调用 LLM，使用流式输出
        chain = fallback_response_prompt | model
        content = ""
        async for chunk in chain.astream({
            "user_question": user_question,
            "summary": summary
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content
        
        # 添加到消息历史
        ai_message = AIMessage(content=content)
        
        logger.info(f"======兜底回复生成: {content}...")
        
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": content, "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"兜底回复生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": error_message, "topic": state.get("current_topic", "")}]
        }

async def summarize_conversation(state: State) -> State:
    """摘要压缩节点"""
    logger.info("进入摘要压缩节点")
    
    # 提取前10轮对话
    messages = state.get("messages", [])
    if len(messages) > 10:
        conversation_to_summarize = messages[:10]
    else:
        conversation_to_summarize = messages
    
    # 构建对话内容
    conversation_text = ""
    for msg in conversation_to_summarize:
        if isinstance(msg, HumanMessage):
            conversation_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conversation_text += f"助手: {msg.content}\n"
    
    try:
        # 生成总结，使用流式输出
        chain = summarization_prompt | model
        summary = ""
        async for chunk in chain.astream({"conversation": conversation_text}):
            # 从AIMessageChunk中提取content属性
            summary += chunk.content
        
        # 保留近5轮对话
        if len(messages) > 10:
            remaining_messages = messages[10:]
        else:
            remaining_messages = messages
        
        logger.info(f"对话总结生成: {summary}")
        logger.info(f"保留近5轮对话，共 {len(remaining_messages)} 条")
        
        return {
            "messages": remaining_messages,
            "summary": summary,
            "conversation_rounds": 0  # 重置对话轮数
        }
    except Exception as e:
        logger.error(f"对话总结生成失败: {str(e)}")
        # 发生异常时返回默认总结
        default_summary = "对话总结生成失败"
        return {
            "messages": messages,
            "summary": default_summary,
            "conversation_rounds": 0  # 重置对话轮数
        }

async def increment_rounds(state: State) -> State:
    """增加对话轮数"""
    current_rounds = state.get("conversation_rounds", 0)
    return {"conversation_rounds": current_rounds + 1}

async def task_dispatcher(state: State) -> State:
    """任务分发节点（分拣中心）
    
    1. 遍历子问题列表
    2. 对每个子问题查TOOL_MAP：命中则调API
    3. 没命中：查向量库
    4. 收集所有结果
    """
    # 获取子问题和它们的意图信息
    multi_intent_results = state.get("multi_intent_results", [])
    sub_questions = state.get("sub_questions", [])
    
    logger.info(f"任务分发：处理 {len(sub_questions)} 个子问题")
    
    # 收集每个子问题的处理结果
    processed_results = []
    
    # 遍历子问题
    for i, (sub_q, intent_id, score) in enumerate(multi_intent_results):
        logger.info(f"处理子问题 {i+1}/{len(multi_intent_results)}: {sub_q} (意图: {intent_id}, 相似度: {score:.4f})")
        
        # 检查是否为Tool意图
        tool = get_tool_by_intent(intent_id)
        if tool:
            logger.info(f"意图 {intent_id} 匹配到工具，执行工具调用")
            
            try:
                # 从子问题中提取物流单号
                # 这里简化处理，实际项目中可能需要更复杂的解析
                tracking_number = sub_q.strip()
                
                # 执行工具
                result = tool.execute(tracking_number=tracking_number)
                logger.info(f"工具执行结果: {result}")
                
                # 记录工具执行结果
                processed_results.append({
                    "sub_question": sub_q,
                    "intent_id": intent_id,
                    "is_tool": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"工具执行失败: {str(e)}")
                # 工具执行失败时，记录错误信息
                processed_results.append({
                    "sub_question": sub_q,
                    "intent_id": intent_id,
                    "is_tool": True,
                    "error": str(e)
                })
        else:
            logger.info(f"意图 {intent_id} 未匹配到工具，需要查询知识库")
            # 记录需要查询知识库的子问题
            processed_results.append({
                "sub_question": sub_q,
                "intent_id": intent_id,
                "is_tool": False
            })
    
    # 构建返回结果
    return {
        "processed_results": processed_results,
        "intent_id": multi_intent_results[0][1] if multi_intent_results else None
    }


