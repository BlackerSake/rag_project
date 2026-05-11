"""状态图构建器"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import State
from .nodes import retrieve_knowledge, retrieve_knowledge_multi, confidence_gate, direct_answer, clarify_question, fallback_response, summarize_conversation, increment_rounds, chat_response, decompose_query, intent_gate_node, llm_rewrite, task_dispatcher
from .edges import check_conversation_rounds
from intent import route_after_intent_gate

# 创建状态图
graph = StateGraph(State)

# 添加节点
graph.add_node("decompose_query", decompose_query)  # 查询拆解节点（入口）
graph.add_node("intent_gate", intent_gate_node)  # 意图门控节点
graph.add_node("llm_rewrite", llm_rewrite)  # LLM查询重写节点
graph.add_node("task_dispatcher", task_dispatcher)  # 任务分发节点（分拣中心）
graph.add_node("retrieve_knowledge", retrieve_knowledge)  # 单意图检索
graph.add_node("retrieve_knowledge_multi", retrieve_knowledge_multi)  # 多意图检索节点
graph.add_node("confidence_gate", confidence_gate)  # 置信度門控節點
graph.add_node("direct_answer", direct_answer)  # 直接回答节点
graph.add_node("clarify_question", clarify_question)  # 澄清提问节点
graph.add_node("fallback_response", fallback_response)  # 兜底回复节点
graph.add_node("chat_response", chat_response)  # 闲聊回复节点
graph.add_node("summarize", summarize_conversation)  # 对话总结节点
graph.add_node("increment_rounds", increment_rounds)  # 增加对话轮数节点
graph.add_node("check_rounds", check_conversation_rounds)  # 检查对话轮数节点

# 添加边
# 新的入口：查询拆解
graph.set_entry_point("decompose_query")

# 查询拆解后，先进行意图门控，再进入LLM查询重写节点或澄清/兜底节点
graph.add_edge("decompose_query", "intent_gate")


def check_intent_gate_route(state: State) -> str:
    """根据IntentGate动作路由，保留当前查询重写链路。"""
    route = route_after_intent_gate(state)
    return "llm_rewrite" if route == "task_dispatcher" else route


graph.add_conditional_edges(
    "intent_gate",
    check_intent_gate_route,
    {
        "llm_rewrite": "llm_rewrite",
        "clarify_question": "clarify_question",
        "fallback_response": "fallback_response",
    }
)
graph.add_edge("llm_rewrite", "task_dispatcher")

# 任务分发后，根据是否需要查询知识库决定下一步
def check_need_knowledge(state: State) -> str:
    """检查是否需要查询知识库"""

    processed_results = state.get("processed_results", [])

    # 高優先級
    if any(result.get("needs_clarification", False) for result in processed_results):
        return "clarify_question"


    intent_id = state.get("intent_id")
    if intent_id == "D1":
        return "chat_response"
    if intent_id == "D2":
        return "fallback_response"

    knowledge_targets = [
        result for result in processed_results
        if not result.get("is_tool", False)
        and not result.get("needs_clarification", False)
        and result.get("sub_question")
    ]
    if not knowledge_targets:
        return "direct_answer"

    return "retrieve_knowledge_multi" if len(knowledge_targets) > 1 else "retrieve_knowledge"

graph.add_conditional_edges(
    "task_dispatcher",
    check_need_knowledge,
    {
        "retrieve_knowledge": "retrieve_knowledge",
        "retrieve_knowledge_multi": "retrieve_knowledge_multi",
        "direct_answer": "direct_answer",
        "clarify_question": "clarify_question",
        "chat_response": "chat_response",
        "fallback_response": "fallback_response"
    }
)

# 知識庫查詢後先進入置信度門控，再依決策路由。
graph.add_edge("retrieve_knowledge", "confidence_gate")
graph.add_edge("retrieve_knowledge_multi", "confidence_gate")


def route_after_confidence(state: State) -> str:
    """依 confidence gate 決策選擇回答或澄清。"""
    decision = state.get("confidence_decision")
    if decision in {"HIGH", "MEDIUM", None}:
        return "direct_answer"
    if decision == "LOW":
        return "clarify_question"
    return "direct_answer"


graph.add_conditional_edges(
    "confidence_gate",
    route_after_confidence,
    {
        "direct_answer": "direct_answer",
        "clarify_question": "clarify_question",
    }
)

# 每个回复节点后增加轮数并检查是否需要总结
graph.add_edge("direct_answer", "increment_rounds")
graph.add_edge("clarify_question", "increment_rounds")
graph.add_edge("fallback_response", "increment_rounds")
graph.add_edge("chat_response", "increment_rounds")

# 检查对话轮数
graph.add_conditional_edges(
    "increment_rounds",
    check_conversation_rounds,
    {
        "summarize": "summarize",
        "end": END
    }
)

# 总结后结束对话
graph.add_edge("summarize", END)

# 创建checkpointer
memory = MemorySaver()

# 编译状态图，使用checkpointer
compiled_graph = graph.compile(checkpointer=memory)
