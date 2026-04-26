from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import State
from .nodes import retrieve_knowledge, retrieve_knowledge_multi, direct_answer, clarify_question, fallback_response, summarize_conversation, increment_rounds, chat_response, decompose_query, task_dispatcher
from .edges import check_conversation_rounds

# 创建状态图
graph = StateGraph(State)

# 添加节点
graph.add_node("decompose_query", decompose_query)  # 查询拆解节点（入口）
graph.add_node("task_dispatcher", task_dispatcher)  # 任务分发节点（分拣中心）
graph.add_node("retrieve_knowledge", retrieve_knowledge)  # 单意图检索
graph.add_node("retrieve_knowledge_multi", retrieve_knowledge_multi)  # 多意图检索节点

graph.add_node("direct_answer", direct_answer)
graph.add_node("clarify_question", clarify_question)
graph.add_node("fallback_response", fallback_response)
graph.add_node("chat_response", chat_response)

graph.add_node("summarize", summarize_conversation)
graph.add_node("increment_rounds", increment_rounds)
graph.add_node("check_rounds", check_conversation_rounds)

# 添加边
# 新的入口：查询拆解
graph.set_entry_point("decompose_query")

# 查询拆解后，直接进入任务分发节点
graph.add_edge("decompose_query", "task_dispatcher")

# 任务分发后，根据是否需要查询知识库决定下一步
# 检查是否有需要查询知识库的子问题
def check_need_knowledge(state: State) -> str:
    intent_id = state.get("intent_id")
    if intent_id == "D1":
        return "chat_response"
    if intent_id == "D2":
        return "fallback_response"

    processed_results = state.get("processed_results", [])
    # 检查是否有子问题需要查询知识库
    need_knowledge = any(not result.get("is_tool", False) for result in processed_results)
    if need_knowledge:
        # 检查是否为多意图
        is_complex = state.get("is_complex_query", False)
        return "retrieve_knowledge_multi" if is_complex else "retrieve_knowledge"
    else:
        return "direct_answer"

graph.add_conditional_edges(
    "task_dispatcher",
    check_need_knowledge,
    {
        "retrieve_knowledge": "retrieve_knowledge",
        "retrieve_knowledge_multi": "retrieve_knowledge_multi",
        "direct_answer": "direct_answer",
        "chat_response": "chat_response",
        "fallback_response": "fallback_response"
    }
)

# 知识库查询后进入直接回答节点
graph.add_edge("retrieve_knowledge", "direct_answer")
graph.add_edge("retrieve_knowledge_multi", "direct_answer")

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
