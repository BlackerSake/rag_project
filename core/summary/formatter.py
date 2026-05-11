"""上下文格式化工具。消息格式化"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .structured_state import format_tool_result

MAX_MESSAGE_CONTENT_CHARS = 4000


def _stable_text(value: Any) -> str:
    """将任意内容转成适合摘要输入的文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _truncate_text(text: str, max_chars: int = MAX_MESSAGE_CONTENT_CHARS) -> str:
    """截断过长文本，避免单条消息挤占摘要输入。"""
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "\n...（内容过长，已截断）"


def _message_role(message: BaseMessage) -> str:
    """获取消息在摘要中的中文角色名。"""
    if isinstance(message, HumanMessage):
        return "用户需求"
    if isinstance(message, AIMessage):
        return "助手回答或决策"
    if isinstance(message, ToolMessage):
        return "工具返回"
    if isinstance(message, SystemMessage):
        return "系统动态规则"
    return message.__class__.__name__


def format_messages_for_summary(cleaned_old_messages: list[BaseMessage]) -> str:
    """将旧消息格式化为摘要模型输入。"""
    lines: list[str] = []
    for index, message in enumerate(cleaned_old_messages, start=1):
        role = _message_role(message)
        if isinstance(message, ToolMessage):
            content = format_tool_result(message)
        else:
            content = _truncate_text(_stable_text(getattr(message, "content", "")))
        if not content:
            continue
        lines.append(f"[{index}] {role}\n{content}")
    return "\n\n".join(lines) if lines else "无新增旧消息"


def build_context_for_model(state: dict[str, Any]) -> str:
    """组装后续模型调用可使用的独立上下文区块。"""
    structured_state = state.get("structured_state") or {}
    summary = state.get("summary") or "无"
    knowledge_results = state.get("knowledge_results") or []
    recent_messages = state.get("messages") or []

    blocks = [
        "【结构化状态】",
        _stable_text(structured_state) or "无",
        "【滚动摘要】",
        str(summary),
        "【检索上下文】",
        _stable_text(knowledge_results) or "无",
        "【近期原文消息】",
        format_messages_for_summary(recent_messages),
    ]
    return "\n".join(blocks)

