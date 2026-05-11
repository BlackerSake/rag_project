"""上下文压缩主工作流。"""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from utils.logging_config import get_logger

from .compression import (
    clean_messages_for_summary,
    handle_summary_failure,
    should_compress_context,
    split_recent_and_old_messages,
    update_state_after_compression,
)
from .formatter import format_messages_for_summary
from .structured_state import compress_tool_results, extract_structured_state, merge_structured_state
from .summarizer import generate_running_summary

logger = get_logger(__name__)


async def compress_context(state: dict, prompt, llm) -> dict:
    """执行完整上下文压缩链路。"""
    if not should_compress_context(state):
        logger.info("上下文未达到压缩条件，跳过压缩")
        return {}

    messages = list(state.get("messages") or [])
    recent_messages, old_messages = split_recent_and_old_messages(messages)
    if not old_messages:
        logger.info("没有可压缩的旧消息，仅重置对话轮数")
        return {"conversation_rounds": 0}

    try:
        cleaned_old_messages = clean_messages_for_summary(old_messages)
        formatted_old_messages = format_messages_for_summary(cleaned_old_messages)
        old_summary = state.get("summary") or ""
        new_summary = await generate_running_summary(
            old_summary=old_summary,
            formatted_old_messages=formatted_old_messages,
            prompt=prompt,
            llm=llm,
        )
        tool_messages = [message for message in cleaned_old_messages if isinstance(message, ToolMessage)]
        compressed_tools = compress_tool_results(tool_messages)
        new_structured_state = extract_structured_state(
            old_messages=cleaned_old_messages,
            old_summary=old_summary,
            tool_results=compressed_tools,
        )
        merged_structured_state = merge_structured_state(
            state.get("structured_state") or {},
            new_structured_state,
        )
        logger.info(
            "上下文压缩完成: old_messages=%s recent_messages=%s tool_results=%s",
            len(old_messages),
            len(recent_messages),
            len(compressed_tools),
        )
        return update_state_after_compression(
            state=state,
            recent_messages=recent_messages,
            new_summary=new_summary,
            structured_state=merged_structured_state,
        )
    except Exception as exc:
        return handle_summary_failure(state, exc)

