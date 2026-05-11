"""上下文压缩触发、切分与状态回写。"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CONTEXT_TOKENS = 8192
DEFAULT_CONTEXT_USAGE_RATIO = 0.7
DEFAULT_RECENT_MESSAGE_COUNT = 5
DEFAULT_MAX_MESSAGE_COUNT = 50
DEFAULT_MAX_TOOL_RESULT_TOKENS = 100
DEFAULT_MAX_CONVERSATION_ROUNDS = 10
DEFAULT_RESERVED_OUTPUT_TOKENS = 1024
DEFAULT_SYSTEM_PROMPT_TOKENS = 1200
DEFAULT_TOOL_SCHEMA_TOKENS = 1200


def _int_env(name: str, default: int) -> int:
    """读取整数环境变量，非法值回退为默认值。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("环境变量 %s=%s 不是有效整数，使用默认值 %s", name, raw_value, default)
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    """读取浮点环境变量，非法值回退为默认值。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("环境变量 %s=%s 不是有效浮点数，使用默认值 %.2f", name, raw_value, default)
        return default
    return value if 0 < value <= 1 else default


def _stable_text(value: Any) -> str:
    """将任意值转成稳定文本，便于粗略估算 token。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _estimate_tokens(text: str) -> int:
    """用轻量规则估算 token，避免为压缩功能新增依赖。"""
    if not text:
        return 0

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    punctuation = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))
    return max(1, chinese_chars + ascii_words + punctuation // 2)


def _message_token_count(message: BaseMessage) -> int:
    """估算单条消息的 token 数。"""
    parts = [
        message.__class__.__name__,
        _stable_text(getattr(message, "content", "")),
        _stable_text(getattr(message, "additional_kwargs", {})),
    ]
    if isinstance(message, ToolMessage):
        parts.extend([
            str(getattr(message, "name", "") or ""),
            str(getattr(message, "tool_call_id", "") or ""),
            str(getattr(message, "status", "") or ""),
        ])
    return _estimate_tokens("\n".join(parts))


def _iter_messages(messages: Iterable[Any] | None) -> list[BaseMessage]:
    """过滤出可处理的 LangChain 消息对象。"""
    return [message for message in messages or [] if isinstance(message, BaseMessage)]


def count_context_tokens(state: dict[str, Any]) -> int:
    """统计当前上下文的粗略 token 占用。"""
    messages = _iter_messages(state.get("messages"))
    message_tokens = sum(_message_token_count(message) for message in messages)
    summary_tokens = _estimate_tokens(_stable_text(state.get("summary")))
    structured_tokens = _estimate_tokens(_stable_text(state.get("structured_state")))
    knowledge_tokens = _estimate_tokens(_stable_text(state.get("knowledge_results")))
    processed_tokens = _estimate_tokens(_stable_text(state.get("processed_results")))
    tool_history_tokens = _estimate_tokens(_stable_text(state.get("tool_execution_history")))

    return (
        message_tokens
        + summary_tokens
        + structured_tokens
        + knowledge_tokens
        + processed_tokens
        + tool_history_tokens
        + _int_env("SUMMARY_RESERVED_OUTPUT_TOKENS", DEFAULT_RESERVED_OUTPUT_TOKENS)
        + _int_env("SUMMARY_SYSTEM_PROMPT_TOKENS", DEFAULT_SYSTEM_PROMPT_TOKENS)
        + _int_env("SUMMARY_TOOL_SCHEMA_TOKENS", DEFAULT_TOOL_SCHEMA_TOKENS)
    )


def should_compress_context(state: dict[str, Any]) -> bool:
    """根据 token、消息数量、工具结果长度和轮数判断是否压缩。"""
    messages = _iter_messages(state.get("messages"))
    token_count = count_context_tokens(state)
    max_context_tokens = _int_env("SUMMARY_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)
    usage_ratio = _float_env("SUMMARY_CONTEXT_USAGE_RATIO", DEFAULT_CONTEXT_USAGE_RATIO)
    max_message_count = _int_env("SUMMARY_MAX_MESSAGE_COUNT", DEFAULT_MAX_MESSAGE_COUNT)
    max_tool_tokens = _int_env("SUMMARY_MAX_TOOL_RESULT_TOKENS", DEFAULT_MAX_TOOL_RESULT_TOKENS)
    max_rounds = _int_env("SUMMARY_MAX_CONVERSATION_ROUNDS", DEFAULT_MAX_CONVERSATION_ROUNDS)
    conversation_rounds = int(state.get("conversation_rounds") or 0)
    longest_tool_tokens = max(
        (_message_token_count(message) for message in messages if isinstance(message, ToolMessage)),
        default=0,
    )

    should_compress = (
        token_count >= max_context_tokens * usage_ratio
        or len(messages) >= max_message_count
        or longest_tool_tokens >= max_tool_tokens
        or conversation_rounds >= max_rounds
    )
    logger.info(
        "上下文压缩判断: should=%s tokens=%s threshold=%.0f messages=%s tool_tokens=%s rounds=%s",
        should_compress,
        token_count,
        max_context_tokens * usage_ratio,
        len(messages),
        longest_tool_tokens,
        conversation_rounds,
    )
    return should_compress


def split_recent_and_old_messages(
    messages: list[BaseMessage],
    recent_count: int | None = None,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """将消息切分为近期原文和待压缩旧消息。"""
    keep_count = recent_count or _int_env("SUMMARY_RECENT_MESSAGE_COUNT", DEFAULT_RECENT_MESSAGE_COUNT)
    if len(messages) <= keep_count:
        return list(messages), []
    return list(messages[-keep_count:]), list(messages[:-keep_count])


def clean_messages_for_summary(old_messages: list[BaseMessage]) -> list[BaseMessage]:
    """清理空消息和完全重复消息，保留可追溯的重要上下文。"""
    cleaned_messages: list[BaseMessage] = []
    seen_keys: set[tuple[str, str]] = set()
    for message in old_messages:
        content = _stable_text(getattr(message, "content", "")).strip()
        if not content:
            continue

        key = (message.__class__.__name__, content)
        if key in seen_keys:
            continue

        seen_keys.add(key)
        cleaned_messages.append(message)

    return cleaned_messages


def update_state_after_compression(
    state: dict[str, Any],
    recent_messages: list[BaseMessage],
    new_summary: str,
    structured_state: dict[str, Any],
) -> dict[str, Any]:
    """生成压缩后的 State 更新，避免覆盖业务中间状态。"""
    return {
        "messages": {
            "__replace__": True,
            "messages": recent_messages,
        },
        "summary": new_summary,
        "structured_state": structured_state,
        "context_token_count": count_context_tokens(state),
        "summary_error": None,
        "conversation_rounds": 0,
    }


def handle_summary_failure(state: dict[str, Any], error: Exception) -> dict[str, Any]:
    """压缩失败时保留原状态，避免写入污染性摘要。"""
    logger.error("上下文压缩失败，保留原摘要和消息: %s", error, exc_info=True)
    return {
        "summary_error": str(error),
    }
