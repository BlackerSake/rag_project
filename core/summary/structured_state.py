"""结构化状态抽取与合并。"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

MAX_FIELD_CHARS = 800
MAX_TOOL_CONTENT_CHARS = 1200
CONSTRAINT_KEYWORDS = (
    "必须",
    "不要",
    "不能",
    "禁止",
    "只需要",
    "无需",
    "优先",
    "保留",
    "不要总结",
    "不需要测试",
)
DISCARDED_KEYWORDS = ("废弃", "放弃", "不用", "不再", "改用", "不要再")


def _stable_text(value: Any) -> str:
    """将值转换为稳定文本。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _truncate_text(text: str, max_chars: int = MAX_FIELD_CHARS) -> str:
    """限制结构化字段长度。"""
    normalized = " ".join(str(text).split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "...（已截断）"


def _unique_items(items: list[Any]) -> list[Any]:
    """按稳定文本去重并保留顺序。"""
    result: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = _stable_text(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _extract_json_like_content(text: str) -> Any:
    """尽量解析工具输出中的 JSON 内容。"""
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def format_tool_result(tool_message: ToolMessage | dict[str, Any]) -> str:
    """压缩单条工具结果为可读摘要输入。"""
    if isinstance(tool_message, ToolMessage):
        tool_name = getattr(tool_message, "name", None) or "unknown_tool"
        tool_call_id = getattr(tool_message, "tool_call_id", "") or ""
        status = getattr(tool_message, "status", None) or "success"
        content = _stable_text(getattr(tool_message, "content", ""))
        args = _stable_text(getattr(tool_message, "additional_kwargs", {}) or {})
    else:
        tool_name = str(tool_message.get("tool_name") or tool_message.get("name") or "unknown_tool")
        tool_call_id = str(tool_message.get("tool_call_id") or "")
        status = str(tool_message.get("status") or ("error" if tool_message.get("error") else "success"))
        content = _stable_text(tool_message.get("content") or tool_message.get("result") or "")
        args = _stable_text(tool_message.get("args") or tool_message.get("parameters") or {})

    parsed_content = _extract_json_like_content(content)
    compressed_content = _truncate_text(_stable_text(parsed_content), MAX_TOOL_CONTENT_CHARS)
    failed = status.lower() in {"error", "failed", "failure"} or bool(re.search(r"error|exception|失败|错误", content, re.I))
    failure_reason = compressed_content if failed else "无"

    return "\n".join([
        f"工具名称: {tool_name}",
        f"调用参数: {args or '无'}",
        f"调用ID: {tool_call_id or '无'}",
        f"是否成功: {'否' if failed else '是'}",
        f"最终可信结果: {compressed_content or '无'}",
        f"失败原因: {failure_reason}",
        "是否可复用: 是",
        "过期时间: 未指定",
    ])


def compress_tool_results(tool_messages: list[ToolMessage | dict[str, Any]]) -> list[dict[str, Any]]:
    """专门压缩工具结果，供结构化状态复用。"""
    compressed_results: list[dict[str, Any]] = []
    for message in tool_messages:
        if isinstance(message, ToolMessage):
            content = _stable_text(getattr(message, "content", ""))
            status = str(getattr(message, "status", "") or "success")
            tool_name = str(getattr(message, "name", "") or "unknown_tool")
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            args = getattr(message, "additional_kwargs", {}) or {}
        else:
            content = _stable_text(message.get("content") or message.get("result") or "")
            status = str(message.get("status") or ("error" if message.get("error") else "success"))
            tool_name = str(message.get("tool_name") or message.get("name") or "unknown_tool")
            tool_call_id = str(message.get("tool_call_id") or "")
            args = message.get("args") or message.get("parameters") or {}

        failed = status.lower() in {"error", "failed", "failure"} or bool(re.search(r"error|exception|失败|错误", content, re.I))
        compressed_results.append({
            "tool_name": tool_name,
            "parameters": args,
            "tool_call_id": tool_call_id,
            "success": not failed,
            "result": _truncate_text(content, MAX_TOOL_CONTENT_CHARS),
            "failure_reason": _truncate_text(content, 400) if failed else "",
            "reusable": True,
            "expires_at": None,
        })
    return compressed_results


def extract_structured_state(
    old_messages: list[BaseMessage],
    old_summary: str | None,
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """从旧消息和工具结果中抽取稳定结构化状态。"""
    human_texts: list[str] = []
    assistant_texts: list[str] = []
    constraints: list[str] = []
    discarded_context: list[str] = []

    for message in old_messages:
        content = _truncate_text(_stable_text(getattr(message, "content", "")))
        if not content:
            continue
        if isinstance(message, HumanMessage):
            human_texts.append(content)
            if any(keyword in content for keyword in CONSTRAINT_KEYWORDS):
                constraints.append(content)
            if any(keyword in content for keyword in DISCARDED_KEYWORDS):
                discarded_context.append(content)
        elif isinstance(message, AIMessage):
            assistant_texts.append(content)

    return {
        "goal": human_texts[-1] if human_texts else "",
        "constraints": _unique_items(constraints),
        "facts": _unique_items([old_summary] if old_summary else []),
        "decisions": _unique_items(assistant_texts[-5:]),
        "tool_results": tool_results,
        "open_questions": [],
        "discarded_context": _unique_items(discarded_context),
    }


def merge_structured_state(
    old_structured_state: dict[str, Any] | None,
    new_structured_state: dict[str, Any],
) -> dict[str, Any]:
    """合并新旧结构化状态，优先保留较新的用户指令。"""
    old_state = old_structured_state or {}
    merged = {
        "goal": new_structured_state.get("goal") or old_state.get("goal") or "",
        "constraints": _unique_items([
            *(old_state.get("constraints") or []),
            *(new_structured_state.get("constraints") or []),
        ]),
        "facts": _unique_items([
            *(old_state.get("facts") or []),
            *(new_structured_state.get("facts") or []),
        ]),
        "decisions": _unique_items([
            *(old_state.get("decisions") or []),
            *(new_structured_state.get("decisions") or []),
        ]),
        "tool_results": _unique_items([
            *(old_state.get("tool_results") or []),
            *(new_structured_state.get("tool_results") or []),
        ]),
        "open_questions": _unique_items([
            *(old_state.get("open_questions") or []),
            *(new_structured_state.get("open_questions") or []),
        ]),
        "discarded_context": _unique_items([
            *(old_state.get("discarded_context") or []),
            *(new_structured_state.get("discarded_context") or []),
        ]),
    }
    return merged

