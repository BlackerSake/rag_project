"""结构化输出 JSON 解析工具。"""

from __future__ import annotations

import json
import re
from typing import Any


class StructuredOutputParseError(ValueError):
    """结构化输出解析失败。"""


def _strip_code_fence(raw_text: str) -> str:
    """移除最外层 Markdown 代码块。"""
    text = str(raw_text or "").strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _find_balanced_json(text: str, open_char: str, close_char: str) -> str:
    """提取第一个括号平衡的 JSON 片段。"""
    start = text.find(open_char)
    if start == -1:
        raise StructuredOutputParseError("未找到 JSON 起始符")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    raise StructuredOutputParseError("JSON 括号不完整")


def extract_json_object(raw_text: str) -> str:
    """从 LLM 原始输出中提取 JSON 对象文本。"""
    text = _strip_code_fence(raw_text)
    if not text:
        raise StructuredOutputParseError("LLM 输出为空")

    if text.startswith("{") and text.endswith("}"):
        return text
    return _find_balanced_json(text, "{", "}")


def extract_json_array_or_object(raw_text: str) -> str:
    """从 LLM 原始输出中提取 JSON 对象或数组文本。"""
    text = _strip_code_fence(raw_text)
    if not text:
        raise StructuredOutputParseError("LLM 输出为空")

    object_pos = text.find("{")
    array_pos = text.find("[")
    positions = [(object_pos, "{", "}"), (array_pos, "[", "]")]
    positions = [item for item in positions if item[0] >= 0]
    if not positions:
        raise StructuredOutputParseError("未找到 JSON 顶层结构")

    _, open_char, close_char = min(positions, key=lambda item: item[0])
    if text.startswith(open_char) and text.endswith(close_char):
        return text
    return _find_balanced_json(text, open_char, close_char)


def parse_json_output(raw_text: str) -> dict[str, Any]:
    """解析 LLM 输出中的 JSON 对象。"""
    json_text = extract_json_object(raw_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputParseError("JSON 顶层结构必须是对象")
    return payload


def parse_json_array_or_object(raw_text: str) -> dict[str, Any] | list[Any]:
    """解析 LLM 输出中的 JSON 对象或数组。"""
    json_text = extract_json_array_or_object(raw_text)
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputParseError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(payload, (dict, list)):
        raise StructuredOutputParseError("JSON 顶层结构必须是对象或数组")
    return payload
