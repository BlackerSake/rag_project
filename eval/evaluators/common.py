"""RAG 评测器共享工具函数。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

LLMClient = Callable[[str], Awaitable[str] | str]


def clamp_score(value: Any, default: float = 0.0) -> float:
    """将任意值转换为 [0.0, 1.0] 区间内的分数。

    参数:
        value: Evaluator 或 LLM 返回的原始值。
        default: 转换失败时使用的默认分数。

    返回:
        被裁剪到闭区间 [0.0, 1.0] 的浮点数。

    异常:
        不主动抛出异常；非法值会使用 ``default``。
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        logger.warning("分数值无效: %r，使用默认值 %.3f", value, default)
        score = default
    return max(0.0, min(1.0, score))


def safe_divide(numerator: float, denominator: float) -> float:
    """带除零保护地执行除法。

    参数:
        numerator: 分子。
        denominator: 分母。

    返回:
        分母非零时返回 ``numerator / denominator``，否则返回 0.0。

    异常:
        不主动抛出异常。
    """
    if denominator == 0:
        logger.warning("已避免除零，返回 0.0")
        return 0.0
    return numerator / denominator


def extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出中提取 JSON 对象。

    参数:
        text: LLM 原始响应，可能包含 Markdown 代码块包裹。

    返回:
        解析后的 JSON 对象。

    异常:
        ValueError: 找不到 JSON 对象时抛出。
        json.JSONDecodeError: JSON 语法无效时抛出。
    """
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM 响应中未找到 JSON 对象")
        parsed = json.loads(raw[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM 响应 JSON 不是对象")
    return parsed


async def call_llm_json(
    llm_client: LLMClient,
    prompt: str,
    fallback: dict[str, Any],
    call_name: str,
) -> dict[str, Any]:
    """调用 LLM 客户端并解析 JSON，对失败情况进行降级处理。

    参数:
        llm_client: 接收 prompt 并返回文本的同步或异步可调用对象。
        prompt: 发送给 LLM 的提示词。
        fallback: 调用失败或解析失败时返回的默认对象。
        call_name: 日志中使用的调用名称。

    返回:
        解析后的 JSON 对象，或 ``fallback`` 的副本。

    异常:
        不主动抛出异常；所有失败都会记录日志并降级。
    """
    try:
        logger.info("调用 LLM 执行 %s", call_name)
        response = llm_client(prompt)
        if inspect.isawaitable(response):
            response = await response
        return extract_json_object(str(response))
    except Exception as exc:
        logger.error("LLM JSON 调用失败，任务=%s: %s", call_name, exc)
        return dict(fallback)


async def maybe_await(value: Any) -> Any:
    """仅在值可等待时执行 await。

    参数:
        value: 注入依赖返回的任意值。

    返回:
        可等待值的 await 结果，或原始值。

    异常:
        可等待对象自身抛出的异常会继续向外传播。
    """
    if inspect.isawaitable(value):
        return await value
    return value


async def run_sync(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """在默认执行器中运行同步函数。

    参数:
        func: 待执行的可调用对象。
        *args: 位置参数。
        **kwargs: 关键字参数。

    返回:
        可调用对象的执行结果。

    异常:
        ``func`` 抛出的异常会继续向外传播。
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def mean(values: list[float]) -> float:
    """带空列表保护地计算算术平均值。

    参数:
        values: 数值列表。

    返回:
        平均值；空列表返回 0.0。

    异常:
        不主动抛出异常。
    """
    if not values:
        logger.warning("空列表请求平均值，返回 0.0")
        return 0.0
    return sum(values) / len(values)
