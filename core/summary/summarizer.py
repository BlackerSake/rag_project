"""滚动摘要生成。"""

from __future__ import annotations

from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_summary_prompt(old_summary: str | None, formatted_old_messages: str) -> dict[str, str]:
    """构造摘要链输入变量。"""
    return {
        "old_summary": old_summary or "无",
        "conversation": formatted_old_messages or "无新增旧消息",
    }


async def generate_running_summary(
    old_summary: str | None,
    formatted_old_messages: str,
    prompt: Any,
    llm: Any,
) -> str:
    """用旧摘要和本次旧消息生成新的滚动摘要。"""
    if prompt is None:
        raise ValueError("summarization_prompt 未初始化")
    if llm is None:
        raise ValueError("model 未初始化")

    chain = prompt | llm
    payload = build_summary_prompt(old_summary, formatted_old_messages)
    summary_parts: list[str] = []
    async for chunk in chain.astream(payload):
        content = getattr(chunk, "content", "")
        if content:
            summary_parts.append(str(content))

    new_summary = "".join(summary_parts).strip()
    if not new_summary:
        raise ValueError("摘要模型返回空内容")

    logger.info("滚动摘要生成完成，长度=%s", len(new_summary))
    return new_summary

