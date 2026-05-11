"""结构化输出轻量修复工具。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .parser import extract_json_array_or_object

REPAIRABLE_ERROR_TYPES = {"JSON_PARSE_ERROR", "SCHEMA_ERROR"}
_DEEPSEEK_REPAIR_MODEL: Any | None = None


def repair_json_text(raw_text: str) -> str:
    """对常见 JSON 格式问题做保守修复。"""
    json_text = extract_json_array_or_object(raw_text)
    json_text = re.sub(r",\s*([}\]])", r"\1", json_text)
    return json_text.strip()


def should_attempt_repair(error_type: str) -> bool:
    """判断错误类型是否适合尝试轻量修复。"""
    return str(error_type or "").upper() in REPAIRABLE_ERROR_TYPES


def _load_env_file() -> None:
    """加载项目 .env 文件。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path)


def _get_deepseek_repair_model() -> Any:
    """按需初始化 DeepSeek 修复模型。"""
    global _DEEPSEEK_REPAIR_MODEL
    if _DEEPSEEK_REPAIR_MODEL is not None:
        return _DEEPSEEK_REPAIR_MODEL

    _load_env_file()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    model_id = os.getenv("DEEPSEEK_MODEL_ID")
    if not api_key or not base_url or not model_id:
        raise RuntimeError("DeepSeek 修复模型缺少 DEEPSEEK_API_KEY/DEEPSEEK_BASE_URL/DEEPSEEK_MODEL_ID 配置")

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError("DeepSeek 修复模型依赖 langchain_openai 未安装") from exc

    _DEEPSEEK_REPAIR_MODEL = ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        streaming=False,
        temperature=0,
        timeout=30,
    )
    return _DEEPSEEK_REPAIR_MODEL


async def repair_judge_output_with_llm(raw_text: str, error: Exception | str, model: Any | None = None) -> str:
    """使用 LLM 修复裁决输出格式，不改变业务语义。"""
    repair_model = model or _get_deepseek_repair_model()

    prompt = (
        "你是结构化输出格式修复器。请只把下面的原始输出修复为合法 JSON 对象。\n"
        "严格要求：\n"
        "1. 不要重新判断业务，不要新增、删除或改写语义。\n"
        "2. 只修复 JSON 语法、字段类型、Markdown 包裹、尾随逗号等格式问题。\n"
        "3. 只能输出 JSON，不要输出解释文字或代码块。\n"
        "4. 顶层必须是对象，且必须包含 items 数组。\n"
        "5. items 内每项格式必须是：candidate_ids 为字符串数组，intent_id 为字符串，reason 为字符串。\n\n"
        "6. candidate_ids、intent_id、reason 的原始取值必须尽量逐字保留；除非是把单个 ID 包成数组、"
        "把数字转成字符串等类型修复，否则不要改字段值。\n\n"
        f"错误信息：\n{error}\n\n"
        f"原始输出：\n{raw_text}"
    )
    response = await repair_model.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return repair_json_text(content)
