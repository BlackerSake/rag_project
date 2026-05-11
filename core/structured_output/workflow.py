"""结构化输出工作流入口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from utils.logging_config import get_logger

from .fallbacks import fallback_judge_items
from .parser import StructuredOutputParseError, parse_json_output
from .repair import repair_json_text, repair_judge_output_with_llm, should_attempt_repair
from .schemas import JudgeOutputSchema
from .validators import sanitize_judge_items

logger = get_logger(__name__)

LAYER_LABELS = {
    "workflow": "完整链路",
    "syntax_parser": "语法解析",
    "schema_validation": "结构校验",
    "business_validation": "业务校验",
    "repair": "格式修复",
    "fallback": "安全兜底",
}
STATUS_LABELS = {
    "start": "开始",
    "success": "成功",
    "failed": "失败",
}
FIELD_LABELS = {
    "workflow": "流程",
    "raw_chars": "原始输出字符数",
    "candidate_count": "候选数",
    "top_level": "顶层类型",
    "item_count": "有效项数",
    "error": "错误",
    "error_type": "错误类型",
    "source": "来源",
    "repair_attempt_count": "修复次数",
}


@dataclass
class RepairBudget:
    """限制结构化输出修复次数。"""

    local_attempts: int = 0
    llm_attempts: int = 0
    max_local_attempts: int = 1
    max_llm_attempts: int = 1

    @property
    def total_attempts(self) -> int:
        """返回当前总修复次数。"""
        return self.local_attempts + self.llm_attempts

    def consume_local(self) -> int:
        """消耗一次本地修复次数。"""
        if self.local_attempts >= self.max_local_attempts:
            raise RuntimeError("本地修复次数已达上限")
        self.local_attempts += 1
        return self.total_attempts

    def consume_llm(self) -> int:
        """消耗一次 LLM 修复次数。"""
        if self.llm_attempts >= self.max_llm_attempts:
            raise RuntimeError("LLM 修复次数已达上限")
        self.llm_attempts += 1
        return self.total_attempts


def _log_layer(layer: str, status: str, **fields: Any) -> None:
    """输出结构化输出链路的分层日志。"""
    layer_label = LAYER_LABELS.get(layer, layer)
    status_label = STATUS_LABELS.get(status, status)
    details = "｜".join(
        f"{FIELD_LABELS.get(key, key)}={value}({key}={value})"
        for key, value in fields.items()
    )
    suffix = f"｜{details}" if details else ""
    logger.info(
        "结构化输出｜阶段=%s(%s)｜状态=%s(%s)%s",
        layer_label,
        layer,
        status_label,
        status,
        suffix,
    )


def _validate_judge_schema(payload: dict[str, Any]) -> JudgeOutputSchema:
    """执行查询拆解裁决 Schema 校验。"""
    return JudgeOutputSchema.model_validate(payload)


def _parse_with_local_repair(raw_text: str, workflow_name: str, repair_budget: RepairBudget) -> dict[str, Any]:
    """执行 JSON 解析，失败时尝试本地轻量修复。"""
    try:
        payload = parse_json_output(raw_text)
        _log_layer("syntax_parser", "success", workflow=workflow_name, top_level=type(payload).__name__)
        return payload
    except StructuredOutputParseError as exc:
        if not should_attempt_repair("JSON_PARSE_ERROR"):
            raise
        _log_layer("syntax_parser", "failed", workflow=workflow_name, error=exc)
        attempt_count = repair_budget.consume_local()
        _log_layer(
            "repair",
            "start",
            workflow=workflow_name,
            error_type="JSON_PARSE_ERROR",
            source="local",
            repair_attempt_count=attempt_count,
        )
        payload = parse_json_output(repair_json_text(raw_text))
        _log_layer(
            "repair",
            "success",
            workflow=workflow_name,
            error_type="JSON_PARSE_ERROR",
            source="local",
            repair_attempt_count=repair_budget.total_attempts,
        )
        return payload


async def _parse_with_repair(raw_text: str, workflow_name: str, repair_budget: RepairBudget) -> dict[str, Any]:
    """执行 JSON 解析，先本地修复，再使用 LLM 修复。"""
    try:
        return _parse_with_local_repair(raw_text, workflow_name, repair_budget)
    except Exception as exc:
        if not should_attempt_repair("JSON_PARSE_ERROR"):
            raise
        _log_layer(
            "repair",
            "failed",
            workflow=workflow_name,
            error_type="JSON_PARSE_ERROR",
            source="local",
            repair_attempt_count=repair_budget.total_attempts,
        )
        attempt_count = repair_budget.consume_llm()
        _log_layer(
            "repair",
            "start",
            workflow=workflow_name,
            error_type="JSON_PARSE_ERROR",
            source="llm",
            repair_attempt_count=attempt_count,
        )
        repaired_text = await repair_judge_output_with_llm(raw_text, exc)
        payload = parse_json_output(repaired_text)
        _log_layer(
            "repair",
            "success",
            workflow=workflow_name,
            error_type="JSON_PARSE_ERROR",
            source="llm",
            repair_attempt_count=repair_budget.total_attempts,
        )
        return payload


def _validate_schema_with_local_repair(
    raw_text: str,
    payload: dict[str, Any],
    workflow_name: str,
    repair_budget: RepairBudget,
) -> JudgeOutputSchema:
    """执行 Schema 校验，失败时尝试本地轻量修复。"""
    try:
        schema_output = _validate_judge_schema(payload)
        _log_layer(
            "schema_validation",
            "success",
            workflow=workflow_name,
            item_count=len(schema_output.items),
        )
        return schema_output
    except ValidationError as exc:
        if not should_attempt_repair("SCHEMA_ERROR"):
            raise
        _log_layer("schema_validation", "failed", workflow=workflow_name, error=exc.__class__.__name__)
        attempt_count = repair_budget.consume_local()
        _log_layer(
            "repair",
            "start",
            workflow=workflow_name,
            error_type="SCHEMA_ERROR",
            source="local",
            repair_attempt_count=attempt_count,
        )
        repaired_payload = parse_json_output(repair_json_text(raw_text))
        schema_output = _validate_judge_schema(repaired_payload)
        _log_layer(
            "repair",
            "success",
            workflow=workflow_name,
            error_type="SCHEMA_ERROR",
            source="local",
            repair_attempt_count=repair_budget.total_attempts,
        )
        return schema_output


async def _validate_schema_with_repair(
    raw_text: str,
    payload: dict[str, Any],
    workflow_name: str,
    repair_budget: RepairBudget,
) -> JudgeOutputSchema:
    """执行 Schema 校验，先本地修复，再使用 LLM 修复。"""
    try:
        return _validate_schema_with_local_repair(raw_text, payload, workflow_name, repair_budget)
    except Exception as exc:
        if not should_attempt_repair("SCHEMA_ERROR"):
            raise
        _log_layer(
            "repair",
            "failed",
            workflow=workflow_name,
            error_type="SCHEMA_ERROR",
            source="local",
            repair_attempt_count=repair_budget.total_attempts,
        )
        attempt_count = repair_budget.consume_llm()
        _log_layer(
            "repair",
            "start",
            workflow=workflow_name,
            error_type="SCHEMA_ERROR",
            source="llm",
            repair_attempt_count=attempt_count,
        )
        repaired_text = await repair_judge_output_with_llm(raw_text, exc)
        repaired_payload = parse_json_output(repaired_text)
        schema_output = _validate_judge_schema(repaired_payload)
        _log_layer(
            "repair",
            "success",
            workflow=workflow_name,
            error_type="SCHEMA_ERROR",
            source="llm",
            repair_attempt_count=repair_budget.total_attempts,
        )
        return schema_output


def _sanitize_or_raise(
    schema_output: JudgeOutputSchema,
    candidate_by_id: Mapping[str, Any],
    valid_intent_ids: set[str] | list[str] | tuple[str, ...],
    workflow_name: str,
) -> list[dict[str, Any]]:
    """执行业务校验并确保输出不为空。"""
    _log_layer("business_validation", "start", workflow=workflow_name)
    sanitized_items = sanitize_judge_items(
        schema_output.model_dump(),
        candidate_by_id,
        valid_intent_ids,
    )
    if not sanitized_items:
        raise ValueError("裁决输出无有效 item")
    _log_layer(
        "business_validation",
        "success",
        workflow=workflow_name,
        item_count=len(sanitized_items),
    )
    _log_layer("workflow", "success", workflow=workflow_name, item_count=len(sanitized_items))
    return sanitized_items


def _fallback_after_error(
    exc: Exception,
    candidate_by_id: Mapping[str, Any],
    workflow_name: str,
) -> list[dict[str, Any]]:
    """结构化链路失败后返回安全兜底。"""
    _log_layer("workflow", "failed", workflow=workflow_name, error=exc.__class__.__name__)
    fallback_items = fallback_judge_items(candidate_by_id)
    _log_layer("fallback", "success", workflow=workflow_name, item_count=len(fallback_items))
    return fallback_items


def parse_and_validate_judge_output(
    raw_text: str,
    candidate_by_id: Mapping[str, Any],
    valid_intent_ids: set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """同步解析、校验并清洗 query_decomposition_judge 输出。"""
    workflow_name = "query_decomposition_judge"
    _log_layer(
        "workflow",
        "start",
        workflow=workflow_name,
        raw_chars=len(str(raw_text or "")),
        candidate_count=len(candidate_by_id),
    )
    try:
        repair_budget = RepairBudget(max_llm_attempts=0)
        payload = _parse_with_local_repair(raw_text, workflow_name, repair_budget)
        schema_output = _validate_schema_with_local_repair(raw_text, payload, workflow_name, repair_budget)
        return _sanitize_or_raise(schema_output, candidate_by_id, valid_intent_ids, workflow_name)
    except Exception as exc:
        return _fallback_after_error(exc, candidate_by_id, workflow_name)


async def parse_and_validate_judge_output_async(
    raw_text: str,
    candidate_by_id: Mapping[str, Any],
    valid_intent_ids: set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """异步解析、校验并清洗 query_decomposition_judge 输出，支持 LLM 修复。"""
    workflow_name = "query_decomposition_judge"
    _log_layer(
        "workflow",
        "start",
        workflow=workflow_name,
        raw_chars=len(str(raw_text or "")),
        candidate_count=len(candidate_by_id),
    )
    try:
        repair_budget = RepairBudget()
        payload = await _parse_with_repair(raw_text, workflow_name, repair_budget)
        schema_output = await _validate_schema_with_repair(raw_text, payload, workflow_name, repair_budget)
        return _sanitize_or_raise(schema_output, candidate_by_id, valid_intent_ids, workflow_name)
    except Exception as exc:
        return _fallback_after_error(exc, candidate_by_id, workflow_name)
