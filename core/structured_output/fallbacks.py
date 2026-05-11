"""结构化输出安全降级策略。"""

from __future__ import annotations

import os
from typing import Any, Mapping

from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_FALLBACK_INTENT_ID = os.getenv("STRUCTURED_OUTPUT_DEFAULT_FALLBACK_INTENT_ID", "D2").strip() or "D2"


def _candidate_text(candidate: Any) -> str:
    """读取候选原文。"""
    if isinstance(candidate, Mapping):
        return str(candidate.get("original") or candidate.get("sub_question") or candidate.get("text") or "").strip()
    if isinstance(candidate, (tuple, list)) and candidate:
        return str(candidate[0]).strip()
    return ""


def _candidate_intent(candidate: Any) -> str:
    """读取候选意图。"""
    if isinstance(candidate, Mapping):
        return str(candidate.get("intent_id") or "").strip()
    if isinstance(candidate, (tuple, list)) and len(candidate) > 1:
        return "" if candidate[1] is None else str(candidate[1]).strip()
    return ""


def _candidate_score(candidate: Any) -> float:
    """读取候选分数。"""
    raw_score = 1.0
    if isinstance(candidate, Mapping):
        raw_score = candidate.get("score", 1.0)
    elif isinstance(candidate, (tuple, list)) and len(candidate) > 2:
        raw_score = candidate[2]
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        return 1.0


def fallback_judge_items(candidate_by_id: Mapping[str, Any]) -> list[dict[str, Any]]:
    """裁决失败时每个候选独立保留，避免丢失用户信息。"""
    results: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_by_id.items():
        cid = str(candidate_id).strip()
        if not cid:
            continue
        intent_id = _candidate_intent(candidate) or DEFAULT_FALLBACK_INTENT_ID
        if not _candidate_intent(candidate):
            logger.warning(
                "结构化输出｜安全兜底｜候选缺少意图，使用默认意图｜候选ID=%s(candidate_id=%s)｜意图=%s(intent_id=%s)",
                cid,
                cid,
                intent_id,
                intent_id,
            )
        results.append({
            "candidate_ids": [cid],
            "intent_id": intent_id,
            "reason": "规则兜底保留",
        })
    return results


def fallback_rewrite_items(candidate_by_id: Mapping[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """改写失败时保留原句。"""
    if isinstance(candidate_by_id, list):
        iterable = ((str(item.get("candidate_id") or index), item) for index, item in enumerate(candidate_by_id, 1))
    else:
        iterable = candidate_by_id.items()

    results: list[dict[str, Any]] = []
    for candidate_id, candidate in iterable:
        original = _candidate_text(candidate)
        intent_id = _candidate_intent(candidate)
        if not original or not intent_id:
            continue
        results.append({
            "candidate_id": str(candidate_id),
            "original": original,
            "intent_id": intent_id,
            "rewritten": original,
            "score": _candidate_score(candidate),
        })
    return results


def fallback_structured_state(
    old_summary: str | None,
    old_messages: list[Any] | None,
    tool_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """结构化状态失败时返回最小可用状态。"""
    latest_user_text = ""
    for message in reversed(old_messages or []):
        message_type = message.__class__.__name__.lower()
        content = str(getattr(message, "content", "")).strip()
        if content and ("human" in message_type or "user" in message_type):
            latest_user_text = content
            break

    return {
        "goal": latest_user_text,
        "constraints": [],
        "facts": [old_summary] if old_summary else [],
        "decisions": [],
        "tool_results": tool_results or [],
        "open_questions": [],
        "discarded_context": [],
    }
