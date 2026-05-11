"""意图候选门控、状态写入与反馈闭环。"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baseline import (
    IntentBaseline,
    PROJECT_ROOT,
    build_intent_baseline_from_eval,
    load_intent_baseline,
    rebuild_intent_baseline as _rebuild_intent_baseline,
)


DEFAULT_DECISION_LOG_PATH = PROJECT_ROOT / "logs" / "intent.log"
DEFAULT_ANNOTATION_POOL_PATH = PROJECT_ROOT / "logs" / "intent_annotation_pool.jsonl"
NEGATIVE_FEEDBACK_KEYWORDS = (
    "不对",
    "错了",
    "不是这个",
    "你没理解",
    "转人工",
)


class IntentConfidenceHistory:
    """保存最近的意图门控分数与 margin。"""

    def __init__(self, max_size: int = 100):
        self.records: deque[dict[str, Any]] = deque(maxlen=max_size)

    def update(self, record: dict[str, Any]) -> None:
        """写入一条滑动窗口记录。"""
        self.records.append(record)

    def __iter__(self):
        return iter(self.records)


_intent_confidence_history = IntentConfidenceHistory(max_size=100)


def get_intent_confidence_history() -> IntentConfidenceHistory:
    """获取全局意图门控滑动窗口。"""
    return _intent_confidence_history


def _now_iso() -> str:
    """生成 UTC ISO 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _safe_score(value: Any) -> float:
    """安全转换候选分数。"""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def normalize_intent_candidates(results: list[tuple[dict[str, Any], float]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 Milvus 原始结果归一化为门控候选结构。"""
    candidates: list[dict[str, Any]] = []
    for item in results or []:
        if isinstance(item, dict):
            entity = item
            score = item.get("score", item.get("distance", 0.0))
        elif isinstance(item, tuple) and len(item) >= 2:
            entity = item[0] if isinstance(item[0], dict) else {}
            score = item[1]
        else:
            continue

        intent_id = str(entity.get("intent_id") or "").strip()
        if not intent_id:
            continue
        candidates.append({
            "intent_id": intent_id,
            "score": _safe_score(score),
            "name": entity.get("name", ""),
            "path": entity.get("path", ""),
        })

    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    return candidates


def calculate_intent_margin(candidates: list[dict[str, Any]]) -> float:
    """计算 Top1 与 Top2 的分数差。"""
    if not candidates:
        return 0.0
    top1_score = _safe_score(candidates[0].get("score"))
    if len(candidates) < 2:
        return top1_score
    top2_score = _safe_score(candidates[1].get("score"))
    return max(top1_score - top2_score, 0.0)


def detect_confusion_pair(top1: dict[str, Any] | None, top2: dict[str, Any] | None, baseline: IntentBaseline | None = None) -> bool:
    """判断 Top1 与 Top2 是否属于已知混淆组。"""
    if not top1 or not top2:
        return False
    first_id = str(top1.get("intent_id") or "").strip()
    second_id = str(top2.get("intent_id") or "").strip()
    if not first_id or not second_id:
        return False

    active_baseline = baseline or load_intent_baseline()
    for group in active_baseline.confusion_groups:
        group_set = set(group)
        if first_id in group_set and second_id in group_set:
            return True
    return False


def build_intent_clarification_question(candidates: list[dict[str, Any]]) -> str:
    """根据候选意图生成简短澄清问题。"""
    if len(candidates) >= 2:
        first = candidates[0].get("name") or candidates[0].get("intent_id")
        second = candidates[1].get("name") or candidates[1].get("intent_id")
        return f"你是想咨询{first}，还是{second}？"
    if candidates:
        name = candidates[0].get("name") or candidates[0].get("intent_id")
        return f"你是想咨询{name}相关问题吗？"
    return "我还不确定你的问题类型，可以再补充一下你想咨询的内容吗？"


def intent_gate_decide(
    query: str,
    candidates: list[dict[str, Any]],
    baseline: IntentBaseline | None = None,
) -> dict[str, Any]:
    """根据候选分数、margin 与 baseline 生成门控决策。"""
    active_baseline = baseline or load_intent_baseline()
    normalized_candidates = normalize_intent_candidates(candidates)
    if not normalized_candidates:
        return {
            "query": query,
            "intent_candidates": [],
            "intent_id": None,
            "intent_score": 0.0,
            "intent_margin": 0.0,
            "intent_confidence_level": "LOW",
            "intent_gate_action": "FALLBACK",
            "intent_gate_reason": "no_candidates",
            "clarification_question": build_intent_clarification_question([]),
        }

    top1 = normalized_candidates[0]
    top2 = normalized_candidates[1] if len(normalized_candidates) > 1 else None
    score = _safe_score(top1.get("score"))
    margin = calculate_intent_margin(normalized_candidates)
    is_confusion_pair = detect_confusion_pair(top1, top2, active_baseline)

    if score < active_baseline.score_low:
        level = "LOW"
        action = "FALLBACK"
        reason = "score_below_low_threshold"
    elif margin >= active_baseline.margin_high:
        level = "HIGH"
        action = "ACCEPT"
        reason = "margin_above_high_threshold"
    elif score >= active_baseline.score_high and margin >= active_baseline.margin_low:
        level = "MEDIUM"
        action = "ACCEPT"
        reason = "score_high_and_margin_medium"
    elif margin < active_baseline.margin_low:
        level = "MEDIUM"
        action = "CLARIFY"
        reason = "margin_below_low_threshold"
    elif is_confusion_pair:
        level = "MEDIUM"
        action = "CLARIFY"
        reason = "known_confusion_pair"
    else:
        level = "MEDIUM"
        action = "CLARIFY"
        reason = "middle_confidence"

    return {
        "query": query,
        "intent_candidates": normalized_candidates,
        "intent_id": top1.get("intent_id"),
        "intent_score": score,
        "intent_margin": margin,
        "intent_confidence_level": level,
        "intent_gate_action": action,
        "intent_gate_reason": reason,
        "clarification_question": build_intent_clarification_question(normalized_candidates),
        "is_confusion_pair": is_confusion_pair,
        "baseline": active_baseline.to_dict(),
    }


def write_intent_gate_to_state(state: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """把门控决策写入 LangGraph State 兼容字段。"""
    intent_id = decision.get("intent_id")
    if decision.get("intent_gate_action") == "FALLBACK" and not intent_id:
        intent_id = "D2"

    return {
        "intent_candidates": decision.get("intent_candidates", []),
        "intent_id": intent_id,
        "current_topic": intent_id,
        "intent_score": decision.get("intent_score", 0.0),
        "intent_margin": decision.get("intent_margin", 0.0),
        "intent_confidence_level": decision.get("intent_confidence_level"),
        "intent_gate_action": decision.get("intent_gate_action"),
        "intent_gate_reason": decision.get("intent_gate_reason"),
        "clarification_question": decision.get("clarification_question", ""),
    }


def route_after_intent_gate(state: dict[str, Any]) -> str:
    """根据意图门控动作选择下游节点。"""
    action = state.get("intent_gate_action")
    if action in {"ACCEPT", "CLARIFY"}:
        return "task_dispatcher"
    if action == "FALLBACK":
        return "fallback_response"
    return "task_dispatcher"


def _resolve_output_path(filepath: str | os.PathLike | None, default_path: Path) -> Path:
    """解析日志或标注文件路径。"""
    path = Path(filepath) if filepath else default_path
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def log_intent_gate_decision(state: dict[str, Any], filepath: str | os.PathLike | None = None) -> None:
    """记录每次意图门控结果，失败时静默避免影响主链路。"""
    path = _resolve_output_path(filepath or os.getenv("INTENT_GATE_LOG_PATH"), DEFAULT_DECISION_LOG_PATH)
    record = {
        "timestamp": _now_iso(),
        "query": state.get("query") or "",
        "top_k": state.get("intent_candidates", []),
        "top1_score": state.get("intent_score", 0.0),
        "margin": state.get("intent_margin", 0.0),
        "gate_level": state.get("intent_confidence_level"),
        "gate_action": state.get("intent_gate_action"),
        "reason": state.get("intent_gate_reason"),
        "final_route": state.get("final_route"),
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return


def update_intent_confidence_history(state: dict[str, Any], history: IntentConfidenceHistory | None = None) -> None:
    """将本次门控分数写入滑动窗口。"""
    target_history = history or _intent_confidence_history
    target_history.update({
        "timestamp": _now_iso(),
        "intent_id": state.get("intent_id"),
        "score": _safe_score(state.get("intent_score")),
        "margin": _safe_score(state.get("intent_margin")),
        "action": state.get("intent_gate_action"),
    })


def detect_strong_negative_feedback(user_text: str) -> bool:
    """判断用户是否给出强负反馈。"""
    text = str(user_text or "").strip()
    return any(keyword in text for keyword in NEGATIVE_FEEDBACK_KEYWORDS)


def add_to_annotation_pool(
    state: dict[str, Any],
    feedback_text: str,
    filepath: str | os.PathLike | None = None,
) -> None:
    """把强负反馈样本写入待标注池。"""
    path = _resolve_output_path(filepath or os.getenv("INTENT_ANNOTATION_POOL_PATH"), DEFAULT_ANNOTATION_POOL_PATH)
    record = {
        "created_at": _now_iso(),
        "query": state.get("query") or "",
        "top_k": state.get("intent_candidates", []),
        "gate_decision": {
            "intent_id": state.get("intent_id"),
            "level": state.get("intent_confidence_level"),
            "action": state.get("intent_gate_action"),
            "reason": state.get("intent_gate_reason"),
        },
        "final_route": state.get("final_route"),
        "user_feedback": feedback_text,
        "human_label": "",
        "error_type": "",
        "should_add_to_eval": False,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，忽略损坏行。"""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def merge_annotation_to_eval_set(annotation_file: str | os.PathLike, eval_file: str | os.PathLike) -> int:
    """把人工确认的标注样本合并到离线评测集。"""
    annotation_path = _resolve_output_path(annotation_file, DEFAULT_ANNOTATION_POOL_PATH)
    eval_path = _resolve_output_path(eval_file, PROJECT_ROOT / "evaluation" / "dataset" / "intent_eval.json")

    annotations = [
        item for item in _read_jsonl(annotation_path)
        if item.get("should_add_to_eval") and str(item.get("human_label") or "").strip()
    ]
    if not annotations:
        return 0

    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {"records": []}

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.setdefault("records", [])
    else:
        payload = {"records": []}
        records = payload["records"]

    existing_keys = {
        (str(item.get("query") or ""), str(item.get("expected_intent_id") or ""))
        for item in records
        if isinstance(item, dict)
    }
    added = 0
    for item in annotations:
        query = str(item.get("query") or "")
        label = str(item.get("human_label") or "")
        key = (query, label)
        if key in existing_keys:
            continue
        records.append({
            "query": query,
            "expected_intent_id": label,
            "source": "annotation_pool",
            "created_at": item.get("created_at"),
        })
        existing_keys.add(key)
        added += 1

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return added


def rebuild_intent_baseline(eval_file: str | os.PathLike, output_file: str | os.PathLike | None = None) -> dict[str, Any]:
    """重新生成意图 baseline，作为 gate 模块的便捷入口。"""
    if output_file is None:
        return build_intent_baseline_from_eval(eval_file)
    return _rebuild_intent_baseline(eval_file, output_file)
