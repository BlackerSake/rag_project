"""结构化输出业务校验与清洗。"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Mapping

from utils.logging_config import get_logger

logger = get_logger(__name__)


def _float_env(name: str, default: float) -> float:
    """读取浮点环境变量，非法值回退默认值。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("结构化输出配置无效，使用默认值: %s=%s default=%.4f", name, raw_value, default)
        return default
    if not math.isfinite(value):
        logger.warning("结构化输出配置不是有限数，使用默认值: %s=%s default=%.4f", name, raw_value, default)
        return default
    return value


LOW_CONFIDENCE_INTENT_SCORE = _float_env("STRUCTURED_OUTPUT_LOW_CONFIDENCE_INTENT_SCORE", 0.55)
DECOMPOSITION_TOPIC_KEYWORDS = {
    "logistics": ["快递", "快遞", "物流", "包裹", "单号", "單號", "丢", "丟", "没到", "未到", "到哪", "哪了"],
    "return": ["退货", "退貨", "换货", "換貨", "退换", "退換", "售后", "售後"],
    "refund": ["退款", "退钱", "退錢", "到账", "到賬", "多久", "几天", "幾天"],
    "repair": ["维修", "維修", "修理", "保修", "质保", "質保"],
    "order": ["订单", "訂單", "商品", "商家", "平台"],
}


def _candidate_text(candidate_id: str, candidate_by_id: Mapping[str, Any]) -> str:
    """取得候选原文。"""
    candidate = candidate_by_id.get(candidate_id)
    if isinstance(candidate, Mapping):
        return str(candidate.get("original") or candidate.get("sub_question") or candidate.get("text") or "")
    if isinstance(candidate, (tuple, list)) and candidate:
        return str(candidate[0])
    return ""


def _candidate_score(candidate_id: str, candidate_by_id: Mapping[str, Any]) -> float:
    """取得候选意图分数。"""
    candidate = candidate_by_id.get(candidate_id)
    raw_score = 0.0
    if isinstance(candidate, Mapping):
        raw_score = candidate.get("score", 0.0)
    elif isinstance(candidate, (tuple, list)) and len(candidate) > 2:
        raw_score = candidate[2]
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _candidate_intent(candidate_id: str, candidate_by_id: Mapping[str, Any]) -> str:
    """取得候选意图。"""
    candidate = candidate_by_id.get(candidate_id)
    if isinstance(candidate, Mapping):
        return str(candidate.get("intent_id") or "").strip()
    if isinstance(candidate, (tuple, list)) and len(candidate) > 1:
        return "" if candidate[1] is None else str(candidate[1]).strip()
    return ""


def _decomposition_identifiers(text: str) -> set[str]:
    """抽取订单号、物流单号等识别符。"""
    normalized = str(text).lower()
    return set(re.findall(r"[a-z]{0,4}\d{3,}[a-z0-9]*|\d{3,}", normalized))


def _decomposition_topics(text: str) -> set[str]:
    """用业务关键词识别语义主题。"""
    return {
        topic
        for topic, keywords in DECOMPOSITION_TOPIC_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }


def _decomposition_terms(text: str) -> set[str]:
    """建立轻量文本特征。"""
    normalized = re.sub(r"\s+", "", str(text).lower())
    terms = _decomposition_identifiers(normalized) | _decomposition_topics(normalized)
    terms.update(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z]{2,}", normalized))
    if len(normalized) >= 2:
        terms.update(normalized[index:index + 2] for index in range(len(normalized) - 1))
    elif normalized:
        terms.add(normalized)
    return terms


def _has_semantic_overlap(left_text: str, right_text: str) -> bool:
    """判断两段文本是否属于同一业务需求。"""
    left_ids = _decomposition_identifiers(left_text)
    right_ids = _decomposition_identifiers(right_text)
    if left_ids and right_ids and left_ids & right_ids:
        return True

    left_topics = _decomposition_topics(left_text)
    right_topics = _decomposition_topics(right_text)
    return bool(left_topics and right_topics and left_topics & right_topics)


def _is_semantically_incomplete_fragment(text: str) -> bool:
    """识别不应单独补回的语义残片。"""
    normalized = re.sub(r"[\s，。！？?!；;,.]+", "", str(text))
    if not normalized:
        return True
    if _decomposition_identifiers(normalized) or _decomposition_topics(normalized):
        return False

    filler_patterns = [
        r"^(对了|對了|然后|然後|还有|還有|另外|顺便|順便)$",
        r"^(是不是|可以吗|可以嗎|怎么办|怎麼辦)$",
        r"^(好的|谢谢|謝謝|知道了|明白|嗯|哦)$",
        r"^(这个|這個|那个|那個|我的|它|他|她)$",
    ]
    return len(normalized) <= 4 or any(re.search(pattern, normalized) for pattern in filler_patterns)


def _item_text(item: Mapping[str, Any], candidate_by_id: Mapping[str, Any]) -> str:
    """合并 item 内候选原文。"""
    candidate_ids = item.get("candidate_ids", [])
    if not isinstance(candidate_ids, list):
        return ""
    return "，".join(_candidate_text(str(candidate_id).strip(), candidate_by_id) for candidate_id in candidate_ids)


def _text_similarity(left_text: str, right_text: str) -> float:
    """用 Jaccard 重叠估算文本相似度。"""
    left_terms = _decomposition_terms(left_text)
    right_terms = _decomposition_terms(right_text)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def validate_intent_id(intent_id: str, valid_intent_ids: set[str] | list[str] | tuple[str, ...]) -> bool:
    """检查意图 ID 是否存在于真实意图目录。"""
    normalized = str(intent_id or "").strip()
    return bool(normalized and (not valid_intent_ids or normalized in set(valid_intent_ids)))


def validate_candidate_ids(candidate_ids: list[Any], candidate_by_id: Mapping[str, Any]) -> list[str]:
    """过滤不存在和重复的候选 ID。"""
    clean_ids: list[str] = []
    seen: set[str] = set()
    if not isinstance(candidate_ids, list):
        return clean_ids

    for candidate_id in candidate_ids:
        cid = str(candidate_id).strip()
        if not cid:
            continue
        if cid not in candidate_by_id:
            logger.warning("结构化输出｜业务校验｜未知候选已忽略｜候选ID=%s(candidate_id=%s)", cid, cid)
            continue
        if cid in seen:
            logger.info("结构化输出｜业务校验｜重复候选已忽略｜候选ID=%s(candidate_id=%s)", cid, cid)
            continue
        clean_ids.append(cid)
        seen.add(cid)
    return clean_ids


def dedupe_candidate_assignments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """确保同一个 candidate_id 只保留第一次有效归属。"""
    deduped_items: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for item in items:
        candidate_ids = item.get("candidate_ids", [])
        if not isinstance(candidate_ids, list):
            continue
        clean_ids: list[str] = []
        for candidate_id in candidate_ids:
            cid = str(candidate_id).strip()
            if not cid:
                continue
            if cid in used_ids:
                logger.info("结构化输出｜业务校验｜候选已有归属，保留首次结果｜候选ID=%s(candidate_id=%s)", cid, cid)
                continue
            clean_ids.append(cid)
            used_ids.add(cid)
        if clean_ids:
            deduped_item = dict(item)
            deduped_item["candidate_ids"] = clean_ids
            deduped_items.append(deduped_item)
    return deduped_items


def _merge_missed_candidate(
    sanitized_items: list[dict[str, Any]],
    candidate_id: str,
    candidate_by_id: Mapping[str, Any],
) -> bool:
    """将遗漏候选归并到同语义 item。"""
    missed_text = _candidate_text(candidate_id, candidate_by_id)
    for item in sanitized_items:
        if _has_semantic_overlap(missed_text, _item_text(item, candidate_by_id)):
            item["candidate_ids"].append(candidate_id)
            reason = str(item.get("reason", "")).strip()
            item["reason"] = f"{reason}；遗漏归并" if reason else "遗漏归并"
            logger.info(
                "结构化输出｜业务校验｜遗漏候选已归并｜候选ID=%s(candidate_id=%s)｜归并后候选=%s(candidate_ids=%s)",
                candidate_id,
                candidate_id,
                item["candidate_ids"],
                item["candidate_ids"],
            )
            return True
    return False


def _closest_item_intent(
    sanitized_items: list[dict[str, Any]],
    candidate_id: str,
    candidate_by_id: Mapping[str, Any],
) -> str:
    """为低置信度遗漏候选寻找最接近的已裁决意图。"""
    missed_text = _candidate_text(candidate_id, candidate_by_id)
    best_intent = ""
    best_score = 0.0
    for item in sanitized_items:
        similarity = _text_similarity(missed_text, _item_text(item, candidate_by_id))
        if similarity > best_score:
            best_score = similarity
            best_intent = str(item.get("intent_id", "")).strip()
    return best_intent if best_score > 0 else ""


def recover_missing_candidates(
    items: list[dict[str, Any]],
    candidate_by_id: Mapping[str, Any],
    valid_intent_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """补回 LLM 遗漏的候选。"""
    valid_intent_ids = valid_intent_ids or set()
    sanitized_items = [dict(item) for item in items]
    used_ids = {
        str(candidate_id).strip()
        for item in sanitized_items
        for candidate_id in item.get("candidate_ids", [])
        if str(candidate_id).strip()
    }
    missed_ids = [candidate_id for candidate_id in candidate_by_id if candidate_id not in used_ids]

    for candidate_id in missed_ids:
        if _merge_missed_candidate(sanitized_items, candidate_id, candidate_by_id):
            used_ids.add(candidate_id)
            continue

        missed_text = _candidate_text(candidate_id, candidate_by_id)
        if _is_semantically_incomplete_fragment(missed_text):
            logger.info("结构化输出｜业务校验｜遗漏候选为语义残片，已忽略｜候选ID=%s(candidate_id=%s)", candidate_id, candidate_id)
            used_ids.add(candidate_id)
            continue

        intent_id = _candidate_intent(candidate_id, candidate_by_id)
        score = _candidate_score(candidate_id, candidate_by_id)
        reason = "兜底-低置信度"
        if score < LOW_CONFIDENCE_INTENT_SCORE:
            corrected_intent = _closest_item_intent(sanitized_items, candidate_id, candidate_by_id)
            if corrected_intent:
                intent_id = corrected_intent
                reason = "低置信度跨意图修正"

        if not validate_intent_id(intent_id, valid_intent_ids or set()):
            logger.warning(
                "结构化输出｜业务校验｜遗漏候选意图无效，已忽略｜候选ID=%s(candidate_id=%s)｜意图=%s(intent_id=%s)",
                candidate_id,
                candidate_id,
                intent_id,
                intent_id,
            )
            used_ids.add(candidate_id)
            continue

        logger.warning(
            "结构化输出｜业务校验｜遗漏候选使用防御性兜底｜候选ID=%s(candidate_id=%s)｜意图=%s(intent_id=%s)｜原因=%s(reason=%s)",
            candidate_id,
            candidate_id,
            intent_id,
            intent_id,
            reason,
            reason,
        )
        sanitized_items.append({
            "candidate_ids": [candidate_id],
            "intent_id": intent_id,
            "reason": reason,
        })
        used_ids.add(candidate_id)

    return sanitized_items


def _extract_judge_items(parsed_output: Mapping[str, Any] | list[Any]) -> list[Any]:
    """从对象或列表中取得裁决项。"""
    if isinstance(parsed_output, Mapping):
        items = parsed_output.get("items", [])
    else:
        items = parsed_output
    return items if isinstance(items, list) else []


def sanitize_judge_items(
    parsed_output: Mapping[str, Any] | list[Any],
    candidate_by_id: Mapping[str, Any],
    valid_intent_ids: set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """校验并清洗查询拆解裁决项。"""
    sanitized_items: list[dict[str, Any]] = []
    for item in _extract_judge_items(parsed_output):
        if not isinstance(item, Mapping):
            continue

        intent_id = str(item.get("intent_id", "")).strip()
        if not validate_intent_id(intent_id, valid_intent_ids):
            logger.warning("结构化输出｜业务校验｜未知意图已忽略｜意图=%s(intent_id=%s)", intent_id, intent_id)
            continue

        clean_ids = validate_candidate_ids(item.get("candidate_ids", []), candidate_by_id)
        if not clean_ids:
            continue
        sanitized_items.append({
            "candidate_ids": clean_ids,
            "intent_id": intent_id,
            "reason": str(item.get("reason", "")).strip(),
        })

    sanitized_items = dedupe_candidate_assignments(sanitized_items)
    return recover_missing_candidates(sanitized_items, candidate_by_id, valid_intent_ids)


def sanitize_rewrite_items(
    parsed_output: Mapping[str, Any] | list[Any],
    candidate_by_id: Mapping[str, Any],
    valid_intent_ids: set[str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """校验并清洗查询改写项。"""
    raw_items = parsed_output.get("rewritten", []) if isinstance(parsed_output, Mapping) else parsed_output
    if not isinstance(raw_items, list):
        return []

    original_by_text = {
        _candidate_text(candidate_id, candidate_by_id).strip(): candidate_id
        for candidate_id in candidate_by_id
        if _candidate_text(candidate_id, candidate_by_id).strip()
    }
    results: list[dict[str, Any]] = []
    seen_originals: set[str] = set()
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        original = str(item.get("original", "")).strip()
        candidate_id = original_by_text.get(original)
        if not candidate_id or original in seen_originals:
            continue

        intent_id = str(item.get("intent_id", "")).strip()
        if not validate_intent_id(intent_id, valid_intent_ids):
            intent_id = _candidate_intent(candidate_id, candidate_by_id)
        rewritten = str(item.get("rewritten", "")).strip() or original
        if not validate_intent_id(intent_id, valid_intent_ids):
            continue

        results.append({
            "candidate_id": candidate_id,
            "original": original,
            "intent_id": intent_id,
            "rewritten": rewritten,
            "score": _candidate_score(candidate_id, candidate_by_id),
        })
        seen_originals.add(original)
    return results
