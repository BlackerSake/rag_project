import os
import sys
import asyncio
import json
import math
import re
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .state import State
from .schema import (
    direct_answer_prompt,
    clarify_question_prompt,
    fallback_response_prompt,
    summarization_prompt,
    chat_response_prompt,
    query_decomposition_judge_prompt,
    query_rewrite_prompt,
)
from .models import model
from .intent_manager import get_intent_manager
from .summary import compress_context
from .structured_output import (
    fallback_judge_items as structured_fallback_judge_items,
    parse_and_validate_judge_output_async,
)

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.knowledge_base import KnowledgeBase
from data.confidence import ConfidenceGate, ConfidenceHistory
from intent import (
    add_to_annotation_pool,
    detect_strong_negative_feedback,
    log_intent_gate_decision,
    route_after_intent_gate,
    update_intent_confidence_history,
    write_intent_gate_to_state,
)
from utils.logging_config import get_logger
from tools.manager import create_customer_service_react_agent

# 获取日志记录器
logger = get_logger(__name__)

# 初始化知识库
knowledge_base = KnowledgeBase()


def _float_env(name: str, default: float) -> float:
    """讀取浮點環境變量，格式錯誤時保守使用預設值。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("環境變量 %s=%s 不是有效浮點數，使用預設值 %.4f", name, raw_value, default)
        return default
    if not math.isfinite(value):
        logger.warning("環境變量 %s=%s 不是有限浮點數，使用預設值 %.4f", name, raw_value, default)
        return default
    return value


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIDENCE_SEED_PATH = _PROJECT_ROOT / "evaluation" / "dataset" / "confidence_margin_scores.json"
_RRF_DEGRADED_TOP1_THRESHOLD = 0.08
_confidence_history = ConfidenceHistory(max_size=100)
confidence_gate_instance = ConfidenceGate(
    history=_confidence_history,
    fallback_p25=_float_env("CONFIDENCE_MARGIN_FALLBACK_P25", 0.03),
    fallback_p75=_float_env("CONFIDENCE_MARGIN_FALLBACK_P75", 0.12),
)

REACT_TOOL_SYSTEM_PROMPT = (
    "你是智能客服的有狀態 ReAct 工具調度節點。你會收到本輪所有子問題、"
    "以及本輪/歷史已執行過的工具與Observation。\n"
    "你需要先整體規劃，再決定是否調用工具。可用工具只用於查詢即時或訂單型資料；"
    "若問題更適合知識庫回答，請不要調用工具。\n"
    "當用戶詢問快遞、物流進度、包裹到哪裡且提供單號時，應調用 query_logistics。\n"
    "如果某個物流單號已在已執行工具記錄中查過，禁止再次調用工具，直接復用已有Observation。\n"
    "如果本輪多個子問題是同一單號，只需查一次，其他子問題復用結果。\n"
    "如果缺少物流單號，不要編造單號，也不要調用工具。\n"
    "最後請簡要說明哪些問題已用工具處理，哪些需要知識庫。"
)

react_tool_agent = create_customer_service_react_agent(
    model,
    system_prompt=REACT_TOOL_SYSTEM_PROMPT,
    enable_logging=True,
)

# 获取意图管理器
intent_manager = None


def _format_knowledge_preview(content: str, max_length: int = 120) -> str:
    """压缩知识库内容，避免日志单行过长。"""
    preview = " ".join(content.split())
    if len(preview) > max_length:
        return preview[:max_length] + "..."
    return preview


def _format_conversation_history(messages: list, max_messages: int = 12) -> str:
    """將近期對話轉成文字，供不支援 MessagesPlaceholder 的 prompt 變量使用。"""
    recent_messages = messages[-max_messages:] if messages else []
    lines = []
    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            role = "用戶"
        elif isinstance(msg, AIMessage):
            role = "助手"
        elif isinstance(msg, ToolMessage):
            role = "工具"
        else:
            role = msg.__class__.__name__
        content = str(getattr(msg, "content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "無"


def _get_clarify_parameter_message(result: dict) -> str:
    """兼容舊字段並為常見缺失槽位生成可讀提示。"""
    message = (
        result.get("clarify_parameters")
        or result.get("clarify_parameters")
        or ""
    )
    if message:
        return str(message).strip()

    missing_slots = result.get("missing_slots") or []
    if "tracking_number" in missing_slots:
        return "請提供需要查詢的物流單號。"
    return ""


async def get_manager():
    global intent_manager
    if intent_manager is None:
        intent_manager = await get_intent_manager()
    return intent_manager


def _aggregate_intent_gate_decisions(user_query: str, gate_results: list[dict]) -> dict:
    """聚合单个或多个子问题的意图门控结果。"""
    decisions = [
        item.get("decision", item)
        for item in gate_results
        if isinstance(item, dict) and isinstance(item.get("decision", item), dict)
    ]
    if not decisions:
        decision = {
            "query": user_query,
            "intent_candidates": [],
            "intent_id": None,
            "intent_score": 0.0,
            "intent_margin": 0.0,
            "intent_confidence_level": "LOW",
            "intent_gate_action": "FALLBACK",
            "intent_gate_reason": "no_gate_decision",
            "clarification_question": "我还不确定你的问题类型，可以再补充一下你想咨询的内容吗？",
        }
    else:
        decision = next((item for item in decisions if item.get("intent_gate_action") == "FALLBACK"), None)
        if decision is None:
            decision = next((item for item in decisions if item.get("intent_gate_action") == "CLARIFY"), None)
        if decision is None:
            decision = decisions[0]

    state_update = write_intent_gate_to_state({}, decision)
    state_update["query"] = user_query
    state_update["intent_candidate_results"] = gate_results
    state_update["final_route"] = route_after_intent_gate(state_update)

    log_intent_gate_decision(state_update)
    update_intent_confidence_history(state_update)
    return state_update


def _resolve_confidence_seed_path(filepath: str | os.PathLike | None = None) -> Path:
    """解析置信度冷啟動資料路徑，支援環境變量與相對路徑。"""
    raw_path = filepath or os.getenv("CONFIDENCE_OFFLINE_PATH")
    path = Path(raw_path) if raw_path else _DEFAULT_CONFIDENCE_SEED_PATH
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _extract_offline_confidence_records(payload: dict) -> list:
    """從離線評測JSON取出 records 陣列，兼容只輸出 margin 子報告的格式。"""
    records = payload.get("records")
    if isinstance(records, list):
        return records

    margin_report = payload.get("margin")
    if isinstance(margin_report, dict) and isinstance(margin_report.get("records"), list):
        return margin_report["records"]

    return []


def seed_confidence_from_offline(filepath: str | os.PathLike | None = None) -> int:
    """使用離線 MARGIN 分數預填 confidence gate 滑動窗口。"""
    path = _resolve_confidence_seed_path(filepath)
    if not path.exists():
        logger.info("confidence gate 冷啟動資料不存在，略過預填: %s", path)
        return 0

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("confidence gate 冷啟動資料讀取失敗: %s, error=%s", path, exc)
        return 0

    if not isinstance(payload, dict):
        logger.warning("confidence gate 冷啟動資料格式錯誤，根節點必須是 object: %s", path)
        return 0

    records = _extract_offline_confidence_records(payload)
    loaded_count = 0
    skipped_count = 0
    for record in records:
        if not isinstance(record, dict) or "confidence_score" not in record:
            skipped_count += 1
            continue

        try:
            _confidence_history.update(record["confidence_score"])
            loaded_count += 1
        except (TypeError, ValueError) as exc:
            skipped_count += 1
            logger.debug("略過無效 confidence_score: record=%s, error=%s", record, exc)

    thresholds = confidence_gate_instance.calibrate_thresholds()
    logger.info(
        "confidence gate 冷啟動完成: path=%s loaded=%d skipped=%d p25=%.4f p75=%.4f sample=%d",
        path,
        loaded_count,
        skipped_count,
        thresholds.p25,
        thresholds.p75,
        thresholds.sample_count,
    )
    return loaded_count


async def detect_topic(state: State) -> State:
    """检测对话主题"""
    # 获取最新消息
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")

    logger.info(f"检测对话主题: {latest_message.content}")

    # 获取异步的意图管理器
    manager = await get_manager()

    decision = await manager.match_intent_with_gate(latest_message.content)
    state_update = _aggregate_intent_gate_decisions(
        str(latest_message.content),
        [{"sub_question": str(latest_message.content), "decision": decision}],
    )
    logger.info(
        "意图门控结果: intent=%s, score=%.4f, margin=%.4f, action=%s",
        state_update.get("intent_id"),
        state_update.get("intent_score", 0.0),
        state_update.get("intent_margin", 0.0),
        state_update.get("intent_gate_action"),
    )
    return state_update

def _extract_json_object(text: str) -> dict:
    """从LLM输出中提取JSON对象，兼容误包裹的代码块。"""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(raw[start:end + 1])


def _build_intent_catalog(manager) -> str:
    intents = manager.get_all_intents() if hasattr(manager, "get_all_intents") else {}
    lines = []
    for intent_id, info in sorted(intents.items()):
        name = info.get("name", "")
        description = info.get("description", "")
        lines.append(f"- {intent_id}: {name}。{description}")
    return "\n".join(lines)


def _fallback_judge_items(candidate_by_id: dict[str, tuple[str, str, float]]) -> list[dict]:
    """LLM 裁决失败时保守兜底：每个候选句独立保留，避免合并造成信息丢失。"""
    return structured_fallback_judge_items(candidate_by_id)


_LOW_CONFIDENCE_INTENT_SCORE = 0.55
_DECOMPOSITION_TOPIC_KEYWORDS = {
    "logistics": ["快递", "快遞", "物流", "包裹", "单号", "單號", "丢", "丟", "没到", "未到", "到哪", "哪了"],
    "return": ["退货", "退貨", "换货", "換貨", "退换", "退換", "售后", "售後"],
    "refund": ["退款", "退钱", "退錢", "到账", "到賬", "多久", "几天", "幾天"],
    "repair": ["维修", "維修", "修理", "保修", "质保", "質保"],
    "order": ["订单", "訂單", "商品", "商家", "平台"],
}


def _candidate_text(candidate_id: str, candidate_by_id: dict[str, tuple[str, str, float]]) -> str:
    """取得候選原文，缺失時回傳空字串。"""
    candidate = candidate_by_id.get(candidate_id)
    return str(candidate[0]) if candidate else ""


def _candidate_score(candidate_id: str, candidate_by_id: dict[str, tuple[str, str, float]]) -> float:
    """取得候選向量意圖分數，格式異常時視為低置信度。"""
    candidate = candidate_by_id.get(candidate_id)
    if not candidate:
        return 0.0
    try:
        score = float(candidate[2])
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _candidate_intent(candidate_id: str, candidate_by_id: dict[str, tuple[str, str, float]]) -> str:
    """取得候選向量意圖。"""
    candidate = candidate_by_id.get(candidate_id)
    return str(candidate[1]).strip() if candidate else ""


def _decomposition_identifiers(text: str) -> set[str]:
    """抽取單號、訂單號等高置信識別符。"""
    normalized = str(text).lower()
    return set(re.findall(r"[a-z]{0,4}\d{3,}[a-z0-9]*|\d{3,}", normalized))


def _decomposition_topics(text: str) -> set[str]:
    """以少量業務詞判斷候選片段所屬語義主題。"""
    return {
        topic
        for topic, keywords in _DECOMPOSITION_TOPIC_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    }


def _decomposition_terms(text: str) -> set[str]:
    """建立輕量文字特徵，用於兜底時尋找最相近的已裁決項。"""
    normalized = re.sub(r"\s+", "", str(text).lower())
    terms = _decomposition_identifiers(normalized) | _decomposition_topics(normalized)
    chinese_or_word = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z]{2,}", normalized)
    terms.update(chinese_or_word)
    if len(normalized) >= 2:
        terms.update(normalized[i:i + 2] for i in range(len(normalized) - 1))
    elif normalized:
        terms.add(normalized)
    return terms


def _has_semantic_overlap(left_text: str, right_text: str) -> bool:
    """判斷兩段候選文本是否應視為同一業務需求的補充。"""
    left_ids = _decomposition_identifiers(left_text)
    right_ids = _decomposition_identifiers(right_text)
    if left_ids and right_ids and left_ids & right_ids:
        return True

    left_topics = _decomposition_topics(left_text)
    right_topics = _decomposition_topics(right_text)
    return bool(left_topics and right_topics and left_topics & right_topics)


def _is_semantically_incomplete_fragment(text: str) -> bool:
    """識別不應被單獨補回的語義殘片。"""
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


def _item_text(item: dict, candidate_by_id: dict[str, tuple[str, str, float]]) -> str:
    """合併 item 內候選原文，供相似度與歸併判斷使用。"""
    candidate_ids = item.get("candidate_ids", [])
    if not isinstance(candidate_ids, list):
        return ""
    return "，".join(_candidate_text(str(candidate_id).strip(), candidate_by_id) for candidate_id in candidate_ids)


def _text_similarity(left_text: str, right_text: str) -> float:
    """使用 Jaccard 特徵重疊估算文本相近程度。"""
    left_terms = _decomposition_terms(left_text)
    right_terms = _decomposition_terms(right_text)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _merge_missed_candidate(
    sanitized_items: list[dict],
    candidate_id: str,
    candidate_by_id: dict[str, tuple[str, str, float]],
) -> bool:
    """若遺漏片段與已裁決 item 屬於同一語義需求，直接歸併。"""
    missed_text = _candidate_text(candidate_id, candidate_by_id)
    for item in sanitized_items:
        if _has_semantic_overlap(missed_text, _item_text(item, candidate_by_id)):
            item["candidate_ids"].append(candidate_id)
            reason = str(item.get("reason", "")).strip()
            item["reason"] = f"{reason}；遺漏歸併" if reason else "遺漏歸併"
            logger.info("裁決遺漏candidate_id已歸併: %s -> %s", candidate_id, item["candidate_ids"])
            return True
    return False


def _closest_item_intent(
    sanitized_items: list[dict],
    candidate_id: str,
    candidate_by_id: dict[str, tuple[str, str, float]],
) -> str:
    """為低置信度遺漏片段尋找文本最接近的已裁決意圖。"""
    missed_text = _candidate_text(candidate_id, candidate_by_id)
    best_intent = ""
    best_score = 0.0
    for item in sanitized_items:
        similarity = _text_similarity(missed_text, _item_text(item, candidate_by_id))
        if similarity > best_score:
            best_score = similarity
            best_intent = str(item.get("intent_id", "")).strip()
    return best_intent if best_score > 0 else ""


def _sanitize_judge_items(
    raw_items: list,
    candidate_by_id: dict[str, tuple[str, str, float]],
    valid_intents: set[str],
) -> list[dict]:
    """校驗裁決items，只保留合法candidate_ids與intent_id。"""
    sanitized_items = []
    used_ids = set()

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        intent_id = str(item.get("intent_id", "")).strip()
        if not intent_id:
            continue
        if valid_intents and intent_id not in valid_intents:
            logger.warning("LLM返回未知意图，已忽略: %s", intent_id)
            continue

        candidate_ids = item.get("candidate_ids", [])
        if not isinstance(candidate_ids, list):
            continue

        clean_ids = []
        for candidate_id in candidate_ids:
            cid = str(candidate_id).strip()
            if cid not in candidate_by_id:
                logger.warning("LLM返回未知candidate_id，已忽略: %s", cid)
                continue
            if cid in used_ids:
                logger.info("LLM返回重复candidate_id，保留首次结果: %s", cid)
                continue
            clean_ids.append(cid)
            used_ids.add(cid)

        if clean_ids:
            sanitized_items.append({
                "candidate_ids": clean_ids,
                "intent_id": intent_id,
                "reason": str(item.get("reason", "")).strip(),
            })

    missed_ids = [candidate_id for candidate_id in candidate_by_id if candidate_id not in used_ids]
    for candidate_id in missed_ids:
        if _merge_missed_candidate(sanitized_items, candidate_id, candidate_by_id):
            used_ids.add(candidate_id)
            continue

        missed_text = _candidate_text(candidate_id, candidate_by_id)
        if _is_semantically_incomplete_fragment(missed_text):
            logger.info("裁決遺漏candidate_id為語義殘片，已忽略: %s", candidate_id)
            used_ids.add(candidate_id)
            continue

        intent_id = _candidate_intent(candidate_id, candidate_by_id)
        score = _candidate_score(candidate_id, candidate_by_id)
        reason = "兜底-低置信度"

        if score < _LOW_CONFIDENCE_INTENT_SCORE:
            corrected_intent = _closest_item_intent(sanitized_items, candidate_id, candidate_by_id)
            if corrected_intent:
                intent_id = corrected_intent
                reason = "低置信度跨意圖修正"

        if valid_intents and intent_id not in valid_intents:
            logger.warning("遺漏candidate_id的意圖無效，已忽略: %s -> %s", candidate_id, intent_id)
            used_ids.add(candidate_id)
            continue

        logger.warning("裁決遺漏candidate_id，使用防禦性兜底: %s, intent=%s, reason=%s", candidate_id, intent_id, reason)
        sanitized_items.append({
            "candidate_ids": [candidate_id],
            "intent_id": intent_id,
            "reason": reason,
        })
        used_ids.add(candidate_id)

    return sanitized_items


async def _judge_final_sub_questions(
    user_query: str,
    candidate_by_id: dict[str, tuple[str, str, float]],
    manager,
    summary: str = "无",
    conversation_history: str = "無",
) -> list[dict]:
    if not candidate_by_id:
        return []
    if query_decomposition_judge_prompt is None:
        logger.warning("query_decomposition_judge_prompt 未加载，使用规则兜底")
        return _fallback_judge_items(candidate_by_id)

    valid_intents = set(manager.get_all_intents().keys()) if hasattr(manager, "get_all_intents") else set()
    candidate_payload = [
        {
            "candidate_id": candidate_id,
            "original_question": sub_q,
            "intent_id": intent_id,
        }
        for candidate_id, (sub_q, intent_id, _) in candidate_by_id.items()
    ]

    chain = query_decomposition_judge_prompt | model
    response = await chain.ainvoke({
        "user_query": user_query,
        "summary": summary,
        "conversation_history": conversation_history,
        "intent_catalog": _build_intent_catalog(manager),
        "candidate_json": json.dumps(candidate_payload, ensure_ascii=False, indent=2),
    })
    content = response.content if hasattr(response, "content") else str(response)
    logger.info("LLM查询拆解裁决原始输出: %s", content)

    judge_items = await parse_and_validate_judge_output_async(content, candidate_by_id, valid_intents)
    if not judge_items:
        raise ValueError("LLM查询拆解输出无有效子问题")
    return judge_items


def split_user_input(text: str) -> list[str]:
    """
    对用户原始输入做轻量句子拆分
    核心逻辑：切句 → 过滤补充句 → 合并残句
    """
    # 1. 按标点切句
    pattern = r'[。？！?!；;，.]+'
    parts = [p.strip() for p in re.split(pattern, text) if p.strip()]

    # 没切出多句，直接返回原始输入
    if len(parts) <= 1:
        return [text]

    # 2. 识别"补充句"模式（不能独立表达意图的句子）
    dependent_patterns = [
        r'^帮我.{0,4}(查|看|处理|确认|一下)',  # "帮我查一下" 无宾语
        r'^(麻烦|劳烦|请帮).{0,6}$',            # 纯礼貌用语
        r'^(好的|谢谢|知道了|明白|嗯|哦)',
        r'^\d{5,}$',                              # 纯数字（单号）
        r'^(我的|这个|那个).{0,4}$',             # 纯指代
        r'^.{1,4}$',                              # 极短片段
    ]

    result = []
    for part in parts:
        is_dependent = any(re.search(p, part) for p in dependent_patterns)
        if is_dependent:
            # 补充句合并到上一句
            if result:
                result[-1] = result[-1] + "，" + part
            # 如果是第一句就是补充句，合并到整体（罕见情况）
        else:
            result.append(part)

    return result if result else [text]


async def decompose_query(state: State) -> State:
    """查询拆解节点 - 直接对用户输入做句子拆分"""
    messages = state.get("messages", [])
    latest_message = messages[-1] if messages else HumanMessage(content="")
    user_query = latest_message.content
    summary = state.get("summary", "无")
    conversation_history = _format_conversation_history(messages)

    logger.info(
        "Decomposer Node: 分析查询 - %s, summary: %s, 历史消息数: %s",
        user_query, summary, len(messages)
    )
    if detect_strong_negative_feedback(user_query):
        add_to_annotation_pool(state, user_query)
        logger.info("检测到强负反馈，已写入意图标注池")

    try:
        # 第一步：直接对用户输入切句，不再检索向量库
        sentences = split_user_input(user_query)
        logger.info(f"句子拆分结果: {sentences}")

        # 第二步：对每个句子召回Top-k候选并执行IntentGate
        manager = await get_manager()
        sub_question_intents = []
        intent_gate_results = []

        for i, sent in enumerate(sentences, 1):
            decision = await manager.match_intent_with_gate(sent)
            intent_id = decision.get("intent_id")
            score = decision.get("intent_score", 0.0)
            sub_question_intents.append((sent, intent_id, score))
            intent_gate_results.append({
                "sub_question": sent,
                "decision": decision,
            })
            logger.info(
                "  子问题%d: %s -> 意图: %s (相似度: %.4f, margin: %.4f, action: %s)",
                i,
                sent,
                intent_id,
                score,
                decision.get("intent_margin", 0.0),
                decision.get("intent_gate_action"),
            )

        candidate_by_id = {
            f"q{i}": (sub_q, intent_id, score)
            for i, (sub_q, intent_id, score) in enumerate(sub_question_intents, 1)
        }
        state_update = _aggregate_intent_gate_decisions(user_query, intent_gate_results)

        if state_update.get("intent_gate_action") == "FALLBACK":
            judge_items = _fallback_judge_items(candidate_by_id)
            logger.info(
                "IntentGate触发兜底，跳过LLM查询拆解决策: action=%s, reason=%s",
                state_update.get("intent_gate_action"),
                state_update.get("intent_gate_reason"),
            )
            return {
                **state_update,
                "sub_questions": [item[0] for item in sub_question_intents],
                "is_complex_query": len(sub_question_intents) > 1,
                "multi_intent_results": sub_question_intents,
                "judge_items": judge_items,
                "candidate_by_id": candidate_by_id,
                "decompose_skipped": False,
            }

        # 第三步：由LLM基于候选子问题和意图目录裁决保留/去重/合并关系
        try:
            judge_items = await _judge_final_sub_questions(
                user_query,
                candidate_by_id,
                manager,
                summary=summary,
                conversation_history=conversation_history,
            )
            logger.info("LLM查询拆解裁决完成: %s", judge_items)
        except Exception as llm_error:
            logger.error("LLM查询拆解裁决失败，使用规则兜底: %s", llm_error)
            judge_items = _fallback_judge_items(candidate_by_id)

        logger.info("查询拆解完成: %s 个候选，%s 个裁决item，后续交由 llm_rewrite 展开重写", len(candidate_by_id), len(judge_items))

        return {
            **state_update,
            "sub_questions": [item[0] for item in sub_question_intents],
            "is_complex_query": len(sub_question_intents) > 1,
            "multi_intent_results": sub_question_intents,
            "judge_items": judge_items,
            "candidate_by_id": candidate_by_id,
            "decompose_skipped": False,
        }

    except Exception as e:
        logger.error(f"查询拆解失败: {str(e)}")
        return {
            "sub_questions": [user_query],
            "is_complex_query": False,
            "multi_intent_results": [],
            "judge_items": [],
            "candidate_by_id": {},
            "intent_candidate_results": [],
            "decompose_skipped": True
        }


async def intent_retrieval_node(state: State) -> State:
    """意图候选召回节点，兼容现有查询拆解链路。"""
    return await decompose_query(state)


async def intent_gate_node(state: State) -> State:
    """意图门控节点，缺少门控结果时补充执行一次判断。"""
    if state.get("intent_gate_action"):
        return {}

    messages = state.get("messages", [])
    latest_message = messages[-1] if messages else HumanMessage(content="")
    user_query = str(getattr(latest_message, "content", ""))
    manager = await get_manager()
    decision = await manager.match_intent_with_gate(user_query)
    return _aggregate_intent_gate_decisions(
        user_query,
        [{"sub_question": user_query, "decision": decision}],
    )


def _expand_judge_candidates(
    judge_items: list[dict],
    candidate_by_id: dict[str, tuple[str, str, float]],
) -> list[dict]:
    candidates = []
    for item in judge_items:
        intent_id = str(item.get("intent_id", "")).strip()
        candidate_ids = item.get("candidate_ids", [])
        if not intent_id or not isinstance(candidate_ids, list):
            continue

        for candidate_id in candidate_ids:
            cid = str(candidate_id).strip()
            if cid not in candidate_by_id:
                logger.warning("llm_rewrite 跳过未知candidate_id: %s", cid)
                continue

            original, original_intent_id, score = candidate_by_id[cid]
            candidates.append({
                "candidate_id": cid,
                "original": original,
                "intent_id": intent_id or original_intent_id,
                "score": score,
            })

    return candidates


def _fallback_rewrite_results(candidates: list[dict]) -> list[tuple[str, str, float]]:
    return [
        (candidate["original"], candidate["intent_id"], candidate.get("score", 1.0))
        for candidate in candidates
        if candidate.get("original") and candidate.get("intent_id")
    ]


async def llm_rewrite(state: State) -> State:
    """按裁決items展開candidate_ids，將每個原句獨立改寫成完整自然句。"""
    judge_items = state.get("judge_items", [])
    candidate_by_id = state.get("candidate_by_id", {})
    summary = state.get("summary", "无")
    conversation_history = _format_conversation_history(state.get("messages", []))

    if not judge_items or not candidate_by_id:
        logger.warning("llm_rewrite 缺少judge_items或candidate_by_id，透传拆解结果")
        fallback_results = state.get("multi_intent_results", [])
        return {
            "sub_questions": [item[0] for item in fallback_results],
            "is_complex_query": len(fallback_results) > 1,
            "multi_intent_results": fallback_results,
        }

    candidates = _expand_judge_candidates(judge_items, candidate_by_id)
    if not candidates:
        logger.warning("llm_rewrite 展开后无候选，透传拆解结果")
        fallback_results = state.get("multi_intent_results", [])
        return {
            "sub_questions": [item[0] for item in fallback_results],
            "is_complex_query": len(fallback_results) > 1,
            "multi_intent_results": fallback_results,
        }

    if query_rewrite_prompt is None:
        logger.warning("query_rewrite_prompt 未加载，使用原句兜底")
        final_results = _fallback_rewrite_results(candidates)
    else:
        manager = await get_manager()
        rewrite_payload = [
            {
                "candidate_id": candidate["candidate_id"],
                "original": candidate["original"],
                "intent_id": candidate["intent_id"],
            }
            for candidate in candidates
        ]

        try:
            chain = query_rewrite_prompt | model
            response = await chain.ainvoke({
                "summary": summary,
                "conversation_history": conversation_history,
                "intent_catalog": _build_intent_catalog(manager),
                "candidates_json": json.dumps(rewrite_payload, ensure_ascii=False, indent=2),
            })
            content = response.content if hasattr(response, "content") else str(response)
            logger.info("llm_rewrite 原始输出: %s", content)
            payload = _extract_json_object(content)
            rewritten_items = payload.get("rewritten", [])
            if not isinstance(rewritten_items, list):
                raise ValueError("query_rewrite 输出缺少rewritten数组")

            score_by_original_intent = {
                (candidate["original"], candidate["intent_id"]): candidate.get("score", 1.0)
                for candidate in candidates
            }
            final_results = []
            for item in rewritten_items:
                if not isinstance(item, dict):
                    continue
                rewritten = str(item.get("rewritten", "")).strip()
                intent_id = str(item.get("intent_id", "")).strip()
                original = str(item.get("original", "")).strip()
                if rewritten and intent_id:
                    score = score_by_original_intent.get((original, intent_id), 1.0)
                    final_results.append((rewritten, intent_id, score))

            if not final_results:
                raise ValueError("query_rewrite 输出无有效改写结果")
        except Exception as rewrite_error:
            logger.error("llm_rewrite 失败，使用原句兜底: %s", rewrite_error)
            final_results = _fallback_rewrite_results(candidates)

    sub_questions = [result[0] for result in final_results]
    logger.info("llm_rewrite 完成: %s 个子问题", len(sub_questions))
    for i, (sub_q, intent_id, score) in enumerate(final_results, 1):
        logger.info("  最终子问题%d: %s -> 意图: %s (相似度: %.4f)", i, sub_q, intent_id, score)

    return {
        "sub_questions": sub_questions,
        "multi_intent_results": final_results,
        "is_complex_query": len(sub_questions) > 1,
    }


async def retrieve_knowledge_multi(state: State) -> State:
    """
    多意图知识库检索节点
    对拆解后的每个子问题分别进行意图匹配和知识库检索
    """
    # 只检索 task_dispatcher 标记为未被工具解决的子问题。
    processed_results = state.get("processed_results", [])
    knowledge_intent_results = [
        (
            result["sub_question"],
            result.get("intent_id"),
            result.get("intent_score", 0.0),
        )
        for result in processed_results
        if not result.get("is_tool", False)
        and not result.get("needs_clarification", False)
        and result.get("sub_question")
    ]

    # 兼容未经过 task_dispatcher 的旧调用路径；如果已有processed_results，
    # 但没有未被工具解决的项，则不能回退到全量multi_intent_results。
    if not knowledge_intent_results and not processed_results:
        knowledge_intent_results = state.get("multi_intent_results", [])

    sub_questions = [item[0] for item in knowledge_intent_results]

    if not sub_questions:
        # 如果没有子问题，直接返回空结果
        return {
            "knowledge_results": [],
            "highest_score": 0.0,
            "intent_id": None,
            "multi_intent_results": state.get("multi_intent_results", [])
        }

    logger.info(f"多意图检索: 处理 {len(sub_questions)} 个子问题")

    multi_results = []

    # 只使用未被工具解决的子问题结果
    for i, (sub_q, intent_id, score) in enumerate(knowledge_intent_results):
        logger.info(f"处理知识库子问题 {i+1}/{len(knowledge_intent_results)}: {sub_q}")
        logger.info(f"子问题意图匹配: {intent_id} (相似度: {score:.4f})")

        # 跳过D1意图的检索
        if intent_id == "D1":
            logger.info(f"子问题{i+1}为闲聊意图，跳过检索")
            continue

        # 知识库检索
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: knowledge_base.search(
                sub_q,
                k=5,
                expected_intent_id=intent_id,
                evaluation_context={"node": "retrieve_knowledge_multi", "sub_question_index": i + 1},
            )
        )

        # 过滤与当前意图相关的结果
        filtered_results = []
        for doc, doc_score in results:
            metadata = doc.metadata
            if ('intent' in metadata and metadata['intent'] == intent_id) or \
               ('intent_id' in metadata and metadata['intent_id'] == intent_id):
                filtered_results.append((doc, doc_score))

        if not filtered_results:
            filtered_results = results[:3]  # 取前3个

        # 排序并取Top3
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:3]

        # 记录结果
        sub_result = {
            "sub_question": sub_q,
            "intent_id": intent_id,
            "intent_score": score,
            "knowledge_results": [(doc.page_content, doc_score) for doc, doc_score in sorted_results],
            "highest_score": sorted_results[0][1] if sorted_results else 0.0
        }
        multi_results.append(sub_result)

        logger.info(f"子问题{i+1}检索结果: {len(sorted_results)} 条记录，最高分数: {sub_result['highest_score']:.4f}")
        for rank, (doc, doc_score) in enumerate(sorted_results, 1):
            content_preview = _format_knowledge_preview(doc.page_content)
            logger.info(
                f"  子问题{i+1} Top{rank} (分数: {doc_score:.4f}, 意图: {intent_id}): {content_preview}"
            )

    # 聚合所有结果
    all_knowledge = []
    highest_score = 0.0
    primary_intent = None

    for result in multi_results:
        all_knowledge.extend(result["knowledge_results"])
        if result["highest_score"] > highest_score:
            highest_score = result["highest_score"]
            primary_intent = result["intent_id"]

    # 去重并保持顺序
    seen = set()
    unique_knowledge = []
    for content, score in all_knowledge:
        if content not in seen:
            seen.add(content)
            unique_knowledge.append((content, score))

    # 聚合后的多子问题结果需要按全局分数排序，否则 confidence_gate 会读取到错误的Top1/Top2。
    final_knowledge = sorted(unique_knowledge, key=lambda item: item[1], reverse=True)[:3]

    logger.info(f"多意图检索完成: 共 {len(final_knowledge)} 条唯一记录，最高分数: {highest_score:.4f}")

    return {
        "knowledge_results": final_knowledge,
        "highest_score": highest_score,
        "intent_id": primary_intent,
        "current_topic": primary_intent,  # 添加current_topic字段，与intent_id保持一致
        "multi_intent_results": state.get("multi_intent_results", []),
        "knowledge_multi_results": multi_results
    }

async def retrieve_knowledge(state: State) -> State:
    """检索知识库"""
    # 获取最新消息
    latest_message = state["messages"][-1] if state["messages"] else HumanMessage(content="")

    logger.info(f"检索知识库: {latest_message.content}")

    # 获取需要查询知识库的子问题；工具已解决的子问题不再检索。
    processed_results = state.get("processed_results", [])
    knowledge_targets = [
        result for result in processed_results
        if not result.get("is_tool", False)
        and not result.get("needs_clarification", False)
        and result.get("sub_question")
    ]

    if knowledge_targets:
        first_target = knowledge_targets[0]
        query = first_target["sub_question"]
        intent_id = first_target.get("intent_id") or state.get("intent_id")
    elif processed_results:
        logger.info("检索节点未收到需要知识库处理的子问题，返回空知识结果")
        return {
            "knowledge_results": [],
            "highest_score": 0.0,
            "intent_id": state.get("intent_id")
        }
    else:
        query = latest_message.content
        intent_id = state.get("intent_id")

    logger.info(f"使用意图ID: {intent_id} 进行检索")

    # 异步搜索知识库（不使用过滤表达式，避免字段不存在的问题）
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: knowledge_base.search(
            query,
            k=10,
            expected_intent_id=intent_id,
            evaluation_context={"node": "retrieve_knowledge"},
        )  # 搜索更多结果，以便后续过滤
    )

    # 根据意图ID进行过滤
    if intent_id == "D1":
        logger.info("意图为D1(闲聊)，跳过检索")
        # 直接返回空结果
        knowledge_results = []
        highest_score = 0.0
        logger.info("意图为D1,知识库不检索")
    else:
        # 过滤结果：只保留与当前意图相关的结果
        filtered_results = []
        for doc, score in results:
            # 检查文档的metadata中是否包含intent或intent_id字段，并且与当前意图匹配
            metadata = doc.metadata
            if ('intent' in metadata and metadata['intent'] == intent_id) or \
               ('intent_id' in metadata and metadata['intent_id'] == intent_id):
                filtered_results.append((doc, score))

        # 如果没有过滤结果，使用原始结果（避免无结果的情况）
        if not filtered_results:
            filtered_results = results

        # 按分数排序并提取前3条
        sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:3]
        # 保存文档内容和分数
        knowledge_results = [(result[0].page_content, result[1]) for result in sorted_results]
        # 提取最高分数
        highest_score = sorted_results[0][1] if sorted_results else 0.0

        # 记录检索结果的具体内容
        logger.info(f"知识库检索结果: {len(sorted_results)} 条记录，最高分数: {highest_score}")
        for i, (doc, score) in enumerate(sorted_results, 1):
            # 截取前100个字符作为预览
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"  Top{i} (分数: {score:.4f}): {content_preview}")

    return {
        "knowledge_results": knowledge_results,
        "highest_score": highest_score,
        "intent_id": intent_id
    }


async def confidence_gate(state: State) -> State:
    """根據檢索分數的 MARGIN 信號決定回答、提示或澄清路由。"""
    knowledge_results = state.get("knowledge_results") or []

    if not knowledge_results:
        decision = confidence_gate_instance.decide(0.0)
        logger.info(
            "confidence_gate: margin=%.4f decision=%s p25=%.4f p75=%.4f sample=%d reason=empty_results",
            decision["confidence_score"],
            "LOW",
            decision["p25"],
            decision["p75"],
            decision["sample_count"],
        )
        return {
            "confidence_decision": "LOW",
            "confidence_score": 0.0,
        }

    try:
        top1_score = float(knowledge_results[0][1])
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning("confidence_gate 讀取 Top-1 分數失敗，轉入 LOW: %s", exc)
        return {
            "confidence_decision": "LOW",
            "confidence_score": 0.0,
        }

    if not math.isfinite(top1_score):
        logger.warning("confidence_gate Top-1 分數不是有限值，轉入 LOW: %s", top1_score)
        return {
            "confidence_decision": "LOW",
            "confidence_score": 0.0,
        }

    if top1_score < _RRF_DEGRADED_TOP1_THRESHOLD:
        placeholder_score = _confidence_history.update(confidence_gate_instance.fallback_p25)
        thresholds = confidence_gate_instance.calibrate_thresholds()
        logger.info(
            "confidence_gate: margin=%.4f decision=%s p25=%.4f p75=%.4f sample=%d reason=rrf_degraded top1=%.4f",
            placeholder_score,
            "MEDIUM",
            thresholds.p25,
            thresholds.p75,
            thresholds.sample_count,
            top1_score,
        )
        return {
            "confidence_decision": "MEDIUM",
            "confidence_score": placeholder_score,
        }

    try:
        top2_score = float(knowledge_results[1][1]) if len(knowledge_results) >= 2 else 0.0
    except (TypeError, ValueError, IndexError) as exc:
        logger.warning("confidence_gate 讀取 Top-2 分數失敗，使用 0.0: %s", exc)
        top2_score = 0.0

    if not math.isfinite(top2_score):
        logger.warning("confidence_gate Top-2 分數不是有限值，使用 0.0: %s", top2_score)
        top2_score = 0.0

    margin = max(top1_score - top2_score, 0.0)
    _confidence_history.update(margin)
    decision = confidence_gate_instance.decide(margin)
    logger.info(
        "confidence_gate: margin=%.4f decision=%s p25=%.4f p75=%.4f sample=%d top1=%.4f top2=%.4f",
        decision["confidence_score"],
        decision["decision"],
        decision["p25"],
        decision["p75"],
        decision["sample_count"],
        top1_score,
        top2_score,
    )

    return {
        "confidence_decision": decision["decision"],
        "confidence_score": decision["confidence_score"],
    }


async def direct_answer(state: State) -> State:
    """直接回答节点"""
    # 构建提示
    knowledge_context = "\n".join([item[0] for item in state["knowledge_results"]]) if state.get("knowledge_results") else "无相关知识"
    summary = state.get("summary", "无")
    confidence_decision = state.get("confidence_decision", "HIGH")
    if confidence_decision == "MEDIUM":
        knowledge_context = (
            f"{knowledge_context}\n"
            "提示：以上知識庫內容的匹配置信度為中等，回答時請標示「僅供參考」，"
            "並避免把不確定資訊說成已確認事實。"
        )

    # 获取任务分发的处理结果
    processed_results = state.get("processed_results", [])

    # 构建工具执行结果的上下文
    tool_results_context = ""
    if processed_results:
        tool_results_context = "\n工具执行结果:\n"
        for result in processed_results:
            if result.get("is_tool"):
                if result.get("result"):
                    tool_results_context += f"- 子问题: {result['sub_question']}\n  结果: {result['result']}\n"
                    if result.get("raw_tool_results"):
                        raw_results = "\n  原始工具返回: ".join(result["raw_tool_results"])
                        tool_results_context += f"  原始工具返回: {raw_results}\n"
                elif result.get("error"):
                    tool_results_context += f"- 子问题: {result['sub_question']}\n  错误: {result['error']}\n"

    logger.info(f"进入直接回答节点，主题: {state.get('current_topic', '未知')}")
    logger.info(f"处理结果数量: {len(processed_results)}")

    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break

    try:
        # 使用 prompt.invoke 调用 LLM，使用流式输出
        chain = direct_answer_prompt | model
        content = ""
        async for chunk in chain.astream({
            "messages": state["messages"],
            "knowledge_context": knowledge_context,
            "tool_results_context": tool_results_context,
            "summary": summary
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content

        # 添加到消息历史
        ai_message = AIMessage(content=content)

        logger.info(f"======直接回答生成: {content}...")

        return {
            "messages": state["messages"] + [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": content, "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"直接回答生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": error_message, "topic": state.get("current_topic", "")}]
        }

async def clarify_question(state: State) -> State:
    """澄清提问节点（含缺参澄清）"""
    processed_results = state.get("processed_results", [])
    messages = state.get("messages", [])
    intent_clarification = str(state.get("clarification_question") or "").strip()

    if (
        state.get("intent_gate_action") == "CLARIFY"
        and state.get("confidence_decision") is None
        and intent_clarification
    ):
        user_message_content = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message_content = msg.content
                break

        ai_message = AIMessage(content=intent_clarification)
        logger.info("IntentGate澄清提问生成: %s", intent_clarification)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{
                "user": user_message_content,
                "assistant": intent_clarification,
                "topic": state.get("current_topic", "")
            }]
        }

    # 收集缺参信息，去重保序
    clarification_messages = list(dict.fromkeys(
        _get_clarify_parameter_message(result)
        for result in processed_results
        if result.get("needs_clarification") and _get_clarify_parameter_message(result)
    ))
    missing_params_context = "\n".join(clarification_messages) if clarification_messages else "无"

    # 提取知識庫相關問題
    if state.get("knowledge_results"):
        top_knowledge = state["knowledge_results"][0][0]
        question_part = top_knowledge.split("\n")[0].replace("问题: ", "")
    else:
        question_part = "无"

    # 提取用戶原始問題
    user_question = messages[-1].content if messages else ""
    conversation_history = _format_conversation_history(messages)
    summary = state.get("summary", "无")

    # 提取用戶消息內容（用於寫入 history）
    user_message_content = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break

    logger.info(
        "进入澄清提问节点，用户问题: %s, 缺参信息: %s, 相关问题: %s, 历史消息数: %s",
        user_question, missing_params_context, question_part, len(messages)
    )

    try:
        chain = clarify_question_prompt | model
        content = ""
        async for chunk in chain.astream({
            "messages": messages,
            "user_question": user_question,
            "conversation_history": conversation_history,
            "summary": summary,
            "related_question": question_part,
            "missing_params_context": missing_params_context,
        }):
            content += chunk.content

        ai_message = AIMessage(content=content)
        logger.info("======澄清提问生成: %s", content)

        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{
                "user": user_message_content,
                "assistant": content,
                "topic": state.get("current_topic", "")
            }]
        }
    except Exception as e:
        logger.error("澄清提问生成失败: %s", str(e))
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{
                "user": user_message_content,
                "assistant": error_message,
                "topic": state.get("current_topic", "")
            }]
        }

async def chat_response(state: State) -> State:
    """闲聊回复节点"""

    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break
    
    summary = state.get("summary", "无")
    logger.info(f"进入闲聊回复节点，用户消息: {user_message_content}")


    try:
        # 使用从配置文件加载的闲聊提示模板
        chain = chat_response_prompt | model
        content = ""
        async for chunk in chain.astream({
            "user_question": user_message_content,
            "summary": summary
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content

        # 添加到消息历史
        ai_message = AIMessage(content=content)

        logger.info(f"======闲聊回复生成: {content}...")

        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{
                "user": user_message_content, 
                "assistant": content, 
                "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"闲聊回复生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{
                "user": user_message_content, 
                "assistant": error_message, 
                "topic": state.get("current_topic", "")}]
        }

async def fallback_response(state: State) -> State:
    """兜底回复节点"""
    # 获取用户问题
    user_question = state["messages"][-1].content if state["messages"] else ""
    summary = state.get("summary", "无")

    logger.info(f"进入兜底回复节点，用户问题: {user_question}")

    # 提取用户消息内容
    user_message_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message_content = msg.content
            break

    try:
        # 使用 prompt.invoke 调用 LLM，使用流式输出
        chain = fallback_response_prompt | model
        content = ""
        async for chunk in chain.astream({
            "user_question": user_question,
            "summary": summary
        }):
            # 从AIMessageChunk中提取content属性
            content += chunk.content

        # 添加到消息历史
        ai_message = AIMessage(content=content)

        logger.info(f"======兜底回复生成: {content}...")

        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": content, "topic": state.get("current_topic", "")}]
        }
    except Exception as e:
        logger.error(f"兜底回复生成失败: {str(e)}")
        # 发生异常时返回兜底回复
        error_message = "抱歉，我在生成回答时遇到了问题，请稍后再试。"
        ai_message = AIMessage(content=error_message)
        return {
            "messages": [ai_message],
            "history": state.get("history", []) + [{"user": user_message_content, "assistant": error_message, "topic": state.get("current_topic", "")}]
        }

async def context_compression_node(state: State) -> State:
    """上下文压缩薄节点，仅负责调用 summary 模块。"""
    logger.info("进入上下文压缩节点")
    return await compress_context(state, summarization_prompt, model)

async def increment_rounds(state: State) -> State:
    """增加对话轮数"""
    current_rounds = state.get("conversation_rounds", 0)
    return {"conversation_rounds": current_rounds + 1}


def _extract_tracking_numbers(text: str) -> list[str]:
    """提取常見物流/訂單號，用於本輪工具結果復用。"""
    if not text:
        return []
    pattern = r"(?<![A-Za-z0-9])(?:\d{5,}|[A-Za-z]{1,4}\d{3,}[A-Za-z0-9]*)(?![A-Za-z0-9])"
    return re.findall(pattern, str(text))


def _needs_logistics_clarification(sub_question: str, intent_id: str) -> bool:
    """物流查詢缺少單號時，提前進入澄清分支，避免誤檢索或編造參數。"""
    if intent_id != "A3":
        return False

    logistics_keywords = ["物流", "快递", "快遞", "包裹", "运到", "運到", "运输", "運輸", "送达", "送達", "到哪"]
    has_logistics_intent = any(keyword in sub_question for keyword in logistics_keywords)
    has_tracking_number = bool(_extract_tracking_numbers(sub_question))

    return has_logistics_intent and not has_tracking_number


def _tool_history_from_processed_results(processed_results: list[dict]) -> list[dict]:
    history = []
    for result in processed_results:
        if not result.get("is_tool"):
            continue

        raw_tool_results = result.get("raw_tool_results") or []
        tool_calls = result.get("tool_calls") or []
        tracking_numbers = set(_extract_tracking_numbers(result.get("sub_question", "")))
        for tool_call in tool_calls:
            args = tool_call.get("args") if isinstance(tool_call, dict) else {}
            tracking_number = (args or {}).get("tracking_number")
            if tracking_number:
                tracking_numbers.add(str(tracking_number))
        for raw_result in raw_tool_results:
            tracking_numbers.update(_extract_tracking_numbers(raw_result))

        history.append({
            "sub_question": result.get("sub_question", ""),
            "tracking_numbers": sorted(tracking_numbers),
            "result": result.get("result", ""),
            "raw_tool_results": raw_tool_results,
            "tool_calls": tool_calls,
        })

    return history


def _build_tool_memory(state: State) -> list[dict]:
    """合併State中的短期工具記憶與本輪已處理結果。"""
    memory = []
    memory.extend(state.get("tool_execution_history", []) or [])
    memory.extend(_tool_history_from_processed_results(state.get("processed_results", []) or []))
    return _dedupe_tool_memory(memory)


def _dedupe_tool_memory(memory: list[dict]) -> list[dict]:
    deduped = []
    seen = set()
    for item in memory:
        tracking_numbers = tuple(item.get("tracking_numbers") or [])
        raw_tool_results = tuple(item.get("raw_tool_results") or [])
        key = (tracking_numbers, raw_tool_results)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _find_reusable_tool_result(sub_question: str, tool_memory: list[dict]) -> dict | None:
    tracking_numbers = set(_extract_tracking_numbers(sub_question))
    if not tracking_numbers:
        return None

    for item in reversed(tool_memory):
        item_numbers = set(item.get("tracking_numbers") or [])
        if tracking_numbers & item_numbers:
            return item

    return None


def _find_single_current_logistics_result(
    sub_question: str,
    intent_id: str,
    current_tool_memory: list[dict],
) -> dict | None:
    """本輪只有一個物流查詢結果時，允許A3殘句復用該結果。"""
    if intent_id != "A3" or _extract_tracking_numbers(sub_question):
        return None

    logistics_keywords = ["物流", "快递", "快遞", "包裹", "多久", "到", "送达", "送達"]
    if not any(keyword in sub_question for keyword in logistics_keywords):
        return None

    current_tracking_numbers = set()
    for item in current_tool_memory:
        current_tracking_numbers.update(item.get("tracking_numbers") or [])

    if len(current_tracking_numbers) != 1 or len(current_tool_memory) != 1:
        return None

    return current_tool_memory[0]


def _tool_call_tracking_number(tool_call: dict) -> str:
    args = tool_call.get("args") if isinstance(tool_call, dict) else {}
    if not isinstance(args, dict):
        return ""
    return str(args.get("tracking_number", "")).strip()


async def _run_stateful_react_tool_chain(
    state: State,
    intent_results: list[tuple[str, str, float]],
) -> dict:
    """以整輪子問題為單位執行有狀態ReAct工具調度。"""
    existing_tool_memory = _build_tool_memory(state)
    payload = {
        "tool_memory": existing_tool_memory,
        "sub_questions": [
            {
                "index": index,
                "sub_question": sub_q,
                "intent_id": intent_id,
                "intent_score": score,
            }
            for index, (sub_q, intent_id, score) in enumerate(intent_results, 1)
        ],
    }

    logger.info(
        "Stateful ReAct tool chain start: sub_questions=%s, memory_items=%s",
        len(intent_results),
        len(existing_tool_memory),
    )
    result = await react_tool_agent.ainvoke({
        "messages": [
            HumanMessage(
                content=(
                    "請根據以下JSON進行本輪工具調度。"
                    "先查看tool_memory，避免重複查詢已存在的單號；"
                    "再針對sub_questions整體規劃需要查詢的工具。\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
                )
            )
        ]
    })
    messages = result.get("messages", []) if isinstance(result, dict) else []

    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]

    tool_calls = []
    for ai_message in ai_messages:
        tool_calls.extend(getattr(ai_message, "tool_calls", []) or [])

    tool_message_by_id = {
        getattr(message, "tool_call_id", ""): str(message.content)
        for message in tool_messages
    }
    executed_tools = []
    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id", "")
        raw_result = tool_message_by_id.get(tool_call_id, "")
        tracking_number = _tool_call_tracking_number(tool_call)
        tracking_numbers = set()
        if tracking_number:
            tracking_numbers.add(tracking_number)
        tracking_numbers.update(_extract_tracking_numbers(raw_result))
        executed_tools.append({
            "tool_call": tool_call,
            "tracking_number": tracking_number,
            "tracking_numbers": sorted(tracking_numbers),
            "raw_result": raw_result,
        })

    final_answer = str(ai_messages[-1].content) if ai_messages else ""
    raw_tool_results = [item["raw_result"] for item in executed_tools if item.get("raw_result")]

    logger.info(
        "Stateful ReAct tool chain done: tool_calls=%s, tool_results=%s",
        tool_calls,
        raw_tool_results,
    )

    return {
        "messages": messages,
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "raw_tool_results": raw_tool_results,
        "executed_tools": executed_tools,
        "tool_memory": existing_tool_memory,
    }

async def task_dispatcher(state: State) -> State:
    """任务分发节点（分拣中心）

    1. 收集本轮所有子问题
    2. 一次性调用有状态 ReAct agent 做整体工具调度
    3. 将工具结果按物流单号回填到对应子问题，重复单号复用结果
    4. 未被工具解决的子问题交给知识库
    """
    # 获取子问题和它们的意图信息
    multi_intent_results = state.get("multi_intent_results", [])
    sub_questions = state.get("sub_questions", [])

    if not multi_intent_results and sub_questions:
        manager = await get_manager()
        multi_intent_results = []
        for sub_q in sub_questions:
            decision = await manager.match_intent_with_gate(sub_q)
            intent_id = decision.get("intent_id")
            score = decision.get("intent_score", 0.0)
            multi_intent_results.append((sub_q, intent_id, score))

    logger.info(f"任务分发：处理 {len(sub_questions)} 个子问题")

    processed_results = []
    existing_tool_memory = _build_tool_memory(state)
    react_batch_result = {
        "executed_tools": [],
        "tool_calls": [],
        "raw_tool_results": [],
        "final_answer": "",
        "tool_memory": existing_tool_memory,
    }

    dispatch_intent_results = [
        item for item in multi_intent_results
        if item[1] not in {"D1", "D2"}
        and not _needs_logistics_clarification(item[0], item[1])
    ]
    if dispatch_intent_results:
        try:
            react_batch_result = await _run_stateful_react_tool_chain(state, dispatch_intent_results)
        except Exception as e:
            logger.error("Stateful ReAct 工具鏈路失敗，全部轉知識庫: %s", e)

    executed_tools = react_batch_result.get("executed_tools", [])
    tool_memory = [*existing_tool_memory]
    current_tool_memory = []
    for executed_tool in executed_tools:
        tool_call = executed_tool.get("tool_call", {})
        raw_result = executed_tool.get("raw_result", "")
        if not raw_result:
            continue
        memory_item = {
            "sub_question": "",
            "tracking_numbers": executed_tool.get("tracking_numbers", []),
            "result": raw_result,
            "raw_tool_results": [raw_result],
            "tool_calls": [tool_call],
        }
        tool_memory.append(memory_item)
        current_tool_memory.append(memory_item)

    for i, (sub_q, intent_id, score) in enumerate(multi_intent_results):
        logger.info(f"处理子问题 {i+1}/{len(multi_intent_results)}: {sub_q} (意图: {intent_id}, 相似度: {score:.4f})")

        if intent_id in {"D1", "D2"}:
            logger.info(f"意图 {intent_id} 为特殊意图，跳过工具判断")
            processed_results.append({
                "sub_question": sub_q,
                "intent_id": intent_id,
                "intent_score": score,
                "is_tool": False
            })
            continue

        if _needs_logistics_clarification(sub_q, intent_id):
            logger.info("物流查询缺少tracking_number，进入澄清: %s", sub_q)
            processed_results.append({
                "sub_question": sub_q,
                "intent_id": intent_id,
                "intent_score": score,
                "is_tool": False,
                "needs_clarification": True,
                "missing_slots": ["tracking_number"],
                "clarify_parameters": "請提供需要查詢的物流單號。",
            })
            continue

        reusable_result = (
            _find_reusable_tool_result(sub_q, tool_memory)
            or _find_single_current_logistics_result(sub_q, intent_id, current_tool_memory)
        )
        if reusable_result:
            logger.info("子问题复用已查询工具结果: %s", sub_q)
            processed_results.append({
                "sub_question": sub_q,
                "intent_id": intent_id,
                "intent_score": score,
                "is_tool": True,
                "result": reusable_result.get("result", ""),
                "tool_calls": reusable_result.get("tool_calls", []),
                "raw_tool_results": reusable_result.get("raw_tool_results", []),
                "reused_tool_result": True,
            })
            continue

        logger.info(f"子問題未觸發工具，需要查詢知識庫: {sub_q}")
        processed_results.append({
            "sub_question": sub_q,
            "intent_id": intent_id,
            "intent_score": score,
            "is_tool": False
        })

    primary_intent = None
    for result in processed_results:
        candidate_intent = result.get("intent_id")
        if candidate_intent not in {"D1", "D2"}:
            primary_intent = candidate_intent
            break
    if primary_intent is None and multi_intent_results:
        primary_intent = multi_intent_results[0][1]

    # 构建返回结果
    return {
        "processed_results": processed_results,
        "tool_execution_history": _dedupe_tool_memory(tool_memory),
        "intent_id": primary_intent,
        "current_topic": primary_intent
    }
