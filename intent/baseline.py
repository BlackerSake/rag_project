"""意图门控阈值加载与离线校准。"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "config" / "intent_gate_baseline.yaml"


@dataclass(frozen=True)
class IntentBaseline:
    """意图门控使用的分数与区分度阈值。"""

    score_high: float = 0.5
    score_low: float = 0.3
    margin_high: float = 0.1
    margin_low: float = 0.05
    min_intent_samples: int = 20
    confusion_groups: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。"""
        payload = asdict(self)
        payload["confusion_groups"] = [list(group) for group in self.confusion_groups]
        return payload


def _safe_float(value: Any, default: float) -> float:
    """安全读取浮点数，非法值回退默认值。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int) -> int:
    """安全读取正整数，非法值回退默认值。"""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _normalize_confusion_groups(raw_groups: Any) -> tuple[tuple[str, ...], ...]:
    """标准化混淆组配置，过滤空值与非字符串成员。"""
    if not isinstance(raw_groups, list):
        return ()

    groups: list[tuple[str, ...]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, (list, tuple, set)):
            continue
        members = tuple(str(item).strip() for item in raw_group if str(item).strip())
        if len(members) >= 2:
            groups.append(members)
    return tuple(groups)


def _read_baseline_payload(path: Path) -> dict[str, Any]:
    """读取 YAML 或 JSON 配置，文件不存在时返回空配置。"""
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            payload = json.load(f)
        else:
            payload = yaml.safe_load(f)

    return payload if isinstance(payload, dict) else {}


def load_intent_baseline(filepath: str | os.PathLike | None = None) -> IntentBaseline:
    """加载意图门控 baseline，支持配置文件与环境变量覆盖。"""
    raw_path = filepath or os.getenv("INTENT_GATE_BASELINE_PATH")
    path = Path(raw_path) if raw_path else DEFAULT_BASELINE_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    baseline = IntentBaseline()
    payload = _read_baseline_payload(path)
    if payload:
        baseline = IntentBaseline(
            score_high=_safe_float(payload.get("score_high"), baseline.score_high),
            score_low=_safe_float(payload.get("score_low"), baseline.score_low),
            margin_high=_safe_float(payload.get("margin_high"), baseline.margin_high),
            margin_low=_safe_float(payload.get("margin_low"), baseline.margin_low),
            min_intent_samples=_safe_int(payload.get("min_intent_samples"), baseline.min_intent_samples),
            confusion_groups=_normalize_confusion_groups(payload.get("confusion_groups")),
        )

    baseline = replace(
        baseline,
        score_high=_safe_float(os.getenv("INTENT_GATE_SCORE_HIGH"), baseline.score_high),
        score_low=_safe_float(os.getenv("INTENT_GATE_SCORE_LOW"), baseline.score_low),
        margin_high=_safe_float(os.getenv("INTENT_GATE_MARGIN_HIGH"), baseline.margin_high),
        margin_low=_safe_float(os.getenv("INTENT_GATE_MARGIN_LOW"), baseline.margin_low),
        min_intent_samples=_safe_int(os.getenv("INTENT_GATE_MIN_INTENT_SAMPLES"), baseline.min_intent_samples),
    )

    return baseline


def _percentile(values: list[float], percentile: float, default: float) -> float:
    """计算近似百分位，样本不足时返回默认值。"""
    clean_values = sorted(value for value in values if math.isfinite(value))
    if not clean_values:
        return default
    if len(clean_values) == 1:
        return clean_values[0]

    if percentile <= 0:
        return clean_values[0]
    if percentile >= 100:
        return clean_values[-1]

    position = (len(clean_values) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return clean_values[int(position)]
    lower_weight = upper - position
    upper_weight = position - lower
    return clean_values[lower] * lower_weight + clean_values[upper] * upper_weight


def _extract_eval_records(payload: Any) -> list[dict[str, Any]]:
    """从常见评测输出格式中提取样本列表。"""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("records", "samples", "data", "results"):
        records = payload.get(key)
        if isinstance(records, list):
            return [item for item in records if isinstance(item, dict)]
    return []


def _extract_scores(record: dict[str, Any]) -> tuple[float | None, float | None]:
    """从单条评测记录中提取 Top1 分数与 margin。"""
    top1_score = record.get("top1_score", record.get("score"))
    margin = record.get("margin", record.get("intent_margin"))

    candidates = record.get("candidates") or record.get("intent_candidates") or []
    if isinstance(candidates, list) and candidates:
        first = candidates[0] if isinstance(candidates[0], dict) else {}
        second = candidates[1] if len(candidates) > 1 and isinstance(candidates[1], dict) else {}
        top1_score = first.get("score", top1_score)
        if margin is None and second:
            margin = _safe_float(first.get("score"), 0.0) - _safe_float(second.get("score"), 0.0)

    score_value = _safe_float(top1_score, math.nan)
    margin_value = _safe_float(margin, math.nan)
    return (
        score_value if math.isfinite(score_value) else None,
        max(margin_value, 0.0) if math.isfinite(margin_value) else None,
    )


def build_intent_baseline_from_eval(eval_file: str | os.PathLike) -> dict[str, Any]:
    """根据离线评测样本生成初始 baseline 配置。"""
    path = Path(eval_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    records = _extract_eval_records(payload)
    scores: list[float] = []
    margins: list[float] = []
    errors: list[dict[str, Any]] = []
    confusion_pairs: dict[tuple[str, str], int] = {}

    for record in records:
        score, margin = _extract_scores(record)
        if score is not None:
            scores.append(score)
        if margin is not None:
            margins.append(margin)

        expected = str(record.get("expected_intent_id") or record.get("expected_intent") or record.get("intent") or "").strip()
        predicted = str(record.get("predicted_intent_id") or record.get("intent_id") or record.get("top1_intent_id") or "").strip()
        if expected and predicted and expected != predicted:
            errors.append(record)
            pair = tuple(sorted((expected, predicted)))
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    fallback = IntentBaseline()
    baseline = {
        "score_high": _percentile(scores, 75, fallback.score_high),
        "score_low": _percentile(scores, 25, fallback.score_low),
        "margin_high": _percentile(margins, 75, fallback.margin_high),
        "margin_low": _percentile(margins, 25, fallback.margin_low),
        "min_intent_samples": fallback.min_intent_samples,
        "confusion_groups": [
            list(pair)
            for pair, count in sorted(confusion_pairs.items(), key=lambda item: item[1], reverse=True)
            if count >= 2
        ],
        "stats": {
            "sample_count": len(records),
            "score_count": len(scores),
            "margin_count": len(margins),
            "error_count": len(errors),
        },
    }
    return baseline


def _iter_history_records(history: Any) -> Iterable[dict[str, Any]]:
    """兼容不同历史容器，统一迭代字典记录。"""
    records = getattr(history, "records", history)
    if isinstance(records, dict):
        records = records.values()
    if not isinstance(records, Iterable):
        return []
    return (record for record in records if isinstance(record, dict))


def get_dynamic_intent_baseline(
    intent_id: str | None = None,
    baseline: IntentBaseline | None = None,
    history: Any = None,
) -> IntentBaseline:
    """根据滑动窗口生成动态阈值，样本不足时回退全局 baseline。"""
    base = baseline or load_intent_baseline()
    if history is None:
        return base

    records = [
        record for record in _iter_history_records(history)
        if not intent_id or record.get("intent_id") == intent_id
    ]
    if len(records) < base.min_intent_samples:
        return base

    scores = [_safe_float(record.get("score"), math.nan) for record in records]
    margins = [_safe_float(record.get("margin"), math.nan) for record in records]
    scores = [score for score in scores if math.isfinite(score)]
    margins = [margin for margin in margins if math.isfinite(margin)]
    if not scores or not margins:
        return base

    return replace(
        base,
        score_high=_percentile(scores, 75, base.score_high),
        score_low=_percentile(scores, 25, base.score_low),
        margin_high=_percentile(margins, 75, base.margin_high),
        margin_low=_percentile(margins, 25, base.margin_low),
    )


def rebuild_intent_baseline(eval_file: str | os.PathLike, output_file: str | os.PathLike | None = None) -> dict[str, Any]:
    """重新生成 baseline，并在指定输出路径写入 YAML。"""
    baseline = build_intent_baseline_from_eval(eval_file)
    if output_file:
        path = Path(output_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(baseline, f, allow_unicode=True, sort_keys=False)
    return baseline
