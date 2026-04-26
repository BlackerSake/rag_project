"""客户服务系统使用的检索评估工具。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalWeights:
    """用于聚合检索质量指标的权重。"""

    recall_at_k: float = 0.30
    precision_at_k: float = 0.20
    mrr: float = 0.25
    ndcg_at_k: float = 0.25


@dataclass(frozen=True)
class RetrievalEvaluationConfig:
    """检索评估的运行时配置。"""

    top_k: int = 5
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)


class RetrievalEvaluator:
    """计算检索文档的 Recall@K、Precision@K、MRR 和 NDCG@K。"""

    def __init__(self, config: RetrievalEvaluationConfig | None = None) -> None:
        """初始化检索评估器。

        参数:
            config: 检索评估配置，如为 None 则使用默认配置。
        """
        self.config = config or RetrievalEvaluationConfig()

    def evaluate_results(
        self,
        query: str,
        results: list[Any],
        *,
        k: int | None = None,
        relevant_docs: list[str] | set[str] | tuple[str, ...] | None = None,
        expected_intent_id: str | None = None,
        response_time: float = 0.0,
        from_cache: bool = False,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """评估检索结果列表并返回可JSON序列化的事件。

        参数:
            query: 用户查询。
            results: 检索结果列表。
            k: 可选的 top-k 覆盖值。
            relevant_docs: 相关文档ID列表或集合。
            expected_intent_id: 预期的意图ID。
            response_time: 响应时间。
            from_cache: 是否来自缓存。
            context: 上下文信息。

        返回:
            包含评估结果的字典。
        """

        top_k = max(1, int(k or self.config.top_k))
        items = self._extract_items(results)[:top_k]
        labels = {str(label).strip() for label in (relevant_docs or []) if str(label).strip()}

        if labels:
            metric_status = "labeled"
            relevance_flags, matched_labels = self._match_relevant_labels(items, labels)
            metrics = self.calculate_metrics(relevance_flags, len(labels), top_k, len(matched_labels))
        elif expected_intent_id:
            metric_status = "intent_labeled"
            relevance_flags = [self._matches_intent(item["metadata"], expected_intent_id) for item in items]
            metrics = self.calculate_metrics(relevance_flags, None, top_k, None)
        else:
            metric_status = "unlabeled"
            relevance_flags = [False for _ in items]
            metrics = self._empty_metrics(use_none=True)

        event = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "top_k": top_k,
            "metric_status": metric_status,
            "metrics": metrics,
            "retrieval_score": self.calculate_score(metrics),
            "response_time": response_time,
            "from_cache": from_cache,
            "relevant_docs": sorted(labels),
            "expected_intent_id": expected_intent_id,
            "results": [
                {
                    "rank": index + 1,
                    "doc_id": item["doc_id"],
                    "score": item["score"],
                    "is_relevant": relevance_flags[index],
                    "metadata": item["metadata"],
                    "content_preview": item["content_preview"],
                }
                for index, item in enumerate(items)
            ],
        }
        if context:
            event["context"] = context
        return event

    def calculate_metrics(
        self,
        relevance_flags: list[bool],
        relevant_total: int | None,
        k: int,
        matched_relevant_total: int | None = None,
    ) -> dict[str, float | None]:
        """根据每个排名的相关性标志计算排序指标。

        参数:
            relevance_flags: 每个排名的相关性标志列表。
            relevant_total: 相关文档总数。
            k: top-k 值。
            matched_relevant_total: 匹配的相关文档总数。

        返回:
            包含 Recall@K、Precision@K、MRR 和 NDCG@K 的字典。
        """

        top_k = max(1, k)
        top_flags = relevance_flags[:top_k]
        hit_count = sum(1 for is_relevant in top_flags if is_relevant)
        matched_total = hit_count if matched_relevant_total is None else matched_relevant_total

        precision = hit_count / top_k
        recall = None if relevant_total is None else self._safe_divide(matched_total, relevant_total)

        mrr = 0.0
        for rank, is_relevant in enumerate(top_flags, start=1):
            if is_relevant:
                mrr = 1.0 / rank
                break

        dcg = sum(
            (1.0 if is_relevant else 0.0) / math.log2(index + 1)
            for index, is_relevant in enumerate(top_flags, start=1)
        )
        if relevant_total is None:
            ideal_relevant_count = min(hit_count, top_k)
        else:
            ideal_relevant_count = min(relevant_total, top_k)
        idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_relevant_count + 1))
        ndcg = self._safe_divide(dcg, idcg)

        return {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
        }

    def calculate_score(self, metrics: dict[str, float | None]) -> float | None:
        """当所有必需指标可用时聚合指标。

        参数:
            metrics: 包含评估指标的字典。

        返回:
            聚合后的检索分数，若缺少指标则返回 None。
        """

        required = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")
        if any(metrics.get(name) is None for name in required):
            return None

        weights = self.config.weights
        return (
            weights.recall_at_k * float(metrics.get("recall_at_k", 0.0))
            + weights.precision_at_k * float(metrics.get("precision_at_k", 0.0))
            + weights.mrr * float(metrics.get("mrr", 0.0))
            + weights.ndcg_at_k * float(metrics.get("ndcg_at_k", 0.0))
        )

    def log_event(self, event: dict[str, Any]) -> None:
        """将一个检索评估事件写入应用日志。

        参数:
            event: 评估事件字典。
        """

        logger.info("RetrievalEvaluation: %s", json.dumps(event, ensure_ascii=False))
        if event.get("metric_status") == "intent_labeled":
            logger.info("检索评测使用意图弱标注，未计算 Recall@K；完整召回率需要人工 relevant_docs 或全库相关集")
        elif event.get("metric_status") == "unlabeled":
            logger.info("检索评测缺少相关文档标注，仅记录检索事件与耗时")

    def _extract_items(self, results: list[Any]) -> list[dict[str, Any]]:
        """从检索结果中提取项目列表。

        参数:
            results: 检索结果列表。

        返回:
            提取的项目列表。
        """
        items: list[dict[str, Any]] = []
        for result in results or []:
            doc = result[0] if isinstance(result, tuple) and result else result
            score = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            metadata = dict(getattr(doc, "metadata", {}) or {})
            content = str(getattr(doc, "page_content", ""))
            if isinstance(doc, dict):
                metadata = dict(doc.get("metadata", {}) or metadata)
                content = str(doc.get("page_content", doc.get("content", content)))

            doc_id = self._extract_doc_id(doc, metadata, content)
            items.append(
                {
                    "doc_id": doc_id,
                    "score": self._json_safe_score(score),
                    "metadata": self._json_safe_mapping(metadata),
                    "content": content,
                    "content_preview": self._preview(content),
                }
            )
        return items

    def _match_relevant_labels(self, items: list[dict[str, Any]], labels: set[str]) -> tuple[list[bool], set[str]]:
        """匹配相关标签并返回相关性标志和匹配的标签。

        参数:
            items: 项目列表。
            labels: 相关标签集合。

        返回:
            相关性标志列表和匹配的标签集合。
        """
        matched_labels: set[str] = set()
        flags: list[bool] = []
        for item in items:
            item_labels = {
                str(item.get("doc_id", "")),
                str(item["metadata"].get("doc_id", "")),
                str(item["metadata"].get("id", "")),
                str(item["metadata"].get("mysql_id", "")),
                str(item["metadata"].get("original_question", "")),
            }
            direct_matches = labels.intersection({label for label in item_labels if label})
            content_matches = {label for label in labels if label and label in item.get("content", "")}
            current_matches = direct_matches.union(content_matches)
            matched_labels.update(current_matches)
            flags.append(bool(current_matches))
        return flags, matched_labels

    def _matches_intent(self, metadata: dict[str, Any], expected_intent_id: str) -> bool:
        """检查元数据是否匹配预期的意图ID。

        参数:
            metadata: 文档元数据。
            expected_intent_id: 预期的意图ID。

        返回:
            是否匹配。
        """
        expected = str(expected_intent_id).strip()
        if not expected:
            return False
        return str(metadata.get("intent_id") or metadata.get("intent") or "").strip() == expected

    def _extract_doc_id(self, doc: Any, metadata: dict[str, Any], content: str) -> str:
        """从文档、元数据或内容中提取文档ID。

        参数:
            doc: 文档对象。
            metadata: 文档元数据。
            content: 文档内容。

        返回:
            文档ID。
        """
        doc_id = metadata.get("doc_id") or metadata.get("id") or metadata.get("mysql_id")
        if doc_id is None and isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("id") or doc.get("mysql_id")
        return str(doc_id) if doc_id is not None else hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def _json_safe_mapping(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """将元数据转换为JSON安全的映射。

        参数:
            metadata: 元数据字典。

        返回:
            JSON安全的元数据映射。
        """
        return {str(key): self._json_safe_score(value) for key, value in metadata.items()}

    def _json_safe_score(self, value: Any) -> Any:
        """确保分数值可JSON序列化。

        参数:
            value: 要检查的值。

        返回:
            JSON安全的值。
        """
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)

    def _preview(self, content: str, max_length: int = 120) -> str:
        """生成内容的预览。

        参数:
            content: 原始内容。
            max_length: 预览最大长度。

        返回:
            内容预览。
        """
        preview = " ".join(content.split())
        if len(preview) > max_length:
            return preview[:max_length] + "..."
        return preview

    def _empty_metrics(self, *, use_none: bool = False) -> dict[str, float | None]:
        """返回空的指标字典。

        参数:
            use_none: 是否使用 None 作为空值。

        返回:
            空的指标字典。
        """
        empty_value = None if use_none else 0.0
        return {
            "recall_at_k": empty_value,
            "precision_at_k": empty_value,
            "mrr": empty_value,
            "ndcg_at_k": empty_value,
        }

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        """安全的除法运算，避免除零错误。

        参数:
            numerator: 分子。
            denominator: 分母。

        返回:
            除法结果，分母为零时返回0.0。
        """
        if denominator == 0:
            return 0.0
        return numerator / denominator
