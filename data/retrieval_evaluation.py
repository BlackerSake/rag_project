"""Retrieval evaluation utilities used by the main customer service system."""

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
    """Weights for aggregating retrieval quality metrics."""

    recall_at_k: float = 0.30
    precision_at_k: float = 0.20
    mrr: float = 0.25
    ndcg_at_k: float = 0.25


@dataclass(frozen=True)
class RetrievalEvaluationConfig:
    """Runtime configuration for retrieval evaluation."""

    top_k: int = 5
    weights: RetrievalWeights = field(default_factory=RetrievalWeights)


class RetrievalEvaluator:
    """Calculate Recall@K, Precision@K, MRR and NDCG@K for retrieved documents."""

    def __init__(self, config: RetrievalEvaluationConfig | None = None) -> None:
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
        """Evaluate a retrieval result list and return a JSON-serializable event."""

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
        """Calculate ranking metrics from per-rank relevance flags."""

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
        """Aggregate metrics when every required metric is available."""

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
        """Write one retrieval evaluation event to the application log."""

        logger.info("RetrievalEvaluation: %s", json.dumps(event, ensure_ascii=False))
        if event.get("metric_status") == "intent_labeled":
            logger.info("检索评测使用意图弱标注，未计算 Recall@K；完整召回率需要人工 relevant_docs 或全库相关集")
        elif event.get("metric_status") == "unlabeled":
            logger.info("检索评测缺少相关文档标注，仅记录检索事件与耗时")

    def _extract_items(self, results: list[Any]) -> list[dict[str, Any]]:
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
        expected = str(expected_intent_id).strip()
        if not expected:
            return False
        return str(metadata.get("intent_id") or metadata.get("intent") or "").strip() == expected

    def _extract_doc_id(self, doc: Any, metadata: dict[str, Any], content: str) -> str:
        doc_id = metadata.get("doc_id") or metadata.get("id") or metadata.get("mysql_id")
        if doc_id is None and isinstance(doc, dict):
            doc_id = doc.get("doc_id") or doc.get("id") or doc.get("mysql_id")
        return str(doc_id) if doc_id is not None else hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def _json_safe_mapping(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {str(key): self._json_safe_score(value) for key, value in metadata.items()}

    def _json_safe_score(self, value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except TypeError:
            return str(value)

    def _preview(self, content: str, max_length: int = 120) -> str:
        preview = " ".join(content.split())
        if len(preview) > max_length:
            return preview[:max_length] + "..."
        return preview

    def _empty_metrics(self, *, use_none: bool = False) -> dict[str, float | None]:
        empty_value = None if use_none else 0.0
        return {
            "recall_at_k": empty_value,
            "precision_at_k": empty_value,
            "mrr": empty_value,
            "ndcg_at_k": empty_value,
        }

    def _safe_divide(self, numerator: float, denominator: float) -> float:
        if denominator == 0:
            return 0.0
        return numerator / denominator
