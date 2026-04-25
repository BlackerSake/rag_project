"""RAG 系统检索质量评测器。"""

from __future__ import annotations

import logging
import math
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import maybe_await, mean, safe_divide

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """评估 Recall@K、Precision@K、MRR 和 NDCG@K。"""

    def __init__(self, knowledge_base: Any, config: EvalConfig | None = None) -> None:
        """初始化检索评测器。

        参数:
            knowledge_base: 暴露 ``search(query, k=...)`` 方法的对象。
            config: 可选评测配置。

        返回:
            无。

        异常:
            ValueError: 当 ``knowledge_base`` 未暴露 ``search`` 方法时抛出。
        """
        if not hasattr(knowledge_base, "search"):
            raise ValueError("knowledge_base 必须暴露 search(query, k=...) 方法")
        self.knowledge_base = knowledge_base
        self.config = config or EvalConfig()

    async def evaluate_case(self, query: str, relevant_docs: list[str] | set[str], k: int | None = None) -> dict[str, Any]:
        """评估单条查询的检索指标。

        参数:
            query: 用户查询。
            relevant_docs: 该查询对应的相关文档 ID。
            k: 可选 Top-K 覆盖值。

        返回:
            包含检索文档 ID 和单条用例指标的字典。

        异常:
            不主动抛出异常；检索失败时返回零值指标。
        """
        top_k = k or self.config.top_k
        relevant_set = {str(doc_id) for doc_id in (relevant_docs or []) if str(doc_id)}
        if not query:
            logger.warning("RetrievalEvaluator 收到空查询")
        if not relevant_set:
            logger.warning("查询未提供相关文档: %s", query)

        try:
            logger.info("执行检索评测，query=%s top_k=%s", query, top_k)
            results = await maybe_await(self.knowledge_base.search(query, k=top_k))
        except Exception as exc:
            logger.error("知识库检索失败，query=%s: %s", query, exc)
            results = []

        retrieved_ids = self._extract_doc_ids(results)[:top_k]
        metrics = self.calculate_metrics(retrieved_ids, relevant_set, top_k)
        return {
            "query": query,
            "relevant_docs": sorted(relevant_set),
            "retrieved_docs": retrieved_ids,
            **metrics,
        }

    async def evaluate_batch(self, cases: list[dict[str, Any]], k: int | None = None) -> dict[str, Any]:
        """评估多条查询的检索指标。

        参数:
            cases: 包含 ``query`` 和 ``relevant_docs`` 字段的用例列表。
            k: 可选 Top-K 覆盖值。

        返回:
            包含平均指标、总分和用例详情的汇总结果。

        异常:
            不主动抛出异常；非法用例会按默认值评估。
        """
        if not cases:
            logger.warning("检索评测用例列表为空")
            return self._empty_summary()

        details = [
            await self.evaluate_case(
                case.get("query", ""),
                case.get("relevant_docs", []),
                k=k,
            )
            for case in cases
        ]

        summary = {
            "recall_at_k": mean([item["recall_at_k"] for item in details]),
            "precision_at_k": mean([item["precision_at_k"] for item in details]),
            "mrr": mean([item["mrr"] for item in details]),
            "ndcg_at_k": mean([item["ndcg_at_k"] for item in details]),
            "details": details,
        }
        summary["retrieval_score"] = self.calculate_score(summary)
        logger.info("检索评测完成: %.4f", summary["retrieval_score"])
        return summary

    def calculate_metrics(self, retrieved_docs: list[str], relevant_docs: set[str], k: int) -> dict[str, float]:
        """计算单个排序检索结果列表的指标。

        参数:
            retrieved_docs: 按排名排序的检索文档 ID。
            relevant_docs: 相关文档 ID 集合。
            k: Precision@K 和 NDCG@K 使用的 Top-K 分母。

        返回:
            指标字典。

        异常:
            不主动抛出异常。
        """
        top_k = max(1, k)
        hits = [doc_id for doc_id in retrieved_docs[:top_k] if doc_id in relevant_docs]
        hit_count = len(hits)

        recall = safe_divide(hit_count, len(relevant_docs))
        precision = safe_divide(hit_count, top_k)
        mrr = 0.0
        for rank, doc_id in enumerate(retrieved_docs[:top_k], start=1):
            if doc_id in relevant_docs:
                mrr = 1.0 / rank
                break

        dcg = sum(
            (1.0 if doc_id in relevant_docs else 0.0) / math.log2(index + 1)
            for index, doc_id in enumerate(retrieved_docs[:top_k], start=1)
        )
        ideal_relevant_count = min(len(relevant_docs), top_k)
        idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_relevant_count + 1))
        ndcg = safe_divide(dcg, idcg)

        return {
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "ndcg_at_k": ndcg,
        }

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """按配置权重聚合检索指标。

        参数:
            metrics: 包含检索指标值的字典。

        返回:
            加权检索质量得分。

        异常:
            不主动抛出异常。
        """
        weights = self.config.retrieval_weights
        return (
            weights.recall_at_k * metrics.get("recall_at_k", 0.0)
            + weights.precision_at_k * metrics.get("precision_at_k", 0.0)
            + weights.mrr * metrics.get("mrr", 0.0)
            + weights.ndcg_at_k * metrics.get("ndcg_at_k", 0.0)
        )

    def _extract_doc_ids(self, results: Any) -> list[str]:
        """从 LangChain 风格检索结果中提取文档 ID。

        参数:
            results: ``(doc, score)`` 元组或文档对象组成的可迭代结果。

        返回:
            文档 ID 列表。

        异常:
            不主动抛出异常；格式异常的条目会被跳过。
        """
        doc_ids: list[str] = []
        for item in results or []:
            doc = item[0] if isinstance(item, tuple) and item else item
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = metadata.get("doc_id") or metadata.get("id")
            if doc_id is None and isinstance(doc, dict):
                doc_id = doc.get("doc_id") or doc.get("id")
            if doc_id is None:
                logger.warning("检索文档缺少 doc_id 元数据: %r", doc)
                continue
            doc_ids.append(str(doc_id))
        return doc_ids

    def _empty_summary(self) -> dict[str, Any]:
        """返回空检索评测汇总。

        参数:
            无。

        返回:
            全零汇总结果。

        异常:
            不主动抛出异常。
        """
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "retrieval_score": 0.0,
            "details": [],
        }
