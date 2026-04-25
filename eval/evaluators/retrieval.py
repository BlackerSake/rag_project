"""Retrieval quality evaluator for RAG systems."""

from __future__ import annotations

import logging
import math
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import maybe_await, mean, safe_divide

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate Recall@K, Precision@K, MRR and NDCG@K."""

    def __init__(self, knowledge_base: Any, config: EvalConfig | None = None) -> None:
        """Initialize the retrieval evaluator.

        Args:
            knowledge_base: Object exposing ``search(query, k=...)``.
            config: Optional evaluation configuration.

        Returns:
            None.

        Raises:
            ValueError: If ``knowledge_base`` does not expose ``search``.
        """
        if not hasattr(knowledge_base, "search"):
            raise ValueError("knowledge_base must expose a search(query, k=...) method")
        self.knowledge_base = knowledge_base
        self.config = config or EvalConfig()

    async def evaluate_case(self, query: str, relevant_docs: list[str] | set[str], k: int | None = None) -> dict[str, Any]:
        """Evaluate retrieval metrics for a single query.

        Args:
            query: User query.
            relevant_docs: Relevant document IDs for the query.
            k: Optional top-k override.

        Returns:
            Dictionary containing retrieved IDs and per-case metrics.

        Raises:
            No exceptions are raised; search failures return zero metrics.
        """
        top_k = k or self.config.top_k
        relevant_set = {str(doc_id) for doc_id in (relevant_docs or []) if str(doc_id)}
        if not query:
            logger.warning("Empty query received by RetrievalEvaluator")
        if not relevant_set:
            logger.warning("No relevant docs supplied for query: %s", query)

        try:
            logger.info("Running retrieval evaluation for query=%s top_k=%s", query, top_k)
            results = await maybe_await(self.knowledge_base.search(query, k=top_k))
        except Exception as exc:
            logger.error("Knowledge base search failed for query=%s: %s", query, exc)
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
        """Evaluate retrieval metrics for multiple queries.

        Args:
            cases: Items with ``query`` and ``relevant_docs`` fields.
            k: Optional top-k override.

        Returns:
            Summary with averaged metrics, score and case-level details.

        Raises:
            No exceptions are raised; invalid cases are evaluated with defaults.
        """
        if not cases:
            logger.warning("Empty retrieval case list")
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
        logger.info("Retrieval evaluation complete: %.4f", summary["retrieval_score"])
        return summary

    def calculate_metrics(self, retrieved_docs: list[str], relevant_docs: set[str], k: int) -> dict[str, float]:
        """Calculate retrieval metrics for one ranked result list.

        Args:
            retrieved_docs: Ranked retrieved document IDs.
            relevant_docs: Set of relevant document IDs.
            k: Top-k denominator used by Precision@K and NDCG@K.

        Returns:
            Metric dictionary.

        Raises:
            No exceptions are raised.
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
        """Aggregate retrieval metrics according to configured weights.

        Args:
            metrics: Dictionary with retrieval metric values.

        Returns:
            Weighted retrieval score.

        Raises:
            No exceptions are raised.
        """
        weights = self.config.retrieval_weights
        return (
            weights.recall_at_k * metrics.get("recall_at_k", 0.0)
            + weights.precision_at_k * metrics.get("precision_at_k", 0.0)
            + weights.mrr * metrics.get("mrr", 0.0)
            + weights.ndcg_at_k * metrics.get("ndcg_at_k", 0.0)
        )

    def _extract_doc_ids(self, results: Any) -> list[str]:
        """Extract document IDs from LangChain-style retrieval results.

        Args:
            results: Iterable of ``(doc, score)`` tuples or document objects.

        Returns:
            List of document IDs.

        Raises:
            No exceptions are raised; malformed items are skipped.
        """
        doc_ids: list[str] = []
        for item in results or []:
            doc = item[0] if isinstance(item, tuple) and item else item
            metadata = getattr(doc, "metadata", {}) or {}
            doc_id = metadata.get("doc_id") or metadata.get("id")
            if doc_id is None and isinstance(doc, dict):
                doc_id = doc.get("doc_id") or doc.get("id")
            if doc_id is None:
                logger.warning("Retrieved document missing doc_id metadata: %r", doc)
                continue
            doc_ids.append(str(doc_id))
        return doc_ids

    def _empty_summary(self) -> dict[str, Any]:
        """Return an empty retrieval summary.

        Args:
            None.

        Returns:
            Zero-valued summary.

        Raises:
            No exceptions are raised.
        """
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "retrieval_score": 0.0,
            "details": [],
        }
