"""Fallback capability evaluator for unanswerable RAG cases."""

from __future__ import annotations

import logging
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import LLMClient, call_llm_json, clamp_score, maybe_await, mean

logger = logging.getLogger(__name__)


class FallbackEvaluator:
    """Evaluate reject accuracy, hallucination rate and safe reply rate."""

    def __init__(self, rag_system: Any, llm_client: LLMClient, config: EvalConfig | None = None) -> None:
        """Initialize the fallback evaluator.

        Args:
            rag_system: Object exposing async or sync ``answer(question)``.
            llm_client: Async or sync callable accepting a prompt and returning text.
            config: Optional evaluation configuration.

        Returns:
            None.

        Raises:
            ValueError: If required callables are missing.
        """
        if not hasattr(rag_system, "answer") or not callable(rag_system.answer):
            raise ValueError("rag_system must expose an answer(question) method")
        if not callable(llm_client):
            raise ValueError("llm_client must be callable")
        self.rag_system = rag_system
        self.llm_client = llm_client
        self.config = config or EvalConfig()

    async def evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """Evaluate fallback behavior for one unanswerable question.

        Args:
            case: Item with ``question``, optional ``retrieved_context`` and
                ``expected_behavior`` fields.

        Returns:
            Per-case fallback metrics and LLM judgments.

        Raises:
            No exceptions are raised; RAG/LLM failures are downgraded.
        """
        question = case.get("question") or case.get("unanswerable_question", "")
        expected_behavior = case.get("expected_behavior", "reject")
        retrieved_context = case.get("retrieved_context", [])
        answer = await self._answer_question(question)

        scope_result = await self._judge_scope(question, retrieved_context)
        hallucination_result = await self._judge_hallucination(retrieved_context, answer)

        in_scope = bool(scope_result.get("in_scope", False))
        should_reject = expected_behavior == "reject" or not in_scope
        is_rejection = self._is_safe_reply(answer)
        reject_correct = is_rejection if should_reject else not is_rejection
        has_hallucination = bool(hallucination_result.get("has_hallucination", False))

        logger.info(
            "Fallback case evaluated: reject_correct=%s has_hallucination=%s is_safe=%s",
            reject_correct,
            has_hallucination,
            is_rejection,
        )
        return {
            "question": question,
            "answer": answer,
            "expected_behavior": expected_behavior,
            "in_scope": in_scope,
            "reject_correct": bool(reject_correct),
            "has_hallucination": has_hallucination,
            "is_safe_reply": bool(is_rejection),
            "attempted_answer": not is_rejection,
            "scope_reasoning": scope_result.get("reasoning", ""),
            "hallucinated_parts": hallucination_result.get("hallucinated_parts", []),
            "faithfulness_score": clamp_score(hallucination_result.get("faithfulness_score", 0.0)),
        }

    async def evaluate_batch(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate fallback capability for multiple unanswerable cases.

        Args:
            cases: Unanswerable test cases.

        Returns:
            Reject accuracy, hallucination rate, safe reply rate and score.

        Raises:
            No exceptions are raised.
        """
        if not cases:
            logger.warning("Empty fallback case list")
            return self._empty_summary()

        details = [await self.evaluate_case(case) for case in cases]
        rejected_cases = [item for item in details if item["is_safe_reply"]]
        attempted_cases = [item for item in details if item["attempted_answer"]]

        reject_accuracy = mean([1.0 if item["reject_correct"] else 0.0 for item in details])
        hallucination_rate = (
            mean([1.0 if item["has_hallucination"] else 0.0 for item in attempted_cases])
            if attempted_cases
            else 0.0
        )
        safe_reply_rate = (
            mean([1.0 if item["is_safe_reply"] else 0.0 for item in rejected_cases])
            if rejected_cases
            else 0.0
        )

        summary = {
            "reject_accuracy": reject_accuracy,
            "hallucination_rate": hallucination_rate,
            "safe_reply_rate": safe_reply_rate,
            "details": details,
        }
        summary["fallback_score"] = self.calculate_score(summary)
        logger.info("Fallback batch evaluated: %.4f", summary["fallback_score"])
        return summary

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """Aggregate fallback metrics according to configured weights.

        Args:
            metrics: Dictionary with fallback metric values.

        Returns:
            Weighted fallback score.

        Raises:
            No exceptions are raised.
        """
        weights = self.config.fallback_weights
        return (
            weights.reject_accuracy * metrics.get("reject_accuracy", 0.0)
            + weights.non_hallucination_rate * (1.0 - metrics.get("hallucination_rate", 0.0))
            + weights.safe_reply_rate * metrics.get("safe_reply_rate", 0.0)
        )

    async def _answer_question(self, question: str) -> str:
        """Call the injected RAG system.

        Args:
            question: User question.

        Returns:
            RAG answer text or a safe error fallback.

        Raises:
            No exceptions are raised.
        """
        try:
            logger.info("Calling RAG system for fallback question")
            answer = await maybe_await(self.rag_system.answer(question))
            return str(answer or "")
        except Exception as exc:
            logger.error("RAG answer call failed: %s", exc)
            return "抱歉，当前无法回答该问题。"

    async def _judge_scope(self, question: str, retrieved_context: list[str] | str) -> dict[str, Any]:
        """Judge whether a question is in knowledge-base scope.

        Args:
            question: User question.
            retrieved_context: Retrieved context snippets.

        Returns:
            JSON judgment with ``in_scope`` and ``reasoning``.

        Raises:
            No exceptions are raised.
        """
        context = self._join_context(retrieved_context)
        prompt = f"""判断以下问题是否在知识库的覆盖范围内。
如果问题涉及知识库中没有的信息（如最新政策、实时数据、个人信息等），应该判定为超出范围。

问题: {question}
知识库内容片段: {context}

输出格式: {{"in_scope": true, "reasoning": "..."}}"""
        return await call_llm_json(
            self.llm_client,
            prompt,
            {"in_scope": False, "reasoning": "LLM evaluation failed or returned invalid JSON."},
            "fallback_scope",
        )

    async def _judge_hallucination(self, retrieved_context: list[str] | str, answer: str) -> dict[str, Any]:
        """Judge whether an answer contains unsupported content.

        Args:
            retrieved_context: Retrieved context snippets.
            answer: RAG answer.

        Returns:
            JSON judgment with hallucination fields.

        Raises:
            No exceptions are raised.
        """
        context = self._join_context(retrieved_context)
        prompt = f"""判断答案是否包含幻觉内容。
幻觉定义：答案中包含知识库/上下文无法支持的信息。

上下文: {context}
答案: {answer}

输出格式: {{"has_hallucination": false, "hallucinated_parts": [], "faithfulness_score": 1.0}}"""
        return await call_llm_json(
            self.llm_client,
            prompt,
            {"has_hallucination": True, "hallucinated_parts": [], "faithfulness_score": 0.0},
            "fallback_hallucination",
        )

    def _is_safe_reply(self, answer: str) -> bool:
        """Check whether an answer contains a safe fallback keyword.

        Args:
            answer: RAG answer.

        Returns:
            True when any configured safe keyword appears.

        Raises:
            No exceptions are raised.
        """
        if not answer:
            logger.warning("Empty answer treated as unsafe fallback reply")
            return False
        return any(keyword in answer for keyword in self.config.safe_reply_keywords)

    def _join_context(self, retrieved_context: list[str] | str) -> str:
        """Normalize retrieved context into text.

        Args:
            retrieved_context: List of snippets or pre-joined context.

        Returns:
            Context string.

        Raises:
            No exceptions are raised.
        """
        if isinstance(retrieved_context, list):
            return "\n".join(str(item) for item in retrieved_context if str(item).strip())
        return str(retrieved_context or "")

    def _empty_summary(self) -> dict[str, Any]:
        """Return an empty fallback summary.

        Args:
            None.

        Returns:
            Zero-valued summary.

        Raises:
            No exceptions are raised.
        """
        return {
            "reject_accuracy": 0.0,
            "hallucination_rate": 0.0,
            "safe_reply_rate": 0.0,
            "fallback_score": 0.0,
            "details": [],
        }

