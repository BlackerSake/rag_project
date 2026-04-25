"""LLM-based generation quality evaluator."""

from __future__ import annotations

import logging
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import LLMClient, call_llm_json, clamp_score, mean

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """Evaluate faithfulness, answer relevancy, context relevancy and recall."""

    def __init__(self, llm_client: LLMClient, config: EvalConfig | None = None) -> None:
        """Initialize the generation evaluator.

        Args:
            llm_client: Async or sync callable accepting a prompt and returning text.
            config: Optional evaluation configuration.

        Returns:
            None.

        Raises:
            ValueError: If ``llm_client`` is not callable.
        """
        if not callable(llm_client):
            raise ValueError("llm_client must be callable")
        self.llm_client = llm_client
        self.config = config or EvalConfig()

    async def evaluate_case(
        self,
        question: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_context: list[str] | str,
    ) -> dict[str, Any]:
        """Evaluate generation quality for one case with a single LLM call.

        Args:
            question: User question.
            generated_answer: Answer produced by the RAG system.
            ground_truth_answer: Reference answer.
            retrieved_context: Retrieved context snippets.

        Returns:
            Scores, reasoning details and aggregate generation score.

        Raises:
            No exceptions are raised; LLM failures use zero-score fallback.
        """
        prompt = self._build_prompt(question, generated_answer, ground_truth_answer, retrieved_context)
        fallback = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevancy": 0.0,
            "context_recall": 0.0,
            "reasoning": {
                "faithfulness_detail": "LLM evaluation failed or returned invalid JSON.",
                "answer_relevancy_detail": "LLM evaluation failed or returned invalid JSON.",
                "context_relevancy_detail": "LLM evaluation failed or returned invalid JSON.",
                "context_recall_detail": "LLM evaluation failed or returned invalid JSON.",
            },
        }
        result = await call_llm_json(self.llm_client, prompt, fallback, "generation_quality")
        normalized = {
            "faithfulness": clamp_score(result.get("faithfulness", 0.0)),
            "answer_relevancy": clamp_score(result.get("answer_relevancy", 0.0)),
            "context_relevancy": clamp_score(result.get("context_relevancy", 0.0)),
            "context_recall": clamp_score(result.get("context_recall", 0.0)),
            "reasoning": result.get("reasoning", fallback["reasoning"]),
        }
        normalized["generation_score"] = self.calculate_score(normalized)
        logger.info("Generation case evaluated: %.4f", normalized["generation_score"])
        return normalized

    async def evaluate_batch(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate generation quality for multiple cases.

        Args:
            cases: Items with question, answer, ground truth and context fields.

        Returns:
            Averaged metrics, aggregate score and per-case details.

        Raises:
            No exceptions are raised.
        """
        if not cases:
            logger.warning("Empty generation case list")
            return self._empty_summary()

        details = [
            await self.evaluate_case(
                question=case.get("question", ""),
                generated_answer=case.get("generated_answer", ""),
                ground_truth_answer=case.get("ground_truth_answer", ""),
                retrieved_context=case.get("retrieved_context", []),
            )
            for case in cases
        ]
        summary = {
            "faithfulness": mean([item["faithfulness"] for item in details]),
            "answer_relevancy": mean([item["answer_relevancy"] for item in details]),
            "context_relevancy": mean([item["context_relevancy"] for item in details]),
            "context_recall": mean([item["context_recall"] for item in details]),
            "details": details,
        }
        summary["generation_score"] = self.calculate_score(summary)
        logger.info("Generation batch evaluated: %.4f", summary["generation_score"])
        return summary

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """Aggregate generation metrics according to configured weights.

        Args:
            metrics: Dictionary with generation metric values.

        Returns:
            Weighted generation score.

        Raises:
            No exceptions are raised.
        """
        weights = self.config.generation_weights
        return (
            weights.faithfulness * metrics.get("faithfulness", 0.0)
            + weights.answer_relevancy * metrics.get("answer_relevancy", 0.0)
            + weights.context_relevancy * metrics.get("context_relevancy", 0.0)
            + weights.context_recall * metrics.get("context_recall", 0.0)
        )

    def _build_prompt(
        self,
        question: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_context: list[str] | str,
    ) -> str:
        """Build the combined generation-quality prompt.

        Args:
            question: User question.
            generated_answer: Generated answer.
            ground_truth_answer: Reference answer.
            retrieved_context: Context snippets or pre-joined context.

        Returns:
            Prompt string.

        Raises:
            No exceptions are raised.
        """
        if isinstance(retrieved_context, list):
            context = "\n".join(str(item) for item in retrieved_context if str(item).strip())
        else:
            context = str(retrieved_context or "")
        return f"""同时评估以下四个维度，返回JSON格式的评分。

维度1 - 忠实度(Faithfulness): 判断答案中的每个陈述是否可从上下文中推断出来。
维度2 - 答案相关性(Answer Relevancy): 判断答案是否直接回应了问题，问题是否被完整回答。
维度3 - 上下文相关性(Context Relevancy): 判断每个检索片段是否与问题相关。
维度4 - 上下文召回(Context Recall): 检查参考答案中的关键信息点有多少出现在检索上下文中。

问题: {question}
上下文: {context}
答案: {generated_answer}
参考答案: {ground_truth_answer}

输出格式:
{{
  "faithfulness": 0.0,
  "answer_relevancy": 0.0,
  "context_relevancy": 0.0,
  "context_recall": 0.0,
  "reasoning": {{
    "faithfulness_detail": "...",
    "answer_relevancy_detail": "...",
    "context_relevancy_detail": "...",
    "context_recall_detail": "..."
  }}
}}"""

    def _empty_summary(self) -> dict[str, Any]:
        """Return an empty generation summary.

        Args:
            None.

        Returns:
            Zero-valued summary.

        Raises:
            No exceptions are raised.
        """
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevancy": 0.0,
            "context_recall": 0.0,
            "generation_score": 0.0,
            "details": [],
        }

