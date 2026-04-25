"""基于 LLM 的生成质量评测器。"""

from __future__ import annotations

import logging
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import LLMClient, call_llm_json, clamp_score, mean

logger = logging.getLogger(__name__)


class GenerationEvaluator:
    """评估忠实度、答案相关性、上下文相关性和上下文召回。"""

    def __init__(self, llm_client: LLMClient, config: EvalConfig | None = None) -> None:
        """初始化生成质量评测器。

        参数:
            llm_client: 接收 prompt 并返回文本的同步或异步可调用对象。
            config: 可选评测配置。

        返回:
            无。

        异常:
            ValueError: 当 ``llm_client`` 不可调用时抛出。
        """
        if not callable(llm_client):
            raise ValueError("llm_client 必须可调用")
        self.llm_client = llm_client
        self.config = config or EvalConfig()

    async def evaluate_case(
        self,
        question: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_context: list[str] | str,
    ) -> dict[str, Any]:
        """通过一次 LLM 调用评估单条用例的生成质量。

        参数:
            question: 用户问题。
            generated_answer: RAG 系统生成的答案。
            ground_truth_answer: 参考答案。
            retrieved_context: 检索上下文片段。

        返回:
            各项分数、推理说明和生成质量聚合得分。

        异常:
            不主动抛出异常；LLM 失败时使用零分降级结果。
        """
        prompt = self._build_prompt(question, generated_answer, ground_truth_answer, retrieved_context)
        fallback = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevancy": 0.0,
            "context_recall": 0.0,
            "reasoning": {
                "faithfulness_detail": "LLM 评测失败或返回了无效 JSON。",
                "answer_relevancy_detail": "LLM 评测失败或返回了无效 JSON。",
                "context_relevancy_detail": "LLM 评测失败或返回了无效 JSON。",
                "context_recall_detail": "LLM 评测失败或返回了无效 JSON。",
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
        logger.info("生成质量单用例评测完成: %.4f", normalized["generation_score"])
        return normalized

    async def evaluate_batch(self, cases: list[dict[str, Any]]) -> dict[str, Any]:
        """评估多条用例的生成质量。

        参数:
            cases: 包含问题、答案、参考答案和上下文字段的用例列表。

        返回:
            平均指标、聚合得分和用例详情。

        异常:
            不主动抛出异常。
        """
        if not cases:
            logger.warning("生成质量评测用例列表为空")
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
        logger.info("生成质量批量评测完成: %.4f", summary["generation_score"])
        return summary

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """按配置权重聚合生成质量指标。

        参数:
            metrics: 包含生成质量指标值的字典。

        返回:
            加权生成质量得分。

        异常:
            不主动抛出异常。
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
        """构造生成质量的合并评测提示词。

        参数:
            question: 用户问题。
            generated_answer: 生成答案。
            ground_truth_answer: 参考答案。
            retrieved_context: 上下文片段或已拼接的上下文。

        返回:
            提示词字符串。

        异常:
            不主动抛出异常。
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
        """返回空生成质量评测汇总。

        参数:
            无。

        返回:
            全零汇总结果。

        异常:
            不主动抛出异常。
        """
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_relevancy": 0.0,
            "context_recall": 0.0,
            "generation_score": 0.0,
            "details": [],
        }
