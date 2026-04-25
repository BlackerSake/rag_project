"""不可回答场景的 RAG 兜底能力评测器。"""

from __future__ import annotations

import logging
from typing import Any

from eval.config import EvalConfig
from eval.evaluators.common import LLMClient, call_llm_json, clamp_score, maybe_await, mean

logger = logging.getLogger(__name__)


class FallbackEvaluator:
    """评估拒答准确率、幻觉率和安全回复率。"""

    def __init__(self, rag_system: Any, llm_client: LLMClient, config: EvalConfig | None = None) -> None:
        """初始化兜底能力评测器。

        参数:
            rag_system: 暴露同步或异步 ``answer(question)`` 方法的对象。
            llm_client: 接收 prompt 并返回文本的同步或异步可调用对象。
            config: 可选评测配置。

        返回:
            无。

        异常:
            ValueError: 当必要可调用对象缺失时抛出。
        """
        if not hasattr(rag_system, "answer") or not callable(rag_system.answer):
            raise ValueError("rag_system 必须暴露 answer(question) 方法")
        if not callable(llm_client):
            raise ValueError("llm_client 必须可调用")
        self.rag_system = rag_system
        self.llm_client = llm_client
        self.config = config or EvalConfig()

    async def evaluate_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """评估单个不可回答问题的兜底行为。

        参数:
            case: 包含 ``question``、可选 ``retrieved_context`` 和
                ``expected_behavior`` 字段的用例。

        返回:
            单条用例的兜底指标和 LLM 判断结果。

        异常:
            不主动抛出异常；RAG/LLM 失败会被降级处理。
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
            "兜底单用例评测完成: reject_correct=%s has_hallucination=%s is_safe=%s",
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
        """评估多条不可回答用例的兜底能力。

        参数:
            cases: 不可回答测试用例列表。

        返回:
            拒答准确率、幻觉率、安全回复率和聚合得分。

        异常:
            不主动抛出异常。
        """
        if not cases:
            logger.warning("兜底评测用例列表为空")
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
        logger.info("兜底批量评测完成: %.4f", summary["fallback_score"])
        return summary

    def calculate_score(self, metrics: dict[str, float]) -> float:
        """按配置权重聚合兜底能力指标。

        参数:
            metrics: 包含兜底能力指标值的字典。

        返回:
            加权兜底能力得分。

        异常:
            不主动抛出异常。
        """
        weights = self.config.fallback_weights
        return (
            weights.reject_accuracy * metrics.get("reject_accuracy", 0.0)
            + weights.non_hallucination_rate * (1.0 - metrics.get("hallucination_rate", 0.0))
            + weights.safe_reply_rate * metrics.get("safe_reply_rate", 0.0)
        )

    async def _answer_question(self, question: str) -> str:
        """调用注入的 RAG 系统。

        参数:
            question: 用户问题。

        返回:
            RAG 回答文本，或安全错误兜底回复。

        异常:
            不主动抛出异常。
        """
        try:
            logger.info("调用 RAG 系统处理兜底问题")
            answer = await maybe_await(self.rag_system.answer(question))
            return str(answer or "")
        except Exception as exc:
            logger.error("RAG 回答调用失败: %s", exc)
            return "抱歉，当前无法回答该问题。"

    async def _judge_scope(self, question: str, retrieved_context: list[str] | str) -> dict[str, Any]:
        """判断问题是否在知识库覆盖范围内。

        参数:
            question: 用户问题。
            retrieved_context: 检索上下文片段。

        返回:
            包含 ``in_scope`` 和 ``reasoning`` 的 JSON 判断结果。

        异常:
            不主动抛出异常。
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
            {"in_scope": False, "reasoning": "LLM 评测失败或返回了无效 JSON。"},
            "fallback_scope",
        )

    async def _judge_hallucination(self, retrieved_context: list[str] | str, answer: str) -> dict[str, Any]:
        """判断答案是否包含上下文不支持的内容。

        参数:
            retrieved_context: 检索上下文片段。
            answer: RAG 回答。

        返回:
            包含幻觉相关字段的 JSON 判断结果。

        异常:
            不主动抛出异常。
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
        """检查答案是否包含安全兜底关键词。

        参数:
            answer: RAG 回答。

        返回:
            任一已配置安全关键词出现时返回 True。

        异常:
            不主动抛出异常。
        """
        if not answer:
            logger.warning("空回答按非安全兜底回复处理")
            return False
        return any(keyword in answer for keyword in self.config.safe_reply_keywords)

    def _join_context(self, retrieved_context: list[str] | str) -> str:
        """将检索上下文标准化为文本。

        参数:
            retrieved_context: 片段列表或已拼接的上下文。

        返回:
            上下文字符串。

        异常:
            不主动抛出异常。
        """
        if isinstance(retrieved_context, list):
            return "\n".join(str(item) for item in retrieved_context if str(item).strip())
        return str(retrieved_context or "")

    def _empty_summary(self) -> dict[str, Any]:
        """返回空兜底能力评测汇总。

        参数:
            无。

        返回:
            全零汇总结果。

        异常:
            不主动抛出异常。
        """
        return {
            "reject_accuracy": 0.0,
            "hallucination_rate": 0.0,
            "safe_reply_rate": 0.0,
            "fallback_score": 0.0,
            "details": [],
        }
