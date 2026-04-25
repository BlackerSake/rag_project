"""Unit tests for the independent RAG evaluators."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.config import EvalConfig
from eval.evaluators import E2EEvaluator, FallbackEvaluator, GenerationEvaluator, RetrievalEvaluator


class FakeKnowledgeBase:
    """Minimal knowledge base mock used by RetrievalEvaluator tests."""

    def __init__(self, results):
        """Initialize with static retrieval results."""
        self.results = results

    def search(self, query, k=5):
        """Return the first k mocked results."""
        return self.results[:k]


class FakeRAG:
    """Minimal RAG mock used by FallbackEvaluator tests."""

    def __init__(self, answer_text):
        """Initialize with static answer text."""
        self.answer_text = answer_text

    async def answer(self, question):
        """Return the mocked answer asynchronously."""
        return self.answer_text


def test_retrieval_evaluator_calculates_all_metrics():
    docs = [
        (SimpleNamespace(metadata={"doc_id": "doc_b"}), 0.9),
        (SimpleNamespace(metadata={"doc_id": "doc_a"}), 0.8),
        (SimpleNamespace(metadata={"doc_id": "doc_c"}), 0.7),
    ]
    evaluator = RetrievalEvaluator(FakeKnowledgeBase(docs), EvalConfig(top_k=3))

    result = asyncio.run(evaluator.evaluate_case("退货", {"doc_a", "doc_c"}, k=3))

    assert result["retrieved_docs"] == ["doc_b", "doc_a", "doc_c"]
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["mrr"] == pytest.approx(1 / 2)
    assert result["ndcg_at_k"] > 0


def test_retrieval_evaluator_empty_batch_returns_zero_summary():
    evaluator = RetrievalEvaluator(FakeKnowledgeBase([]))

    result = asyncio.run(evaluator.evaluate_batch([]))

    assert result["retrieval_score"] == 0.0
    assert result["details"] == []


def test_generation_evaluator_uses_mock_llm_json():
    async def llm(prompt):
        return json.dumps(
            {
                "faithfulness": 0.8,
                "answer_relevancy": 0.9,
                "context_relevancy": 0.7,
                "context_recall": 0.6,
                "reasoning": {
                    "faithfulness_detail": "ok",
                    "answer_relevancy_detail": "ok",
                    "context_relevancy_detail": "ok",
                    "context_recall_detail": "ok",
                },
            }
        )

    evaluator = GenerationEvaluator(llm)
    result = asyncio.run(evaluator.evaluate_case(
        question="如何退货",
        generated_answer="在订单详情申请退货。",
        ground_truth_answer="在订单详情申请售后退货。",
        retrieved_context=["订单详情支持申请售后退货。"],
    ))

    assert result["faithfulness"] == pytest.approx(0.8)
    assert result["answer_relevancy"] == pytest.approx(0.9)
    assert result["context_relevancy"] == pytest.approx(0.7)
    assert result["context_recall"] == pytest.approx(0.6)
    assert result["generation_score"] > 0


def test_generation_evaluator_invalid_json_falls_back_to_zero():
    async def llm(prompt):
        return "not json"

    evaluator = GenerationEvaluator(llm)
    result = asyncio.run(evaluator.evaluate_case("q", "a", "gt", []))

    assert result["faithfulness"] == 0.0
    assert result["generation_score"] == 0.0


def test_e2e_evaluator_exact_match_scores_without_bertscore():
    evaluator = E2EEvaluator(enable_bertscore=False)

    result = evaluator.evaluate_case("客服工作时间是9点到18点", "客服工作时间是9点到18点")

    assert result["rouge_1"] == pytest.approx(1.0)
    assert result["rouge_l"] == pytest.approx(1.0)
    assert result["bleu"] > 0
    assert result["bertscore_f1"] == 0.0
    assert result["e2e_score"] > 0


def test_e2e_evaluator_empty_batch_returns_zero_summary():
    evaluator = E2EEvaluator(enable_bertscore=False)

    result = evaluator.evaluate_batch([])

    assert result["e2e_score"] == 0.0
    assert result["details"] == []


def test_fallback_evaluator_calculates_reject_and_safe_reply():
    calls = {"count": 0}

    async def llm(prompt):
        calls["count"] += 1
        if "覆盖范围" in prompt:
            return json.dumps({"in_scope": False, "reasoning": "实时天气不在知识库范围内"})
        return json.dumps({"has_hallucination": False, "hallucinated_parts": [], "faithfulness_score": 1.0})

    evaluator = FallbackEvaluator(FakeRAG("抱歉，这个问题超出范围，无法回答。"), llm)
    result = asyncio.run(
        evaluator.evaluate_batch(
            [{"question": "明天天气怎么样", "expected_behavior": "reject", "retrieved_context": []}]
        )
    )

    assert calls["count"] == 2
    assert result["reject_accuracy"] == pytest.approx(1.0)
    assert result["hallucination_rate"] == pytest.approx(0.0)
    assert result["safe_reply_rate"] == pytest.approx(1.0)
    assert result["fallback_score"] == pytest.approx(1.0)


def test_fallback_evaluator_invalid_llm_json_uses_fallback():
    async def llm(prompt):
        return "invalid json"

    evaluator = FallbackEvaluator(FakeRAG("这是一个没有依据的具体答案。"), llm)
    result = asyncio.run(
        evaluator.evaluate_batch(
            [{"question": "实时股票价格是多少", "expected_behavior": "reject", "retrieved_context": []}]
        )
    )

    assert result["reject_accuracy"] == pytest.approx(0.0)
    assert result["hallucination_rate"] == pytest.approx(1.0)
    assert result["safe_reply_rate"] == pytest.approx(0.0)
