import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.retrieval_evaluation import RetrievalEvaluationConfig, RetrievalEvaluator


def test_labeled_retrieval_evaluation_calculates_core_metrics():
    """测试有标签的检索评估能计算核心指标"""
    evaluator = RetrievalEvaluator(RetrievalEvaluationConfig(top_k=3))
    results = [
        (Document(page_content="问题: 退货流程", metadata={"doc_id": "doc_a"}), 0.91),
        (Document(page_content="问题: 配送时间", metadata={"doc_id": "doc_b"}), 0.82),
        (Document(page_content="问题: 退款到账", metadata={"doc_id": "doc_c"}), 0.73),
    ]

    event = evaluator.evaluate_results("如何退货", results, k=3, relevant_docs={"doc_a", "doc_c"})

    assert event["metric_status"] == "labeled"
    assert event["metrics"]["recall_at_k"] == 1.0
    assert event["metrics"]["precision_at_k"] == pytest.approx(2 / 3)
    assert event["metrics"]["mrr"] == 1.0
    assert event["metrics"]["ndcg_at_k"] == pytest.approx(0.9197207891)
    assert event["retrieval_score"] is not None


def test_intent_labeled_evaluation_does_not_fake_recall():
    """测试意图标签评估不会伪造召回率"""
    evaluator = RetrievalEvaluator(RetrievalEvaluationConfig(top_k=2))
    results = [
        (Document(page_content="问题: 配送时间", metadata={"intent_id": "L1", "mysql_id": "1"}), 0.9),
        (Document(page_content="问题: 退货流程", metadata={"intent_id": "A1", "mysql_id": "2"}), 0.8),
    ]

    event = evaluator.evaluate_results("如何退货", results, k=2, expected_intent_id="A1")

    assert event["metric_status"] == "intent_labeled"
    assert event["metrics"]["recall_at_k"] is None
    assert event["metrics"]["precision_at_k"] == pytest.approx(0.5)
    assert event["metrics"]["mrr"] == pytest.approx(0.5)
    assert event["retrieval_score"] is None


def test_unlabeled_evaluation_records_event_without_metrics():
    """测试无标签评估会记录事件但无指标"""
    evaluator = RetrievalEvaluator()
    event = evaluator.evaluate_results("你好", [], k=5)

    assert event["metric_status"] == "unlabeled"
    assert event["metrics"] == {
        "recall_at_k": None,
        "precision_at_k": None,
        "mrr": None,
        "ndcg_at_k": None,
    }
