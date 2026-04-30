from types import SimpleNamespace
import sys
from pathlib import Path

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.confidence import ConfidenceGate, ConfidenceHistory
from data.retrieval import RetrievalService
from evaluation.run_confidence_eval import extract_signal, summarize


def test_confidence_history_keeps_fifo_window():
    history = ConfidenceHistory(max_size=3)

    for score in [0.1, 0.2, 0.3, 0.4]:
        history.update(score)

    assert history.values() == [0.2, 0.3, 0.4]


def test_confidence_gate_uses_fallback_for_small_sample_and_percentiles_afterwards():
    history = ConfidenceHistory(max_size=100)
    gate = ConfidenceGate(history)

    assert gate.decide(0.7) == {
        "decision": "MEDIUM",
        "confidence_score": 0.7,
        "p25": 0.5,
        "p75": 0.9,
        "sample_count": 0,
        "window_size": 100,
    }

    history.extend([0.1, 0.2, 0.8, 0.9])
    thresholds = gate.calibrate_thresholds()

    assert thresholds.p25 == 0.17500000000000002
    assert thresholds.p75 == 0.8250000000000001
    assert gate.decide(0.9)["decision"] == "HIGH"
    assert gate.decide(0.5)["decision"] == "MEDIUM"
    assert gate.decide(0.1)["decision"] == "LOW"


def test_vector_search_updates_confidence_history_with_top1_score():
    class FakeVectorStore:
        def similarity_search_with_score(self, query, **kwargs):
            return [
                (Document(page_content="a", metadata={"mysql_id": 1}), 0.72),
                (Document(page_content="b", metadata={"mysql_id": 2}), 0.41),
            ]

    stores = SimpleNamespace(vector_store=FakeVectorStore())
    config = SimpleNamespace(
        confidence_window_size=3,
        confidence_fallback_p25=0.5,
        confidence_fallback_p75=0.9,
    )
    service = RetrievalService(config=config, stores=stores, cache={}, thread_pool=None, observer=None)

    results, _elapsed = service.vector_search("退货", k=2)

    assert len(results) == 2
    assert service.confidence_history.values() == [0.72]
    assert service.confidence_gate(0.72)["decision"] == "MEDIUM"


def test_margin_signal_degraded_to_top1_does_not_update_window():
    result = extract_signal(
        [(Document(page_content="only", metadata={"mysql_id": 1}), 0.81)],
        signal_type="margin",
    )

    assert result == {
        "score": 0.81,
        "degraded": True,
        "update_window": False,
        "reason": "margin_degraded_to_top1",
    }


def test_summarize_uses_record_phase_labels_instead_of_index_slices():
    records = [
        {"phase": "steady", "decision": "HIGH", "recall_at_k": 1.0},
        {"phase": "steady", "decision": "LOW", "recall_at_k": 0.0},
    ]

    summary = summarize(records, window_size=100)

    assert summary["phase_summary"]["fill"]["num_queries"] == 0
    assert summary["phase_summary"]["steady"]["num_queries"] == 2
