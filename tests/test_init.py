import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from init import FAQItem, build_intents_from_faq, load_faq_data, normalize_milvus_uri


def test_normalize_milvus_uri_converts_tcp_to_http():
    assert normalize_milvus_uri("tcp://localhost:19530") == "http://localhost:19530"
    assert normalize_milvus_uri("http://localhost:19530") == "http://localhost:19530"


def test_load_faq_data_rejects_duplicate_intent_question(tmp_path):
    payload = [
        {"intent_id": "A1", "question": "怎么退货", "answer": "打开订单申请售后"},
        {"intent_id": "A1", "question": "怎么退货", "answer": "重复问题"},
    ]
    path = tmp_path / "faq_data.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError):
        load_faq_data(path)


def test_build_intents_from_faq_preserves_d1_d2():
    faq_items = [
        FAQItem(intent_id="A1", question="怎么退货", answer="打开订单申请售后"),
        FAQItem(intent_id="D1", question="你好", answer="您好"),
        FAQItem(intent_id="D2", question="转人工", answer="正在为您转接"),
    ]
    existing = {
        "D1": {"name": "寒暄", "description": "保留寒暄节点", "intent_id": "D1"},
        "D2": {"name": "人工介入", "description": "保留人工节点", "intent_id": "D2"},
    }

    intents = build_intents_from_faq(faq_items, existing)

    assert "A1" in intents
    assert intents["D1"] == existing["D1"]
    assert intents["D2"] == existing["D2"]
