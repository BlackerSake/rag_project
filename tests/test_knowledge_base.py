from types import SimpleNamespace

from langchain_core.documents import Document
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data import knowledge_base as knowledge_base_module
from data.knowledge_base import KnowledgeBase


def test_generate_doc_id_is_stable_for_same_content():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    first = Document(page_content="问题: 退货\n答案: 可以申请售后", metadata={"b": 2, "a": "中文"})
    second = Document(page_content="问题: 退货\n答案: 可以申请售后", metadata={"a": "中文", "b": 2})

    first_id = kb._generate_doc_id(first)
    second_id = kb._generate_doc_id(second)

    assert first_id == second_id
    assert len(first_id) == 32


def test_cache_key_is_stable():
    kb = KnowledgeBase.__new__(KnowledgeBase)

    assert kb._get_cache_key("退货", 3, {"intent": "A1"}) == kb._get_cache_key("退货", 3, {"intent": "A1"})


def test_connect_pymilvus_does_not_reset_alias_while_vector_store_is_alive(monkeypatch):
    calls = {"connect": 0, "disconnect": 0}

    def fail_if_connect_called(*args, **kwargs):
        calls["connect"] += 1
        raise AssertionError("connections.connect must not run while vector_store is alive")

    def fail_if_disconnect_called(*args, **kwargs):
        calls["disconnect"] += 1
        raise AssertionError("connections.disconnect must not run while vector_store is alive")

    fake_connections = SimpleNamespace(
        has_connection=lambda alias: True,
        connect=fail_if_connect_called,
        disconnect=fail_if_disconnect_called,
    )
    monkeypatch.setattr(knowledge_base_module, "connections", fake_connections)

    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._milvus_alias = "default"
    kb._milvus_uri = "http://localhost:19530"
    kb._milvus_connect_retries = 3
    kb.vector_store = object()
    kb._is_milvus_connection_alive = lambda: False

    assert kb._connect_pymilvus() is False
    assert calls == {"connect": 0, "disconnect": 0}
