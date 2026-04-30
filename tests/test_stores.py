from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import stores as stores_module
from data.stores import StoreManager


def test_init_milvus_vector_store_registers_langchain_orm_alias(monkeypatch):
    registered_aliases = set()
    connect_calls = []
    client_args = []

    class FakeMilvusClient:
        def __init__(self, **kwargs):
            client_args.append(kwargs)
            self._using = "cm-test"

    fake_connections = SimpleNamespace(
        has_connection=lambda alias: alias in registered_aliases,
        connect=lambda alias, **kwargs: (registered_aliases.add(alias), connect_calls.append((alias, kwargs))),
    )

    class FakeMilvus:
        def __init__(self, **kwargs):
            assert "alias" not in kwargs["connection_args"]
            assert "cm-test" in registered_aliases
            self.kwargs = kwargs

    monkeypatch.setattr(stores_module, "MilvusClient", FakeMilvusClient)
    monkeypatch.setattr(stores_module, "connections", fake_connections)
    monkeypatch.setattr(stores_module, "Milvus", FakeMilvus)

    config = SimpleNamespace(
        milvus_uri="http://localhost:19530",
        milvus_collection="customer_service",
        milvus_timeout=5,
    )
    manager = StoreManager(config, embeddings=object())

    manager._init_milvus_vector_store()

    assert manager.vector_store is not None
    assert client_args == [{"uri": "http://localhost:19530", "timeout": 5}]
    assert connect_calls == [("cm-test", {"uri": "http://localhost:19530", "timeout": 5})]
