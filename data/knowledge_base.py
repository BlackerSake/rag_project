from data.kb_config import KBConfig
from data.stores import Collection, StoreManager, connections, utility
from data.retrieval import RetrievalService
from data.ingestion import DocumentManager
from data.observability import SearchLogger
from data.retrieval_evaluation import RetrievalEvaluationConfig, RetrievalEvaluator


from langchain_community.embeddings import DashScopeEmbeddings
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import time

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, config: KBConfig | None = None):
        self.config = config or KBConfig.from_env()
        self._dashscope_api_key = self.config.dashscope_api_key
        self._dashscope_model_id = self.config.dashscope_model_id
        self._milvus_uri = self.config.milvus_uri
        self._milvus_collection = self.config.milvus_collection
        self._milvus_alias = self.config.milvus_alias
        self._milvus_connect_retries = self.config.milvus_connect_retries
        self._es_url = self.config.es_url
        self._es_index = self.config.es_index
        self._rerank_enabled = self.config.rerank_enabled
        self._rerank_top_k = self.config.rerank_top_k
        self._retrieval_eval_enabled = self.config.retrieval_eval_enabled

        self.embeddings = DashScopeEmbeddings(
            model=self.config.dashscope_model_id,
            dashscope_api_key=self.config.dashscope_api_key
        )

        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self.cache = TTLCache(maxsize=self.config.cache_maxsize, ttl=self.config.cache_ttl)
        evaluator = RetrievalEvaluator(RetrievalEvaluationConfig(top_k=self.config.retrieval_eval_top_k))

        self.stores = StoreManager(self.config, self.embeddings)
        self.observer = SearchLogger(self.config, self._thread_pool, evaluator)
        self.retriever = RetrievalService(self.config, self.stores, self.cache, self._thread_pool, self.observer)
        self.writer = DocumentManager(self.config, self.stores, self.cache)
        self._retrieval_evaluator = evaluator
        self.retrieval_evaluation_events = self.observer.retrieval_evaluation_events

    @property
    def vector_store(self):
        if hasattr(self, "stores"):
            return self.stores.vector_store
        return getattr(self, "_vector_store", None)

    @vector_store.setter
    def vector_store(self, value):
        if hasattr(self, "stores"):
            self.stores.vector_store = value
        else:
            self._vector_store = value

    @property
    def elasticsearch_store(self):
        if hasattr(self, "stores"):
            return self.stores.elasticsearch_store
        return getattr(self, "_elasticsearch_store", None)

    @elasticsearch_store.setter
    def elasticsearch_store(self, value):
        if hasattr(self, "stores"):
            self.stores.elasticsearch_store = value
        else:
            self._elasticsearch_store = value

    # --- 對外接口，全部委託給子模塊 ---
    def search(self, *args, **kwargs):
        return self.retriever.search(*args, **kwargs)

    def ensure_connected(self):
        return self.stores.ensure_connected()

    def add_documents(self, file_paths):
        return self.writer.add_documents(file_paths)

    def add_faq(self, faq_items):
        return self.writer.add_faq(faq_items)

    def vector_search(self, *args, **kwargs):
        return self.retriever.vector_search(*args, **kwargs)

    def bm25_search(self, *args, **kwargs):
        return self.retriever.bm25_search(*args, **kwargs)

    def hybrid_search(self, *args, **kwargs):
        return self.retriever.hybrid_search(*args, **kwargs)

    def _rerank_with_dashscope(self, *args, **kwargs):
        return self.retriever._rerank_with_dashscope(*args, **kwargs)
    
    def hybrid_rerank_search(self, *args, **kwargs):
        return self.retriever.hybrid_rerank_search(*args, **kwargs)
    
    def hybrid_rerank_retrieve(self, *args, **kwargs):
        return self.retriever.hybrid_rerank_retrieve(*args, **kwargs)

    def multi_query_retrieve(self, *args, **kwargs):
        return self.retriever.multi_query_retrieve(*args, **kwargs)
    
    def intent_filtered_retrieve(self, *args, **kwargs):
        return self.retriever.intent_filtered_retrieve(*args, **kwargs)
    
    def rewrite_query(self, *args, **kwargs):
        return self.retriever.rewrite_query(*args, **kwargs)

    def confidence_gate(self, *args, **kwargs):
        return self.retriever.confidence_gate(*args, **kwargs)

    def seed_confidence_history(self, *args, **kwargs):
        return self.retriever.seed_confidence_history(*args, **kwargs)

    def reset_confidence_history(self):
        return self.retriever.reset_confidence_history()

    def get_confidence_thresholds(self):
        return self.retriever._calibrate_thresholds()

    def get_milvus_collection_stats(self, collection_name=None):
        return self.stores.get_milvus_collection_stats(collection_name)

    def get_retrieval_evaluation_summary(self):
        return self.observer.get_retrieval_evaluation_summary()

    def _generate_doc_id(self, doc: Document) -> str:
        """生成文档ID，保留原 KnowledgeBase API。"""
        payload = doc.page_content + json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _get_cache_key(self, query: str, k: int, filter_expr) -> str:
        """生成缓存键，保留原 KnowledgeBase API。"""
        key_data = {"query": query, "k": k, "filter_expr": filter_expr}
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _is_milvus_connection_alive(self) -> bool:
        if hasattr(self, "stores"):
            return self.stores._is_milvus_connection_alive()

        if connections is None or utility is None:
            logger.warning("pymilvus 未安装，跳过 Milvus 探活")
            return False
        try:
            if not connections.has_connection(self._milvus_alias):
                return False
            utility.list_collections(using=self._milvus_alias, timeout=3)
            return True
        except Exception as exc:
            logger.warning("Milvus 连接探活失败: %s", exc)
            return False

    def _connect_pymilvus(self) -> bool:
        if hasattr(self, "stores"):
            return self.stores._connect_pymilvus()

        if connections is None:
            logger.warning("pymilvus 未安装，无法连接 Milvus")
            return False
        for attempt in range(self._milvus_connect_retries):
            try:
                if connections.has_connection(self._milvus_alias):
                    if self._is_milvus_connection_alive():
                        logger.info("Milvus alias %s 已可用，复用现有连接", self._milvus_alias)
                        return True
                    if self.vector_store is not None:
                        logger.warning(
                            "Milvus alias %s 探活失败，但 Milvus 对象仍存活，跳过重连以避免破坏业务对象",
                            self._milvus_alias,
                        )
                        return False
                logger.info("尝试连接 Milvus: %s (尝试 %s/%s)", self._milvus_uri, attempt + 1, self._milvus_connect_retries)
                connections.connect(alias=self._milvus_alias, uri=self._milvus_uri)
                if self._is_milvus_connection_alive():
                    logger.info("成功连接到 Milvus")
                    return True
            except Exception as exc:
                logger.error("建立 pymilvus 连接失败 (尝试 %s/%s): %s", attempt + 1, self._milvus_connect_retries, exc)

            if attempt < self._milvus_connect_retries - 1:
                time.sleep(min(2 ** attempt, 30))
        logger.error("建立 pymilvus 连接次数耗尽，最终失败")
        return False

    def _get_milvus_collection(self):
        if hasattr(self, "stores"):
            return self.stores._get_milvus_collection()

        if Collection is None or connections is None:
            return None
        try:
            if not connections.has_connection(self._milvus_alias):
                logger.warning("原生连接不存在，尝试建立...")
                connections.connect(alias=self._milvus_alias, uri=self._milvus_uri)
            col = Collection(self._milvus_collection, using=self._milvus_alias)
            col.load()
            return col
        except Exception as e:
            logger.error("获取原生Milvus集合失败: %s", e)
            return None
