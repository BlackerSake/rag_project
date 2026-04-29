from data.kb_config import KBConfig
from data.stores import StoreManager
from data.retrieval import RetrievalService
from data.ingestion import DocumentManager
from data.observability import SearchLogger
from data.retrieval_evaluation import RetrievalEvaluationConfig, RetrievalEvaluator


from langchain_community.embeddings import DashScopeEmbeddings
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor

class KnowledgeBase:
    def __init__(self, config: KBConfig | None = None):
        self.config = config or KBConfig.from_env()

        self.embeddings = DashScopeEmbeddings(
            model=self.config.dashscope_model_id,
            dashscope_api_key=self.config.dashscope_api_key
        )

        thread_pool = ThreadPoolExecutor(max_workers=4)
        cache = TTLCache(maxsize=self.config.cache_maxsize, ttl=self.config.cache_ttl)
        evaluator = RetrievalEvaluator(RetrievalEvaluationConfig(top_k=self.config.retrieval_eval_top_k))

        self.stores = StoreManager(self.config, self.embeddings)
        self.observer = SearchLogger(self.config, thread_pool, evaluator)
        self.retriever = RetrievalService(self.config, self.stores, cache, thread_pool, self.observer)
        self.writer = DocumentManager(self.config, self.stores, cache)

    # --- 對外接口，全部委託給子模塊 ---
    def search(self, *args, **kwargs):
        return self.retriever.search(*args, **kwargs)

    def add_documents(self, file_paths):
        return self.writer.add_documents(file_paths)

    def add_faq(self, faq_items):
        return self.writer.add_faq(faq_items)

    def get_milvus_collection_stats(self, collection_name=None):
        return self.stores.get_milvus_collection_stats(collection_name)

    def get_retrieval_evaluation_summary(self):
        return self.observer.get_retrieval_evaluation_summary()