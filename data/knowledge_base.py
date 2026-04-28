from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

try:
    from langchain_elasticsearch import ElasticsearchStore
except ImportError:
    from langchain_community.vectorstores import ElasticsearchStore
import hashlib
import os
from langchain_core.documents import Document
import time
import json
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache
from data.retrieval_evaluation import RetrievalEvaluationConfig, RetrievalEvaluator

# 尝试导入pymilvus
try:
    from pymilvus import Collection, connections, utility
except ImportError:
    Collection = None
    connections = None
    utility = None

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库类，提供向量检索和混合检索功能"""
    
    def __init__(self):
        """初始化知识库"""
        self._dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
        self._dashscope_model_id = os.getenv("DASHSCOPE_MODEL_ID")
        self._milvus_uri = self._normalize_milvus_uri(os.getenv("MILVUS_URI", "http://localhost:19530"))
        self._milvus_collection = os.getenv("MILVUS_COLLECTION_NAME", "customer_service")
        self._milvus_alias = "default"
        self._es_url = os.getenv("ES_URL", "http://localhost:9200")
        self._es_index = os.getenv("ES_INDEX_NAME", "customer_service")
        self._milvus_connect_retries = int(os.getenv("MILVUS_CONNECT_RETRIES", "3"))

        self.embeddings = DashScopeEmbeddings(
            model=self._dashscope_model_id,
            dashscope_api_key=self._dashscope_api_key
        )

        self.vector_store = None
        self.elasticsearch_store = None

        self._init_lock = threading.Lock()
        self._initialized = threading.Event()
        self._thread_pool = ThreadPoolExecutor(max_workers=4)

        self.cache = TTLCache(maxsize=1000, ttl=3600)

        self._retrieval_eval_enabled = os.getenv("RETRIEVAL_EVAL_ENABLED", "true").lower() == "true"
        self._retrieval_evaluator = RetrievalEvaluator(
            RetrievalEvaluationConfig(top_k=int(os.getenv("RETRIEVAL_EVAL_TOP_K", "5")))
        )
        self.retrieval_evaluation_events = []

        self._rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        self._rerank_top_k = int(os.getenv("RERANK_TOP_K", "6"))

    @staticmethod
    def _normalize_milvus_uri(raw_uri: str) -> str:
        """标准化Milvus URI"""
        milvus_uri = raw_uri.strip()
        if milvus_uri.startswith("tcp://"):
            return f"http://{milvus_uri[len('tcp://'):]}"
        return milvus_uri

    def _is_milvus_connection_alive(self) -> bool:
        """检查Milvus连接是否活跃"""
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
        """连接到Milvus"""
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

    def _init_milvus_vector_store(self) -> None:
        """初始化Milvus向量存储"""
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                logger.info("尝试初始化 Milvus 向量库 (尝试 %s/%s)", attempt + 1, max_attempts)
                self.vector_store = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self._milvus_collection,
                    connection_args={
                        "uri": self._milvus_uri,
                        "alias": self._milvus_alias
                    },
                    index_params={
                        "metric_type": "COSINE"
                    },
                    auto_id=True
                )
                logger.info("成功连接到 Milvus 向量数据库")
                return
            except Exception as exc:
                logger.error("连接 Milvus 向量数据库失败 (尝试 %s/%s): %s", attempt + 1, max_attempts, exc)
                self.vector_store = None
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 30))

        logger.error("连接 Milvus 向量数据库次数耗尽，最终失败")

    def _init_elasticsearch_store(self) -> None:
        """初始化Elasticsearch存储"""
        try:
            self.elasticsearch_store = ElasticsearchStore(
                es_url=self._es_url,
                index_name=self._es_index,
                embedding=self.embeddings
            )
            logger.info("成功连接到 Elasticsearch")
            self._initialize_elasticsearch_index()
        except Exception as exc:
            self.elasticsearch_store = None
            logger.error("连接 Elasticsearch 失败: %s", exc)

    def _is_connection_error(self, exc: Exception) -> bool:
        """判断是否为连接错误"""
        message = str(exc).lower()
        connection_markers = (
            "should create connection first",
            "connection",
            "connect",
            "grpc",
            "channel",
            "unavailable",
            "deadline exceeded",
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "broken pipe",
        )
        return any(marker in message for marker in connection_markers)

    def _rerank_with_dashscope(self, query: str, candidates: list, top_k: int) -> list:
        """使用Dashscope进行重排序"""
        if not self._rerank_enabled or not candidates:
            return candidates[:top_k]

        try:
            import dashscope
            from dashscope import TextEmbedding

            doc_texts = [doc.page_content for doc, score in candidates]

            doc_embeddings = TextEmbedding.call(
                [query] + doc_texts,
                model=self._dashscope_model_id,
                api_key=self._dashscope_api_key
            ).output.embeddings

            query_vec = doc_embeddings[0]
            doc_vecs = doc_embeddings[1:]

            def cosine_sim(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = sum(x * x for x in a) ** 0.5
                norm_b = sum(x * x for x in b) ** 0.5
                return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

            scored = []
            for i, doc_vec in enumerate(doc_vecs):
                score = cosine_sim(query_vec, doc_vec)
                scored.append((candidates[i][0], score))

            scored.sort(key=lambda x: x[1], reverse=True)
            logger.info("Dashscope Rerank 完成，从 %s 条候选中选取 top %s 条", len(candidates), top_k)
            return scored[:top_k]
        except Exception as e:
            logger.error("Dashscope Rerank 失败: %s，返回原始候选结果", e)
            return candidates[:top_k]

    def _generate_doc_id(self, doc: Document) -> str:
        """生成文档ID"""
        payload = doc.page_content + json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    # 移除重复的connect函数，统一使用ensure_connected函数

    def ensure_connected(self):
        """确保连接到存储服务"""
        if self.vector_store is not None and self.elasticsearch_store is not None:
            return self.vector_store

        with self._init_lock:
            if self._initialized.is_set():
                return self.vector_store

            if self.vector_store is None:
                if self._is_milvus_connection_alive() or self._connect_pymilvus():
                    self._init_milvus_vector_store()
                else:
                    logger.error("Milvus 连接不可用，跳过向量库初始化")

            if self.elasticsearch_store is None:
                self._init_elasticsearch_store()

            if self.vector_store is not None and self.elasticsearch_store is not None:
                self._initialized.set()

        return self.vector_store

    def _initialize_elasticsearch_index(self):
        """初始化Elasticsearch索引"""
        try:
            es_client = self.elasticsearch_store.client
            index_name = self._es_index

            if not es_client.indices.exists(index=index_name):
                test_text = "test_dimension"
                test_embedding = self.embeddings.embed_query(test_text)
                vector_dim = len(test_embedding)

                es_client.indices.create(
                    index=index_name,
                    body={
                        "mappings": {
                            "properties": {
                                "content": {
                                    "type": "text"
                                },
                                "metadata": {
                                    "type": "object"
                                },
                                "vector": {
                                    "type": "dense_vector",
                                    "dims": vector_dim
                                }
                            }
                        }
                    }
                )
                logger.info("成功创建 Elasticsearch 索引: %s, 向量维度: %s", index_name, vector_dim)
            else:
                logger.info("Elasticsearch 索引 %s 已存在", index_name)
        except Exception as e:
            logger.error("初始化 Elasticsearch 索引失败: %s", e)

    def get_milvus_collection_stats(self, collection_name: str | None = None) -> dict:
        """获取Milvus集合统计信息"""
        summary = {
            "collection_exists": False,
            "entity_count": 0,
            "fields": [],
        }
        if Collection is None or utility is None:
            logger.warning("pymilvus 未安装，无法检查 Milvus 集合")
            return summary

        collection_name = collection_name or self._milvus_collection
        if not self._is_milvus_connection_alive() and not self._connect_pymilvus():
            logger.warning("Milvus 连接不可用，无法检查集合: %s", collection_name)
            return summary

        collection = None
        try:
            if utility.has_collection(collection_name, using=self._milvus_alias):
                collection = Collection(collection_name, using=self._milvus_alias)
                collection.load()
                summary["collection_exists"] = True
                summary["entity_count"] = int(collection.num_entities)
                summary["fields"] = [field.name for field in collection.schema.fields]
        except Exception as e:
            logger.error("检查 Milvus 集合失败: %s", e)
        finally:
            if collection is not None:
                try:
                    collection.release()
                except Exception as e:
                    logger.warning("释放 Milvus 集合失败: %s", e)

        return summary

    def add_documents(self, file_paths):
        """添加文档到知识库"""
        self.ensure_connected()

        documents = []
        for file_path in file_paths:
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["doc_id"] = self._generate_doc_id(doc)
            documents.extend(docs)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        for doc in split_docs:
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = self._generate_doc_id(doc)

        self.vector_store.add_documents(split_docs)
        self.elasticsearch_store.add_documents(split_docs)

        self.cache.clear()

        return len(split_docs)

    def add_faq(self, faq_items):
        """添加FAQ到知识库"""
        self.ensure_connected()

        documents = []
        for ques, anws in faq_items.items():
            content = f"问题: {ques}\n答案: {anws}"
            doc = Document(
                page_content=content,
                metadata={"type": "faq", "question": ques}
            )
            doc.metadata["doc_id"] = self._generate_doc_id(doc)
            documents.append(doc)

        self.vector_store.add_documents(documents)
        self.elasticsearch_store.add_documents(documents)

        self.cache.clear()

        return len(documents)

    def _get_cache_key(self, query: str, k: int, filter_expr) -> str:
        """生成缓存键"""
        key_data = {"query": query, "k": k, "filter_expr": filter_expr}
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def vector_search(self, query: str, k: int, filter_expr=None):
        """执行向量检索"""
        vector_results = []
        vector_time = 0

        try:

            if self.vector_store:
                start = time.time()
                vector_search_kwargs = {"k": k}
                if filter_expr:
                    vector_search_kwargs["expr"] = filter_expr
                vector_results = self.vector_store.similarity_search_with_score(query, **vector_search_kwargs)
                vector_time = time.time() - start
                logger.info("向量检索成功，返回 %s 条结果", len(vector_results))
            
            else:
                # 兜底,使用原生 pymilvus
                logger.warning("向量存储未初始化，使用 pymilvus 原生进行兜底")
                col = self._get_milvus_collection()
                if col:
                    start_fallback = time.time()
                    query_vec = self.embeddings.embed_query(query)
                    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                    result = col.search(
                        data=[query_vec],
                        anns_field="vector",
                        param=search_params,
                        limit=k,
                        expr=filter_expr,
                        output_fields=["text", "mysql_id", "intent_id", "original_question",
                                   "action", "type", "doc_id"]
                    )

                    converted = []
                    if result and len(result) > 0:
                        for hit in result[0]:
                            metadata = {
                                "mysql_id": hit.entity.get("mysql_id"),
                                "intent_id": hit.entity.get("intent_id"),
                                "original_question": hit.entity.get("original_question"),
                                "action": hit.entity.get("action"),
                                "type": hit.entity.get("type"),
                                "doc_id": hit.entity.get("doc_id")
                            }
                            metadata = {k: v for k, v in metadata.items() if v is not None}
                            doc = Document(page_content=hit.entity.get("text", ""), metadata=metadata)
                            converted.append((doc, hit.score))
                    vector_time = time.time() - start_fallback
                    logger.info("pymilvus 兜底检索完成，返回 %s 条结果", len(converted))
                    return converted, vector_time

        except Exception as e:
            logger.error("廢了,langchain 和 pymilvus 向量检索都失败了: %s", e)

        return vector_results, vector_time

    def bm25_search(self, query: str, k: int):
        """执行BM25检索"""
        bm25_results = []
        bm25_time = 0
        try:
            if self.elasticsearch_store:
                start = time.time()
                es_client = self.elasticsearch_store.client

                response = es_client.search(
                    index=self._es_index,
                    body={
                        "query": {
                            "match": {
                                "text": query
                            }
                        },
                        "size": k,
                        "_source": True
                    }
                )

                for hit in response["hits"]["hits"]:
                    source = hit["_source"]
                    metadata = source.get("metadata", {})
                    page_content = source.get("text", "")
                    doc = Document(page_content=page_content, metadata=metadata)
                    bm25_results.append((doc, hit["_score"]))

                bm25_time = time.time() - start
                logger.info("BM25 純文本检索成功，返回 %s 条结果", len(bm25_results))
            else:
                logger.warning("Elasticsearch存储未初始化，跳过BM25检索")
        except Exception as e:
            logger.error("BM25检索失败: %s", e)
        return bm25_results, bm25_time

    def _local_fallback_search(self, query: str, k: int) -> list:
        """本地兜底搜索"""
        logger.warning("使用本地兜底搜索（difflib）")
        all_docs = []
        try:
            if self.vector_store:
                all_docs = self.vector_store.similarity_search(query, k=100)
            if not all_docs and self.elasticsearch_store:
                all_docs = self.elasticsearch_store.similarity_search(query, k=100)
        except Exception as e:
            logger.error("获取兜底文档失败: %s", e)

        if not all_docs:
            return []

        query_lower = query.lower()
        scored = []
        for doc in all_docs:
            content_lower = doc.page_content.lower()
            if query_lower in content_lower:
                score = len(query_lower) / len(content_lower)
            else:
                score = 0.0
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(doc, (score, "fallback")) for doc, score in scored[:k]]

    def hybrid_search(self, vector_results: list, bm25_results: list, k: int):
        """使用加权RRF算法合并检索结果"""
        doc_rank_map = {}

        # Vector 的結果
        for rank, (doc, score) in enumerate(vector_results, 1):
            doc_id = doc.metadata.get("doc_id", self._generate_doc_id(doc))
            if doc_id not in doc_rank_map:
                doc_rank_map[doc_id] = {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_score": score, "bm25_score": 0}
            doc_rank_map[doc_id]["vector_rank"] = rank
            doc_rank_map[doc_id]["vector_score"] = score

        # BM25 的結果
        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get("doc_id", self._generate_doc_id(doc))
            if doc_id not in doc_rank_map:
                doc_rank_map[doc_id] = {"doc": doc, "vector_rank": None, "bm25_rank": None, "vector_score": 0, "bm25_score": score}
            doc_rank_map[doc_id]["bm25_rank"] = rank
            doc_rank_map[doc_id]["bm25_score"] = score

        # 進行 RRF 加權計算
        rrf_k = 60
        vector_weight = 1.0
        bm25_weight = 1.0

        rrf_results = []
        for doc_id, data in doc_rank_map.items():
            rrf_score = 0.0
            if data["vector_rank"] is not None:
                rrf_score += vector_weight / (rrf_k + data["vector_rank"])
            if data["bm25_rank"] is not None:
                rrf_score += bm25_weight / (rrf_k + data["bm25_rank"])

            rrf_results.append({
                "doc": data["doc"],
                "rrf_score": rrf_score,
                "vector_score": data["vector_score"],
                "bm25_score": data["bm25_score"],
                "source": "hybrid"
            })

        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return rrf_results[:k]

    def search(
        self,
        query,
        k=3,
        filter_expr=None,
        evaluate=False,
        relevant_docs=None,
        expected_intent_id=None,
        evaluation_context=None,
    ):
        """执行搜索
        
        参数:
            query: 查询文本
            k: 返回结果数量
            filter_expr: 过滤表达式
            evaluate: 是否评估
            relevant_docs: 相关文档
            expected_intent_id: 预期意图ID
            evaluation_context: 评估上下文
            
        返回:
            搜索结果列表
        """
        self.ensure_connected()

        cache_key = self._get_cache_key(query, k, filter_expr)
        start_time = time.time()
        from_cache = False

        if cache_key in self.cache:
            final_results = self.cache[cache_key]
            from_cache = True
            response_time = time.time() - start_time
            self._async_log_search(query, len(final_results), response_time, from_cache)
            self._async_evaluate_retrieval(
                query,
                final_results,
                k,
                relevant_docs,
                expected_intent_id,
                response_time,
                from_cache,
                evaluation_context,
            )
            return final_results

        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self.vector_search, query, k * 2, filter_expr)
            bm25_future = executor.submit(self.bm25_search, query, k * 2)

            vector_results, vector_time = vector_future.result()
            bm25_results, bm25_time = bm25_future.result()

        rrf_results = self.hybrid_search(vector_results, bm25_results, k * 2)

        if rrf_results:
            candidates = [(r["doc"], r["rrf_score"]) for r in rrf_results]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("混合检索 + Rerank 成功，返回 %s 条结果", len(final_results))
        elif vector_results:
            candidates = vector_results[:k * 2]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("使用向量检索结果 + Rerank")
        elif bm25_results:
            candidates = bm25_results[:k * 2]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("使用BM25检索结果 + Rerank")
        else:
            logger.warning("所有检索都失败，尝试使用本地兜底搜索")
            fallback_results = self._local_fallback_search(query, k)
            if fallback_results:
                final_results = fallback_results
            else:
                final_results = []

        merge_time = time.time() - start_time - vector_time - bm25_time
        response_time = time.time() - start_time

        self._async_log_search(
            query, len(final_results), response_time, from_cache,
            vector_time, bm25_time, merge_time
        )

        self._async_evaluate_retrieval(
            query,
            final_results,
            k,
            relevant_docs,
            expected_intent_id,
            response_time,
            from_cache,
            evaluation_context,
        )

        self.cache[cache_key] = final_results

        return final_results

    def _async_log_search(self, query, result_count, response_time, from_cache, vector_time=0, bm25_time=0, merge_time=0):
        """异步记录搜索日志"""
        def log_task():
            try:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "result_count": result_count,
                    "response_time": response_time,
                    "from_cache": from_cache,
                    "vector_time": vector_time,
                    "bm25_time": bm25_time,
                    "merge_time": merge_time
                }
                # 记录到日志文件
                logger.info("Search: %s", json.dumps(log_entry, ensure_ascii=False))
            except Exception as e:
                logger.error("异步记录搜索日志失败: %s", e)

        self._thread_pool.submit(log_task)

    def _async_evaluate_retrieval(
        self,
        query,
        results,
        k,
        relevant_docs,
        expected_intent_id,
        response_time,
        from_cache,
        evaluation_context,
    ):
        """异步记录主系统检索评测事件。"""
        if not self._retrieval_eval_enabled:
            return

        def evaluate_task():
            try:
                event = self._retrieval_evaluator.evaluate_results(
                    query,
                    results,
                    k=k,
                    relevant_docs=relevant_docs,
                    expected_intent_id=expected_intent_id,
                    response_time=response_time,
                    from_cache=from_cache,
                    context=evaluation_context,
                )
                self.retrieval_evaluation_events.append(event)
                self._retrieval_evaluator.log_event(event)
            except Exception as e:
                logger.error("异步检索评测失败: %s", e)

        self._thread_pool.submit(evaluate_task)

    def get_retrieval_evaluation_summary(self):
        """返回当前进程内已记录检索评测事件的均值摘要。"""
        if not self.retrieval_evaluation_events:
            return {"message": "暂无检索评测数据"}

        metric_names = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k")
        summary = {
            "total_queries": len(self.retrieval_evaluation_events),
            "labeled_queries": 0,
            "intent_labeled_queries": 0,
            "unlabeled_queries": 0,
        }
        metric_values = {name: [] for name in metric_names}

        for event in self.retrieval_evaluation_events:
            status = event.get("metric_status", "unlabeled")
            summary[f"{status}_queries"] = summary.get(f"{status}_queries", 0) + 1
            metrics = event.get("metrics", {})
            for name in metric_names:
                value = metrics.get(name)
                if value is not None:
                    metric_values[name].append(value)

        for name, values in metric_values.items():
            summary[f"average_{name}"] = sum(values) / len(values) if values else None

        return summary
        # 可以加在 KnowledgeBase 类里，作为辅助方法
    def _get_milvus_collection(self):
        """获取pymilvus原生Collection对象，可作为备用方案。"""
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

"""
    def vector_search(self, query: str, k: int, filter_expr=None):
        #执行向量检索（带终极兜底
        vector_results = []
        vector_time = 0

        # ---- 1. 主路径：使用 langchain Milvus 对象 (self.vector_store) ----
        if self.vector_store:
            try:
                start = time.time()
                vector_search_kwargs = {"k": k}
                if filter_expr:
                    vector_search_kwargs["expr"] = filter_expr
                vector_results = self.vector_store.similarity_search_with_score(query, **vector_search_kwargs)
                vector_time = time.time() - start if 'start' in locals() else 0
                logger.info("向量检索成功（langchain），返回 %s 条结果", len(vector_results))
                return vector_results, vector_time
            except Exception as e:
                logger.warning("langchain向量检索失败，尝试使用 pymilvus 原生对象进行兜底: %s", e)
        
        # ---- 2. 兜底路径：使用 pymilvus 原生 Collection 对象 ----
        if not vector_results:
            try:
                col = self._get_milvus_collection()
                if col:
                    start_fallback = time.time()
                    
                    query_vector = self.embeddings.embed_query(query)
                    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
                    results = col.search(
                        data=[query_vector],
                        anns_field="vector",
                        param=search_params,
                        limit=k,
                        expr=filter_expr,
                        output_fields=["text", "mysql_id", "intent_id", "original_question", "action", "type", "doc_id"]
                    )

                    # 将原生结果转换为 LangChain Document 格式
                    from langchain_core.documents import Document
                    converted_results = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            # 重建metadata
                            metadata = {
                                "mysql_id": hit.entity.get("mysql_id"),
                                "intent_id": hit.entity.get("intent_id"),
                                "original_question": hit.entity.get("original_question"),
                                "action": hit.entity.get("action"),
                                "type": hit.entity.get("type"),
                                "doc_id": hit.entity.get("doc_id")
                            }
                            # 过滤掉值为None的字段，保持干净
                            metadata = {k: v for k, v in metadata.items() if v is not None}
                            
                            doc = Document(
                                page_content=hit.entity.get("text", ""),
                                metadata=metadata
                            )
                            converted_results.append((doc, hit.score))
                    
                    vector_time = time.time() - start_fallback if 'start_fallback' in locals() else 0
                    logger.info("向量检索成功（pymilvus 兜底），返回 %s 条结果", len(converted_results))
                    return converted_results, vector_time
            except Exception as e:
                logger.error("pymilvus 兜底检索失败: %s", e)

        # ---- 3. 所有尝试都失败 ----
        if not vector_results:
            logger.warning("所有向量检索路径均失败，跳过向量检索")
        
        return vector_results, vector_time
"""
