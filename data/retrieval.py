
import time
import json
import hashlib
import logging

from data.stores import StoreManager
from data.kb_config import KBConfig

from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(self, config: KBConfig ,stores: StoreManager, cache, thead_pool):
        self.config = config
        self.stores = stores
        self.cache = cache
        self._thead_pool = thead_pool
    def vector_search(self, query: str, k: int, filter_expr=None):
        """執行向量檢索"""
        vector_results = []
        vector_time = 0

        try:

            if self.stores.vector_store:
                start = time.time()
                vector_search_kwargs = {"k": k}
                if filter_expr:
                    vector_search_kwargs["expr"] = filter_expr
                vector_results = self.stores.vector_store.similarity_search_with_score(query, **vector_search_kwargs)
                vector_time = time.time() - start
                logger.info("向量檢索成功，返回 %s 條結果", len(vector_results))
            
            else:
                # 兜底,使用原生 pymilvus
                logger.warning("向量存儲未初始化，使用 pymilvus 原生進行兜底")
                col = self._get_milvus_collection()
                if col:
                    start_fallback = time.time()
                    query_vec = self.stores.embeddings.embed_query(query)
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
                    logger.info("pymilvus 兜底檢索完成，返回 %s 條結果", len(converted))
                    return converted, vector_time

        except Exception as e:
            logger.error("廢了,langchain 和 pymilvus 向量檢索都失敗了: %s", e)

        return vector_results, vector_time

    def bm25_search(self, query: str, k: int):
        """執行BM25檢索"""
        bm25_results = []
        bm25_time = 0
        try:
            if self.elasticsearch_store:
                start = time.time()
                es_client = self.elasticsearch_store.client

                response = es_client.search(
                    index=self.config.es_index,
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
                logger.info("BM25 純文本檢索成功，返回 %s 條結果", len(bm25_results))
            else:
                logger.warning("Elasticsearch存儲未初始化，跳过BM25檢索")
        except Exception as e:
            logger.error("BM25檢索失敗: %s", e)
        return bm25_results, bm25_time

    def _local_fallback_search(self, query: str, k: int) -> list:
        """本地兜底搜索"""
        logger.warning("使用本地兜底搜索（difflib）")
        all_docs = []
        try:
            if self.stores.vector_store:
                all_docs = self.stores.vector_store.similarity_search(query, k=100)
            if not all_docs and self.elasticsearch_store:
                all_docs = self.elasticsearch_store.similarity_search(query, k=100)
        except Exception as e:
            logger.error("获取兜底文檔失敗: %s", e)

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
        """使用加權 RRF 算法合併檢索結果"""
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
        """執行搜索
        
        Args:
            query: 查詢文本
            k: 返回結果數量
            filter_expr: 過濾表達式
            evaluate: 是否評估
            relevant_docs: 相关文檔
            expected_intent_id: 预期意图ID
            evaluation_context: 評估上下文
            
        Returns:
            搜索結果列表
        """
        self.stores.ensure_connected()

        cache_key = self.observer._get_cache_key()(query, k, filter_expr)
        start_time = time.time()
        from_cache = False

        if cache_key in self.cache:
            final_results = self.cache[cache_key]
            from_cache = True
            response_time = time.time() - start_time
            self.observer._async_log_search(query, len(final_results), response_time, from_cache)
            self.observer._async_evaluate_retrieval(
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
            logger.info("混合檢索 + Rerank 成功，返回 %s 條結果", len(final_results))
        elif vector_results:
            candidates = vector_results[:k * 2]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("使用向量檢索結果 + Rerank")
        elif bm25_results:
            candidates = bm25_results[:k * 2]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("使用BM25檢索結果 + Rerank")
        else:
            logger.warning("所有檢索都失敗，尝试使用本地兜底搜索")
            fallback_results = self._local_fallback_search(query, k)
            if fallback_results:
                final_results = fallback_results
            else:
                final_results = []

        merge_time = time.time() - start_time - vector_time - bm25_time
        response_time = time.time() - start_time

        self.observer._async_log_search(
            query, len(final_results), response_time, from_cache,
            vector_time, bm25_time, merge_time
        )

        self.observer._async_evaluate_retrieval(
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

    def _rerank_with_dashscope(self, query: str, candidates: list, top_k: int) -> list:
        """使用 Dashscope 進行重排序"""
        if not self.config.rerank_enabled or not candidates:
            return candidates[:top_k]

        try:
            import dashscope
            from dashscope import TextEmbedding

            doc_texts = [doc.page_content for doc, score in candidates]

            doc_embeddings = TextEmbedding.call(
                [query] + doc_texts,
                model=self.config.dashscope_model_id,
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
            logger.info("Dashscope Rerank 完成，从 %s 條候选中选取 top %s 條", len(candidates), top_k)
            return scored[:top_k]
        except Exception as e:
            logger.error("Dashscope Rerank 失敗: %s，返回原始候选結果", e)
            return candidates[:top_k]
    
    def _get_cache_key(self, query: str, k: int, filter_expr) -> str:
        """生成缓存键"""
        key_data = {"query": query, "k": k, "filter_expr": filter_expr}
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()









