

import time
import json
import hashlib
import logging

from dashscope import Generation
from dotenv.main import rewrite
import yaml
from pathlib import Path

from data.stores import StoreManager
from data.kb_config import KBConfig
from data.query_processing import split_user_input

from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, config: KBConfig, stores: StoreManager, cache, thread_pool, observer):
        self.config = config
        self.stores = stores
        self.cache = cache
        self._thread_pool = thread_pool
        self.observer = observer

    def _generate_doc_id(self, doc: Document) -> str:
        """生成稳定文档ID。"""
        payload = doc.page_content + json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

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
                logger.info("向量檢索成功✅，返回 %s 條結果", len(vector_results))

            else:
                # 兜底,使用原生 pymilvus
                logger.warning("向量存儲未初始化❌，使用 pymilvus 原生進行兜底")
                col = self.stores._get_milvus_collection()
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
                    logger.info("pymilvus 兜底檢索完成✅，返回 %s 條結果", len(converted))
                    return converted, vector_time

        except Exception as e:
            logger.error("廢了,langchain 和 pymilvus 向量檢索都失敗了❌❌: %s", e)

        return vector_results, vector_time

    def bm25_search(self, query: str, k: int):
        """執行BM25檢索"""
        bm25_results = []
        bm25_time = 0
        try:
            if self.stores.elasticsearch_store:
                start = time.time()
                es_client = self.stores.elasticsearch_store.client

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
                logger.info("BM25 純文本檢索成功✅，返回 %s 條結果", len(bm25_results))
            else:
                logger.warning("Elasticsearch存儲未初始化❌，跳过BM25檢索")
        except Exception as e:
            logger.error("BM25檢索失敗❌: %s", e)
        return bm25_results, bm25_time

    
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

    def _rerank_with_dashscope(self, query: str, candidates: list, top_k: int) -> list:
        """使用 Qwen3-rerank 進行真正的二階重排序。"""
        logger.info("即將開始使用 Qwen3-rerank 進行重排序...")

        if not self.config.rerank_enabled or not candidates:
            logger.warning("重排序功能已禁用或沒有候選項目❌")
            return candidates[:top_k]

        try:
            import dashscope

            doc_texts = [doc.page_content for doc, _score in candidates]

            # 建議單獨配置 rerank_model，不要復用 embedding model
            rerank_model = getattr(self.config, "rerank_model", None) or "qwen3-rerank"

            resp = dashscope.TextReRank.call(
                model=rerank_model,
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(doc_texts)),
                api_key=self.config.dashscope_api_key,
            )

            if getattr(resp, "status_code", 200) != 200:
                logger.error(
                    "Qwen3-rerank 調用失敗❌: status_code=%s, code=%s, message=%s",
                    getattr(resp, "status_code", None),
                    getattr(resp, "code", None),
                    getattr(resp, "message", None),
                )
                return candidates[:top_k]

            rerank_results = self._extract_dashscope_rerank_results(resp)

            if not rerank_results:
                logger.error("Qwen3-rerank 返回結果為空❌，返回原始候選結果")
                return candidates[:top_k]

            final_results = []
            for item in rerank_results:
                index = item["index"]
                score = item["relevance_score"]

                if 0 <= index < len(candidates):
                    doc = candidates[index][0]
                    final_results.append((doc, score))

            if not final_results:
                logger.error("Qwen3-rerank 結果 index 無效❌，返回原始候選結果")
                return candidates[:top_k]

            logger.info(
                "Qwen3-rerank 完成✅，從 %s 條候選中選取 top %s 條",
                len(candidates),
                top_k,
            )
            return final_results[:top_k]

        except Exception as e:
            logger.exception("Qwen3-rerank 失敗❌，返回原始候選結果: %s", e)
            return candidates[:top_k]


    def _extract_dashscope_rerank_results(self, resp) -> list[dict]:
        """解析 DashScope Qwen3-rerank 返回格式 """
        logger.info("解析 Qwen3-rerank 返回格式...")
        output = getattr(resp, "output", None)

        if output is None:
            return []

        if isinstance(output, dict):
            results = output.get("results", [])
        else:
            results = getattr(output, "results", [])

        parsed = []
        for item in results:
            if isinstance(item, dict):
                index = item.get("index")
                score = item.get("relevance_score")
            else:
                index = getattr(item, "index", None)
                score = getattr(item, "relevance_score", None)

            if index is not None and score is not None:
                parsed.append({
                    "index": int(index),
                    "relevance_score": float(score),
                })

        parsed.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.info("Qwen3-rerank 返回格式 解析完成 ✅")
        return parsed 
           
    def hybrid_rerank_search(self, query: str, vector_results: list, bm25_results: list, k: int) -> list:
        """混合檢索hybrid ➕ 重排序(僞)rerank"""
        rrf_results = self.hybrid_search(vector_results, bm25_results, k * 2)
        if rrf_results:
            candidates = [(item["doc"], item["rrf_score"]) for item in rrf_results]
            hybrid_rerank_results = self._rerank_with_dashscope(query, candidates, k)
            logger.info("hybrid +rerank 檢索成功✅,從 %d 個結果中選擇 %d 個結果", len(candidates),k)
        else: 
            logger.error("hybrid_search 無結果❌, 重排序(僞)失敗, query: %s", query[:30])
            hybrid_rerank_results = []
        return hybrid_rerank_results[:k]

    def hybrid_rerank_retrieve(self, query: str, k: int = 3, filter_expr=None) -> list:
        """
        單問題完整檢索入口：
        Vector + BM25 -> RRF Hybrid -> Rerank。
        """
        self.stores.ensure_connected()

        vector_results, vector_time = self.vector_search(query, k * 2, filter_expr)
        bm25_results, bm25_time = self.bm25_search(query, k * 2)

        if not vector_results and not bm25_results:
            logger.warning("單問題 hybrid_rerank_retrieve 無召回結果❌，query=%s", query[:50])
            return self._local_fallback_search(query, k)

        results = self.hybrid_rerank_search(
            query=query,
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=k,
        )

        logger.info(
            "單問題 hybrid_rerank_retrieve 完成✅: query=%s, vector=%d, bm25=%d, final=%d",
            query[:50],
            len(vector_results),
            len(bm25_results),
            len(results),
        )

        return results
    def _merge_multi_query_results_by_rrf(self, query_results: list, k: int, rrf_k: int = 60) -> list:
        """使用 RRF 融合多個子問題的檢索結果。"""
        doc_map = {}

        for sub_query, results in query_results:
            for rank, (doc, raw_score) in enumerate(results, 1):
                doc_id = doc.metadata.get("doc_id") or self._generate_doc_id(doc)

                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "doc": doc,
                        "fused_score": 0.0,
                        "best_raw_score": raw_score,
                        "matched_queries": [],
                    }

                doc_map[doc_id]["fused_score"] += 1.0 / (rrf_k + rank)

                try:
                    doc_map[doc_id]["best_raw_score"] = max(
                        doc_map[doc_id]["best_raw_score"],
                        raw_score,
                    )
                except TypeError:
                    # 防止 fallback score 是 tuple 時 max 比較失敗
                    doc_map[doc_id]["best_raw_score"] = raw_score

                doc_map[doc_id]["matched_queries"].append({
                    "query": sub_query,
                    "rank": rank,
                    "raw_score": raw_score,
                })

        merged = []

        for item in doc_map.values():
            doc = item["doc"]

            doc.metadata = {
                **doc.metadata,
                "multi_query_matched_queries": item["matched_queries"],
                "multi_query_best_raw_score": item["best_raw_score"],
            }

            merged.append((doc, item["fused_score"]))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]
    
    def multi_query_retrieve(self, query: str, k: int = 3, per_query_k: int | None = None, filter_expr=None) -> list:
        """
        Multi-query retrieval:
        1. 使用 split_user_input 將原始 query 拆成多個子問題
        2. 每個子問題執行 hybrid + rerank
        3. 使用 RRF 融合多個子問題的結果
        4. 返回 top-k
        """
        self.stores.ensure_connected()

        start_time = time.time()

        sub_queries = split_user_input(query)
        logger.info("Multi-query 拆分結果: %s", sub_queries)

        if per_query_k is None:
            per_query_k = max(k * 2, 6)

        # 只有一個子問題時，退化為單問題 hybrid + rerank
        if len(sub_queries) <= 1:
            logger.info("Multi-query 只有一個子問題，退化為單問題 hybrid_rerank_retrieve")
            return self.hybrid_rerank_retrieve(query=query, k=k, filter_expr=filter_expr)

        query_results = []

        for index, sub_query in enumerate(sub_queries, 1):
            logger.info("Multi-query 子問題 %d/%d: %s", index, len(sub_queries), sub_query)

            results = self.hybrid_rerank_retrieve(query=sub_query, k=per_query_k, 
                                                  filter_expr=filter_expr)

            query_results.append((sub_query, results))

            logger.info("Multi-query 子問題 %d 返回 %d 條結果", index, len(results))

        final_results = self._merge_multi_query_results_by_rrf(
            query_results=query_results,
            k=k,
        )

        elapsed = time.time() - start_time
        logger.info(
            "Multi-query retrieval 完成: 原始query=%s, 子問題數=%d, 最終返回=%d, 耗時=%.2fs",
            query[:80],
            len(sub_queries),
            len(final_results),
            elapsed,
        )

        return final_results

    def rewrite_query(self, query: str) -> str:
        """調用llm 重寫query"""
        try:
            prompt_path = Path(__file__).parent.parent / "config" / "prompt.yaml"
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f)

                rewrite_prompt = prompts.get("query_rewrite", {})
                system_prompt = rewrite_prompt.get("system", "")  
                user_prompt = rewrite_prompt.get("user", "").format(query=query)
                full_prompt = system_prompt + "\n\n" + user_prompt

                response = Generation.call(
                    model=self.config.dashscope_model_id,
                    prompt=full_prompt,
                    api_key=self.config.dashscope_api_key,
                    max_tokens=1000
                )
                if response.output and response.output.text:
                    rewritten = response.output.text.strip()
                    if len(rewrite) > 2:
                        logger.info(f"查詢改寫:'{query}' -> '{rewritten}'")
                        return rewritten
                    
        except Exception as e:
            logger.error(f"查詢改寫 失敗❌,使用原查詢: {e} ")
            return query

    def _filter_results_by_intent(self, results: list, intent_id: str | None, 
                                  fallback_to_original: bool = True,) -> list:
        """
        按 intents.yaml 配置的 intent_id 過濾結果
        results 格式: [(doc, score), ...]
        """
        if not intent_id:
            return results

        if intent_id == "D1":
            logger.info("⚠️ intent_id=D1，閒聊意圖跳過知識庫結果")
            return []

        filtered = []
        for doc, score in results:
            metadata = doc.metadata or {}
            if metadata.get("intent") == intent_id or metadata.get("intent_id") == intent_id:
                filtered.append((doc, score))

        if filtered:
            logger.info(
                "Intent post-filter 完成 ✅: intent_id=%s, before=%d, after=%d",
                intent_id, len(results), len(filtered),
            )
            return filtered

        logger.warning(
            "Intent post-filter 後無結果 ❌: intent_id=%s, before=%d",
            intent_id, len(results),
        )

        return results if fallback_to_original else []
    
    def _build_milvus_intent_filter(self, intent_id: str | None) -> str | None:
        """
        構造 Milvus intent 過濾表達式。
        注意：字段名必須與 Milvus collection schema 一致。
        """
        if not intent_id or intent_id == "D1":
            return None

        safe_intent_id = str(intent_id).replace('"', '\\"')
        return f'intent_id == "{safe_intent_id}"'
    
    def intent_filtered_retrieve(self, query: str, intent_id: str | None, k: int = 3,
        filter_expr=None, pre_filter: bool = True, fallback_to_unfiltered=False) -> list:
        """
        Intent-filtered retrieval:
        1. 根據 intent_id 構造向量檢索 filter_expr
        2. 執行 hybrid + rerank
        3. 對最終結果再做 metadata intent 校驗
        4. 如果過濾後無結果，可回退到未過濾結果
        """
        if intent_id == "D1":
            logger.info("⚠️ intent_id=D1，跳過知識庫檢索")
            return []

        self.stores.ensure_connected()

        final_filter_expr = filter_expr

        if pre_filter:
            intent_filter_expr = self._build_milvus_intent_filter(intent_id)
            if intent_filter_expr:
                if final_filter_expr:
                    final_filter_expr = f"({final_filter_expr}) and ({intent_filter_expr})"
                else:
                    final_filter_expr = intent_filter_expr

        logger.info("意圖過濾 檢索 開始 😄: query=%s, intent_id=%s, filter_expr=%s",
                    query[:80], intent_id, final_filter_expr
                    )

        results = self.hybrid_rerank_retrieve(query=query, 
                                              k=max(k * 2, 6), filter_expr=final_filter_expr )

        filtered = self._filter_results_by_intent( results=results, 
                                                  intent_id=intent_id, fallback_to_original=False,)

        if filtered:
            return filtered[:k]

        if fallback_to_unfiltered:
            logger.warning("意圖過濾 檢索 無結果❌，回退未過濾檢索: query=%s, intent_id=%s", 
                           query[:80],intent_id
                           )

            fallback_results = self.hybrid_rerank_retrieve(
                query=query, 
                k=max(k * 2, 6), 
                filter_expr=filter_expr
                )

            fallback_filtered = self._filter_results_by_intent(
                results=fallback_results,
                intent_id=intent_id,
                fallback_to_original=True,
            )

            return fallback_filtered[:k]

        return []
    def _local_fallback_search(self, query: str, k: int) -> list:
        """本地兜底搜索"""
        logger.warning("⚠️ 使用本地兜底搜索（difflib）")
        all_docs = []
        try:
            if self.stores.vector_store:
                all_docs = self.stores.vector_store.similarity_search(query, k=100)
            if not all_docs and self.stores.elasticsearch_store:
                all_docs = self.stores.elasticsearch_store.similarity_search(query, k=100)
        except Exception as e:
            logger.error("获取兜底文檔失敗❌: %s", e)

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

        cache_key = self._get_cache_key(query, k, filter_expr)
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
            logger.info("混合檢索 + Rerank 成功✅，返回 %s 條結果", len(final_results))
        elif vector_results:
            candidates = vector_results[:k * 2]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("使用向量檢索結果 + Rerank✅")
        elif bm25_results:
            candidates = bm25_results[:k * 2]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info("使用BM25檢索結果 + Rerank✅")
        else:
            logger.warning("所有檢索都失敗，尝试使用本地兜底搜索❌✅")
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



    def _get_cache_key(self, query: str, k: int, filter_expr) -> str:
        """生成缓存键"""
        key_data = {"query": query, "k": k, "filter_expr": filter_expr}
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()





