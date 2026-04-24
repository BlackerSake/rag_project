
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
import os
import uuid
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.documents import Document
import time
import json
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache

# 尝试导入pymilvus
try:
    from pymilvus import connections, utility
except ImportError:
    connections = None
    utility = None
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/search_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

langchain_smith_api_key = os.getenv("LANGCHAIN_SMITH_API_KEY")
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope_model_id = os.getenv("DASHSCOPE_MODEL_ID")

class KnowledgeBase:
    def __init__(self):
        self.embeddings = DashScopeEmbeddings(
            model=dashscope_model_id,
            dashscope_api_key=dashscope_api_key
        )

        self.vector_store = None
        self.elasticsearch_store = None
        
        self._init_lock = threading.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        
        self.evaluation_data = []
        self.evaluation_file = "search_evaluation.json"
        
        self._rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"
        self._rerank_top_k = int(os.getenv("RERANK_TOP_K", "6"))
    
    def _rerank_with_dashscope(self, query: str, candidates: list, top_k: int) -> list:
        if not self._rerank_enabled or not candidates:
            return candidates[:top_k]
        
        try:
            import dashscope
            from dashscope import TextEmbedding
            
            doc_texts = [doc.page_content for doc, score in candidates]
            
            query_embedding = TextEmbedding.call(
                query,
                model=dashscope_model_id,
                api_key=dashscope_api_key
            ).output.embedding
            
            doc_embeddings = TextEmbedding.call(
                [query] + doc_texts,
                model=dashscope_model_id,
                api_key=dashscope_api_key
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
            logger.info(f"Dashscope Rerank 完成，从 {len(candidates)} 条候选中选取 top {top_k} 条")
            return scored[:top_k]
        except Exception as e:
            logger.error(f"Dashscope Rerank 失败: {str(e)}，返回原始候选结果")
            return candidates[:top_k]
    
    def _generate_doc_id(self, doc: Document) -> str:
        content_hash = hash(doc.page_content + json.dumps(doc.metadata, sort_keys=True))
        return f"{uuid.uuid4().hex[:8]}_{content_hash}"
    
    # 移除重复的connect函数，统一使用ensure_connected函数
    
    def ensure_connected(self):
        if self.vector_store is None or self.elasticsearch_store is None:
            with self._init_lock:
                if self.vector_store is None:
                    # ============连接Milvus如果未连接则连接=====================
                    try:
                        if not connections.has_connection("default"):
                            logger.info("未连接Milvus，尝试连接...")
                            connections.connect("default", uri=os.getenv("MILVUS_URI", "tcp://localhost:19530"))
                            logger.info("成功连接到Milvus")
                        else:
                            logger.info("pymilvus 已连接,无需重复连接")
                    except Exception as e:
                        logger.error(f"建立 pymilvus 连接失败: {str(e)}")
                        return None
                    # ============连接Milvus向量数据库=====================
                    max_attempts = 2
                    for attempt in range(max_attempts):
                        try:
                            print(f"尝试连接Milvus (尝试 {attempt + 1}/{max_attempts})...")
                            self.vector_store = Milvus(
                                embedding_function=self.embeddings,
                                collection_name="customer_service",
                                connection_args={
                                    "uri": os.getenv("MILVUS_URI", "tcp://localhost:19530"),
                                    "alias": "default"
                                },
                                index_params={
                                    "metric_type": "COSINE"
                                },
                                auto_id=True
                            )
                            logger.info("成功连接到Milvus向量数据库")
                            break
                        except Exception as e:
                                logger.error(f"连接Milvus向量数据库失败 (尝试 {attempt + 1}/{max_attempts}): {str(e)}")
                                if attempt < max_attempts - 1:
                                    import time
                                    time.sleep(2)  # 等待2秒后重试
                                else:
                                    logger.error("连接Milvus向量数据库次数耗尽,最终失败")
                                    self.vector_store = None
                if self.elasticsearch_store is None:
                    try:
                        self.elasticsearch_store = ElasticsearchStore(
                            es_url=os.getenv("ES_URL", "http://localhost:9200"),
                            index_name="customer_service",
                            embedding=self.embeddings
                        )
                        logger.info("成功连接到Elasticsearch")
                        self._initialize_elasticsearch_index()
                    except Exception as e:
                        logger.error(f"连接Elasticsearch失败: {str(e)}")

        return self.vector_store
    
    def _initialize_elasticsearch_index(self):
        try:
            es_client = self.elasticsearch_store.client
            index_name = "customer_service"
            
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
                logger.info(f"成功创建Elasticsearch索引: {index_name}, 向量维度: {vector_dim}")
            else:
                logger.info(f"Elasticsearch索引 {index_name} 已存在")
        except Exception as e:
            logger.error(f"初始化Elasticsearch索引失败: {str(e)}")
    
    def add_documents(self, file_paths):
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
        self.ensure_connected()
        
        documents = []
        for ques, anws in faq_items.items():
            content = f"问题: {ques}\n答案: {anws}"
            # 确保包含所有必需的字段
            metadata = {
                "type": "faq",
                "question": ques,
                "mysql_id": "",
                "intent_id": "",
                "original_question": ques,
                "action": ""
            }
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            doc.metadata["doc_id"] = self._generate_doc_id(doc)
            documents.append(doc)
        
        try:
            if self.vector_store:
                self.vector_store.add_documents(documents)
                print("成功添加FAQ到Milvus")
        except Exception as e:
            print(f"添加FAQ到Milvus失败: {str(e)}")
        
        try:
            if self.elasticsearch_store:
                self.elasticsearch_store.add_documents(documents)
                print("成功添加FAQ到Elasticsearch")
        except Exception as e:
            print(f"添加FAQ到Elasticsearch失败: {str(e)}")
        
        self.cache.clear()
        
        return len(documents)
    
    def _get_cache_key(self, query: str, k: int, filter_expr) -> str:
        key_data = {"query": query, "k": k, "filter_expr": filter_expr}
        return hash(json.dumps(key_data, sort_keys=True))
    
    def _vector_search(self, query: str, k: int, filter_expr=None):
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
                logger.info(f"向量检索成功，返回 {len(vector_results)} 条结果")
            else:
                logger.warning("向量存储未初始化，跳过向量检索")
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
        return vector_results, vector_time
    
    def _bm25_search(self, query: str, k: int):
        bm25_results = []
        bm25_time = 0
        try:
            if self.elasticsearch_store:
                start = time.time()
                bm25_results = self.elasticsearch_store.similarity_search_with_score(query, k=k)
                bm25_time = time.time() - start
                logger.info(f"BM25检索成功，返回 {len(bm25_results)} 条结果")
            else:
                logger.warning("Elasticsearch存储未初始化，跳过BM25检索")
        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
        return bm25_results, bm25_time
    
    def _local_fallback_search(self, query: str, k: int) -> list:
        logger.warning("使用本地兜底搜索（difflib）")
        all_docs = []
        try:
            if self.vector_store:
                all_docs = self.vector_store.similarity_search(query, k=100)
            if not all_docs and self.elasticsearch_store:
                all_docs = self.elasticsearch_store.similarity_search(query, k=100)
        except Exception as e:
            logger.error(f"获取兜底文档失败: {str(e)}")
        
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
    
    def _merge_results_with_rrf(self, vector_results: list, bm25_results: list, k: int):
        doc_rank_map = {}
        
        for rank, (doc, score) in enumerate(vector_results, 1):
            doc_id = doc.metadata.get("doc_id", self._generate_doc_id(doc))
            if doc_id not in doc_rank_map:
                doc_rank_map[doc_id] = {"doc": doc, "ranks": [], "vector_score": score, "bm25_score": 0}
            doc_rank_map[doc_id]["ranks"].append(rank)
            doc_rank_map[doc_id]["vector_score"] = score
        
        for rank, (doc, score) in enumerate(bm25_results, 1):
            doc_id = doc.metadata.get("doc_id", self._generate_doc_id(doc))
            if doc_id not in doc_rank_map:
                doc_rank_map[doc_id] = {"doc": doc, "ranks": [], "vector_score": 0, "bm25_score": score}
            doc_rank_map[doc_id]["ranks"].append(rank)
            doc_rank_map[doc_id]["bm25_score"] = score
        
        rrf_k = 60
        rrf_results = []
        for doc_id, data in doc_rank_map.items():
            rrf_score = sum(1 / (rrf_k + rank) for rank in data["ranks"])
            combined_score = (data["vector_score"] or 0) + (data["bm25_score"] or 0)
            rrf_results.append({
                "doc": data["doc"],
                "rrf_score": rrf_score,
                "combined_score": combined_score,
                "source": "hybrid"
            })
        
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return rrf_results[:k]
    
    def search(self, query, k=3, filter_expr=None, evaluate=False, relevant_docs=None):
        self.ensure_connected()
        
        cache_key = self._get_cache_key(query, k, filter_expr)
        start_time = time.time()
        from_cache = False
        
        if cache_key in self.cache:
            final_results = self.cache[cache_key]
            from_cache = True
            response_time = time.time() - start_time
            self._async_log_search(query, len(final_results), response_time, from_cache)
            if evaluate and relevant_docs:
                self._async_evaluate_search(query, final_results, relevant_docs, response_time, from_cache)
            return final_results
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self._vector_search, query, k * 2, filter_expr)
            bm25_future = executor.submit(self._bm25_search, query, k * 2)
            
            vector_results, vector_time = vector_future.result()
            bm25_results, bm25_time = bm25_future.result()
        
        rrf_results = self._merge_results_with_rrf(vector_results, bm25_results, k * 2)
        
        if rrf_results:
            candidates = [(r["doc"], r["combined_score"]) for r in rrf_results]
            reranked_candidates = self._rerank_with_dashscope(query, candidates, k * 2)
            final_results = reranked_candidates[:k]
            logger.info(f"混合检索 + Rerank 成功，返回 {len(final_results)} 条结果")
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
        
        if evaluate and relevant_docs:
            self._async_evaluate_search(query, final_results, relevant_docs, response_time, from_cache)
        else:
            self._async_log_basic_evaluation(query, final_results, response_time, from_cache)
        
        self.cache[cache_key] = final_results
        
        return final_results
    
    def evaluate_search(self, query, results, relevant_docs, response_time, from_cache):
        """评估搜索结果"""
        # 计算命中率（至少有一个相关结果）
        hit_rate = 0
        relevant_count = 0
        
        # 计算精确率和召回率
        retrieved_docs = [doc.page_content for doc, score in results]
        
        # 统计相关结果数量
        for doc_content in retrieved_docs:
            if any(relevant in doc_content for relevant in relevant_docs):
                relevant_count += 1
        
        # 计算指标
        precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0
        recall = relevant_count / len(relevant_docs) if relevant_docs else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        hit_rate = 1 if relevant_count > 0 else 0
        
        # 构建评估数据
        evaluation_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "results": [{
                "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "score": score,
                "is_relevant": any(relevant in doc.page_content for relevant in relevant_docs)
            } for doc, score in results],
            "relevant_docs": relevant_docs,
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "hit_rate": hit_rate,
                "response_time": response_time,
                "from_cache": from_cache
            }
        }
        
        # 保存评估数据
        self.evaluation_data.append(evaluation_entry)
        self.save_evaluation_data()
        
        eval_json = json.dumps({
            'query': query,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'hit_rate': hit_rate,
            'response_time': response_time
        })
        logger.info(f"Evaluation: {eval_json}")
    
    def save_evaluation_data(self):
        """保存评估数据到文件"""
        try:
            with open(self.evaluation_file, 'w', encoding='utf-8') as f:
                json.dump(self.evaluation_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存评估数据失败: {str(e)}")
    
    def load_evaluation_data(self):
        """从文件加载评估数据"""
        try:
            if os.path.exists(self.evaluation_file):
                with open(self.evaluation_file, 'r', encoding='utf-8') as f:
                    self.evaluation_data = json.load(f)
        except Exception as e:
            logger.error(f"加载评估数据失败: {str(e)}")
    
    def get_evaluation_summary(self):
        """获取评估摘要"""
        if not self.evaluation_data:
            return {"message": "暂无评估数据"}
        
        # 计算平均指标
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_hit_rate = 0
        total_response_time = 0
        
        for entry in self.evaluation_data:
            metrics = entry.get("metrics", {})
            total_precision += metrics.get("precision", 0)
            total_recall += metrics.get("recall", 0)
            total_f1 += metrics.get("f1_score", 0)
            total_hit_rate += metrics.get("hit_rate", 0)
            total_response_time += metrics.get("response_time", 0)
        
        count = len(self.evaluation_data)
        
        return {
            "total_queries": count,
            "average_precision": total_precision / count if count > 0 else 0,
            "average_recall": total_recall / count if count > 0 else 0,
            "average_f1_score": total_f1 / count if count > 0 else 0,
            "average_hit_rate": total_hit_rate / count if count > 0 else 0,
            "average_response_time": total_response_time / count if count > 0 else 0
        }
    
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
                logger.info(f"Search: {json.dumps(log_entry)}")
            except Exception as e:
                logger.error(f"异步记录搜索日志失败: {str(e)}")
        
        self._thread_pool.submit(log_task)
    
    def _async_evaluate_search(self, query, results, relevant_docs, response_time, from_cache):
        """异步评估搜索结果"""
        def evaluate_task():
            try:
                # 计算命中率（至少有一个相关结果）
                hit_rate = 0
                relevant_count = 0
                
                # 计算精确率和召回率
                retrieved_docs = [doc.page_content for doc, score in results]
                
                # 统计相关结果数量
                for doc_content in retrieved_docs:
                    if any(relevant in doc_content for relevant in relevant_docs):
                        relevant_count += 1
                
                # 计算指标
                precision = relevant_count / len(retrieved_docs) if retrieved_docs else 0
                recall = relevant_count / len(relevant_docs) if relevant_docs else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                hit_rate = 1 if relevant_count > 0 else 0
                
                # 构建评估数据
                evaluation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "results": [{
                        "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        "score": score,
                        "is_relevant": any(relevant in doc.page_content for relevant in relevant_docs)
                    } for doc, score in results],
                    "relevant_docs": relevant_docs,
                    "metrics": {
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "hit_rate": hit_rate,
                        "response_time": response_time,
                        "from_cache": from_cache
                    }
                }
                
                self.save_evaluation_data()
                
                eval_json = json.dumps({
                    'query': query,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'hit_rate': hit_rate,
                    'response_time': response_time
                })
                logger.info(f"Evaluation: {eval_json}")
            except Exception as e:
                logger.error(f"异步评估搜索结果失败: {str(e)}")
        
        # 提交到线程池执行
        self._thread_pool.submit(evaluate_task)
    
    def _async_log_basic_evaluation(self, query, results, response_time, from_cache):
        """异步记录基本评估数据（没有相关文档时）"""
        def log_task():
            try:
                # 构建基本评估数据
                evaluation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "results": [{
                        "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                        "score": score
                    } for doc, score in results],
                    "metrics": {
                        "response_time": response_time,
                        "from_cache": from_cache,
                        "result_count": len(results)
                    }
                }
                
                basic_eval_json = json.dumps({
                    'query': query,
                    'response_time': response_time,
                    'from_cache': from_cache,
                    'result_count': len(results)
                })
                logger.info(f"Basic Evaluation: {basic_eval_json}")
            except Exception as e:
                logger.error(f"异步记录基本评估数据失败: {str(e)}")
        
        self._thread_pool.submit(log_task)

        