
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import ElasticsearchStore
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
import time
import json
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

# 创建线程池用于异步处理
thread_pool = ThreadPoolExecutor(max_workers=4)
# 加载环境变量
load_dotenv('.env')
langchain_smith_api_key = os.getenv("langchain_smith_api_key")

dashscope_api_key = os.getenv("dashscope_api_key")
dashscope_model_id = os.getenv("dashscope_model_id")

class KnowledgeBase:
    def __init__(self):
        # 初始化嵌入模型
        self.embeddings = DashScopeEmbeddings(
            model=dashscope_model_id
        )
        
        # 初始化向量存储为None，稍后通过异步方法连接
        self.vector_store = None
        
        # 初始化Elasticsearch存储为None，稍后通过方法连接
        self.elasticsearch_store = None
        
        # 添加缓存
        self.cache = {}
        self.cache_expiry = 3600  # 缓存过期时间（秒）
        
        # 初始化评估数据存储
        self.evaluation_data = []
        self.evaluation_file = "search_evaluation.json"
    
    async def connect(self):
        """异步连接到Milvus向量数据库和Elasticsearch"""
        if self.vector_store is None:
            # 初始化Milvus向量存储
            self.vector_store = Milvus(
                embedding_function=self.embeddings,
                collection_name="customer_service",
                connection_args={
                    "uri": "tcp://localhost:19530"
                },
                index_params={
                    "metric_type": "COSINE"
                },
                auto_id=True
            )
            print("成功连接到Milvus向量数据库")
        
        if self.elasticsearch_store is None:
            # 初始化Elasticsearch存储
            self.elasticsearch_store = ElasticsearchStore(
                es_url="http://localhost:9200",
                index_name="customer_service",
                embedding=self.embeddings
            )
            print("成功连接到Elasticsearch")
        
        return self.vector_store
    
    def ensure_connected(self):
        """确保已连接到Milvus向量数据库和Elasticsearch（同步方法）"""
        if self.vector_store is None:
            try:
                # 同步初始化Milvus向量存储
                self.vector_store = Milvus(
                    embedding_function=self.embeddings,
                    collection_name="customer_service",
                    connection_args={
                        "uri": "tcp://localhost:19530"
                    },
                    index_params={
                        "metric_type": "COSINE"
                    },
                    auto_id=True
                )
                print("成功连接到Milvus向量数据库")
            except Exception as e:
                logger.error(f"连接Milvus向量数据库失败: {str(e)}")
                # 即使Milvus连接失败，也继续尝试连接Elasticsearch
        
        if self.elasticsearch_store is None:
            try:
                # 同步初始化Elasticsearch存储
                self.elasticsearch_store = ElasticsearchStore(
                    es_url="http://localhost:9200",
                    index_name="customer_service",
                    embedding=self.embeddings
                )
                print("成功连接到Elasticsearch")
                # 尝试初始化索引
                self._initialize_elasticsearch_index()
            except Exception as e:
                logger.error(f"连接Elasticsearch失败: {str(e)}")
                # 即使Elasticsearch连接失败，也继续执行
        
        return self.vector_store
    
    def _initialize_elasticsearch_index(self):
        """初始化Elasticsearch索引"""
        try:
            # 尝试获取Elasticsearch客户端
            es_client = self.elasticsearch_store.client
            index_name = "customer_service"
            
            # 检查索引是否存在
            if not es_client.indices.exists(index=index_name):
                # 创建索引，使用默认分析器，向量维度设为1024
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
                                    "dims": 1024
                                }
                            }
                        }
                    }
                )
                logger.info(f"成功创建Elasticsearch索引: {index_name}")
            else:
                logger.info(f"Elasticsearch索引 {index_name} 已存在")
        except Exception as e:
            logger.error(f"初始化Elasticsearch索引失败: {str(e)}")
    
    def add_documents(self, file_paths):
        """添加文档到知识库"""
        # 确保已连接到Milvus和Elasticsearch
        self.ensure_connected()
        
        documents = []
        for file_path in file_paths:
            loader = TextLoader(file_path)
            docs = loader.load()
            documents.extend(docs)
        
        # 分割文档
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        
        # 添加到向量存储
        self.vector_store.add_documents(split_docs)
        
        # 添加到Elasticsearch
        self.elasticsearch_store.add_documents(split_docs)
        
        # 清空缓存
        self.cache = {}
        
        return len(split_docs)
    
    def add_faq(self, faq_items):
        """添加FAQ到知识库"""
        # 确保已连接到Milvus和Elasticsearch
        self.ensure_connected()
        
        documents = []
        for ques, anws in faq_items.items():
            content = f"问题: {ques}\n答案: {anws}"
            #直接使用langchain的document对象,可以把doc.dict()轻松转化为json
            doc=Document(
                page_content=content,
                metadata={"type":"faq","question":ques}
            )
            documents.append(doc)
        
        # 添加到milvus
        self.vector_store.add_documents(documents)
        
        # 添加到Elasticsearch
        self.elasticsearch_store.add_documents(documents)
        
        # 清空缓存
        self.cache = {}
        
        return len(documents)
    
    def search(self, query, k=3, filter_expr=None, evaluate=False, relevant_docs=None):
        """搜索知识库"""
        # 确保已连接到Milvus和Elasticsearch
        self.ensure_connected()
        
        # 检查缓存
        cache_key = f"{query}_{k}_{filter_expr}"
        current_time = time.time()
        
        # 记录开始时间
        start_time = time.time()
        
        from_cache = False
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_expiry:
                from_cache = True
                final_results = cached_result
                response_time = time.time() - start_time
                # 异步记录搜索结果
                self._async_log_search(query, len(final_results), response_time, from_cache)
                # 异步评估搜索结果
                if evaluate and relevant_docs:
                    self._async_evaluate_search(query, final_results, relevant_docs, response_time, from_cache)
                return final_results
        
        # 1. 向量检索
        vector_start = time.time()
        vector_results = []
        try:
            if self.vector_store:
                vector_search_kwargs = {"k": k}
                if filter_expr:
                    vector_search_kwargs["expr"] = filter_expr
                vector_results = self.vector_store.similarity_search_with_score(query, **vector_search_kwargs)
                logger.info(f"向量检索成功，返回 {len(vector_results)} 条结果")
            else:
                logger.warning("向量存储未初始化，跳过向量检索")
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
        vector_time = time.time() - vector_start
        
        # 2. BM25检索
        bm25_start = time.time()
        bm25_results = []
        try:
            if self.elasticsearch_store:
                bm25_results = self.elasticsearch_store.similarity_search_with_score(query, k=k)
                logger.info(f"BM25检索成功，返回 {len(bm25_results)} 条结果")
            else:
                logger.warning("Elasticsearch存储未初始化，跳过BM25检索")
        except Exception as e:
            logger.error(f"BM25检索失败: {str(e)}")
        bm25_time = time.time() - bm25_start
        
        # 3. 合并结果并使用RRF算法打分
        merge_start = time.time()
        # 构建文档到排名的映射
        doc_rank_map = {}
        
        # 处理向量检索结果
        for rank, (doc, score) in enumerate(vector_results, 1):
            if doc.page_content not in doc_rank_map:
                doc_rank_map[doc.page_content] = {"doc": doc, "ranks": []}
            doc_rank_map[doc.page_content]["ranks"].append(rank)
        
        # 处理BM25检索结果
        for rank, (doc, score) in enumerate(bm25_results, 1):
            if doc.page_content not in doc_rank_map:
                doc_rank_map[doc.page_content] = {"doc": doc, "ranks": []}
            doc_rank_map[doc.page_content]["ranks"].append(rank)
        
        # 计算RRF分数
        rrf_k = 60  # RRF常数
        rrf_results = []
        for content, data in doc_rank_map.items():
            rrf_score = sum(1 / (rrf_k + rank) for rank in data["ranks"])
            rrf_results.append((data["doc"], rrf_score))
        
        # 按RRF分数排序
        rrf_results.sort(key=lambda x: x[1], reverse=True)
        
        # 限制结果数量
        final_results = rrf_results[:k]
        
        # 如果没有结果，尝试使用单个检索的结果
        if not final_results:
            if vector_results:
                # 使用向量检索结果
                final_results = vector_results[:k]
                logger.info("使用向量检索结果")
            elif bm25_results:
                # 使用BM25检索结果
                final_results = bm25_results[:k]
                logger.info("使用BM25检索结果")
            else:
                # 两个检索都失败，返回空结果
                final_results = []
                logger.warning("所有检索都失败，返回空结果")
        merge_time = time.time() - merge_start
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 异步记录搜索结果
        self._async_log_search(query, len(final_results), response_time, from_cache, vector_time, bm25_time, merge_time)
        
        # 异步评估搜索结果
        if evaluate and relevant_docs:
            self._async_evaluate_search(query, final_results, relevant_docs, response_time, from_cache)
        else:
            # 即使没有提供相关文档，也要输出基本的评估数据
            self._async_log_basic_evaluation(query, final_results, response_time, from_cache)
        
        # 缓存结果
        self.cache[cache_key] = (final_results, current_time)
        
        return final_results
    """
    # 清空知识库
    def drop_collection(self):
        #危险操作：清空当前集合的所有数据
        from pymilvus import utility
        if utility.has_collection("customer_service"):
            utility.drop_collection("customer_service")
            print("集合已清空，可以重新开始。")
    """
    
    def log_search(self, query, result_count, response_time, from_cache, vector_time=0, bm25_time=0, merge_time=0):
        """记录搜索日志"""
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
        
        # 记录评估结果
        logger.info(f"Evaluation: {json.dumps({
            'query': query,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'hit_rate': hit_rate,
            'response_time': response_time
        })}")
    
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
        
        # 提交到线程池执行
        thread_pool.submit(log_task)
    
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
                
                # 保存评估数据
                self.evaluation_data.append(evaluation_entry)
                self.save_evaluation_data()
                
                # 记录评估结果
                logger.info(f"Evaluation: {json.dumps({
                    'query': query,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'hit_rate': hit_rate,
                    'response_time': response_time
                })}")
            except Exception as e:
                logger.error(f"异步评估搜索结果失败: {str(e)}")
        
        # 提交到线程池执行
        thread_pool.submit(evaluate_task)
    
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
                
                # 记录到日志文件
                logger.info(f"Basic Evaluation: {json.dumps({
                    'query': query,
                    'response_time': response_time,
                    'from_cache': from_cache,
                    'result_count': len(results)
                })}")
            except Exception as e:
                logger.error(f"异步记录基本评估数据失败: {str(e)}")
        
        # 提交到线程池执行
        thread_pool.submit(log_task)

        