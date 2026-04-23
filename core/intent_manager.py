import os
import sys
import yaml
import time
import asyncio
import threading
from typing import Dict, Any, List, Optional
from langchain_milvus import Milvus
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from pymilvus import connections, utility

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
load_dotenv('.env')

dashscope_api_key = os.getenv("dashscope_api_key")
dashscope_model_id = os.getenv("dashscope_model_id")

class IntentManager:
    def __init__(self, yaml_path="config/intents.yaml", collection_name="intent_index"):
        """
        初始化意图管理器
        
        Args:
            yaml_path: 意图树YAML文件路径
            collection_name: Milvus集合名称
        """
        self.yaml_path = yaml_path
        self.collection_name = collection_name
        self.intent_tree = {}
        self.intent_map = {}
        self.last_modified = 0
        self.embeddings = DashScopeEmbeddings(model=dashscope_model_id)
        self.vector_store = None
        self._initialized = False
        
    async def initialize(self):
        """异步初始化"""
        if self._initialized:
            return
            
        try:
            # 初始化Milvus连接
            await self._init_milvus_async()
            
            # 加载YAML文件
            await self._load_yaml_async()
            
            # 启动文件监测器
            self._start_watcher()
            
            self._initialized = True
            print("IntentManager初始化完成")
            
        except Exception as e:
            print(f"IntentManager初始化失败: {str(e)}")
            raise
    
    async def _init_milvus_async(self):
        """异步初始化Milvus向量存储"""
        try:
            # 测试Milvus连接
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    connections.connect("default", uri="tcp://localhost:19530")
                    print(f"Milvus连接成功 (尝试 {attempt + 1})")
                    break
                except Exception as e:
                    print(f"Milvus连接尝试 {attempt + 1} 失败: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2)
            
            # 创建向量存储
            loop = asyncio.get_event_loop()
            self.vector_store = await loop.run_in_executor(
                None,
                lambda: Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={
                        "uri": "tcp://localhost:19530"
                    },
                    index_params={
                        "metric_type": "COSINE"
                    },
                    auto_id=True
                )
            )
            print("Milvus向量存储初始化成功")
            
        except Exception as e:
            print(f"Milvus初始化失败: {str(e)}")
            raise
    
    async def _load_yaml_async(self):
        """异步加载YAML文件"""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self.intent_tree = data.get('intents', {})
            self.intent_map = {}
            
            # 构建意图映射
            self._build_intent_map(self.intent_tree)
            
            # 生成意图向量并存储到Milvus
            await self._generate_intent_vectors_async()
            
            # 更新最后修改时间
            self.last_modified = os.path.getmtime(self.yaml_path)
            print(f"成功加载意图树，共加载 {len(self.intent_map)} 个意图")
            
        except Exception as e:
            print(f"加载YAML文件失败: {str(e)}")
            raise
    
    def _build_intent_map(self, intent_tree, parent_path=""):
        """构建意图映射"""
        for key, value in intent_tree.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            
            if isinstance(value, dict) and 'children' in value:
                self._build_intent_map(value['children'], current_path)
            
            if isinstance(value, dict) and 'intent_id' in value:
                self.intent_map[value['intent_id']] = {
                    'name': value['name'],
                    'description': value['description'],
                    'path': current_path
                }
    
    async def _generate_intent_vectors_async(self):
        """异步生成意图向量并存储到Milvus"""
        try:
            documents = []
            for intent_id, info in self.intent_map.items():
                # 拼接label+description
                content = f"{info['name']}: {info['description']}"
                doc = Document(
                    page_content=content,
                    metadata={
                        'intent_id': intent_id,
                        'name': info['name'],
                        'path': info['path']
                    }
                )
                documents.append(doc)
            
            if documents and self.vector_store:
                # 清空现有数据并重新添加
                loop = asyncio.get_event_loop()
                
                # 检查并删除现有集合
                def drop_collection():
                    try:
                        if utility.has_collection(self.collection_name):
                            utility.drop_collection(self.collection_name)
                            print(f"已删除现有集合: {self.collection_name}")
                    except Exception as e:
                        print(f"删除集合时出错: {str(e)}")
                
                await loop.run_in_executor(None, drop_collection)
                
                # 添加新文档
                await loop.run_in_executor(None, self.vector_store.add_documents, documents)
                print(f"成功生成并存储 {len(documents)} 个意图向量")
                
        except Exception as e:
            print(f"生成意图向量失败: {str(e)}")
            raise
    
    def _start_watcher(self):
        """启动文件变化监测器"""
        def watcher():
            while True:
                try:
                    current_modified = os.path.getmtime(self.yaml_path)
                    if current_modified > self.last_modified:
                        print("检测到YAML文件变化，重新加载...")
                        # 使用asyncio.run运行异步加载
                        asyncio.run(self._load_yaml_async())
                    time.sleep(5)  # 每5秒检查一次
                except Exception as e:
                    print(f"文件监测器错误: {str(e)}")
                    time.sleep(5)
        
        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        print("文件变化监测器已启动")
    
    async def match_intent(self, query, k=3):
        """
        异步匹配用户查询的意图
        
        Args:
            query: 用户查询文本
            k: 返回前k个匹配结果
            
        Returns:
            匹配度最高的intent_id和相似度分数
        """
        try:
            if not self.vector_store:
                print("向量存储未初始化")
                return None, 0.0
            
            # 异步搜索意图向量
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self.vector_store.similarity_search_with_score,
                query,
                k
            )
            
            if results:
                # 获取匹配度最高的结果
                best_result = max(results, key=lambda x: x[1])
                intent_id = best_result[0].metadata.get('intent_id')
                score = best_result[1]
                
                print(f"匹配意图: {intent_id} (相似度: {score:.4f})")
                return intent_id, score
            
            return None, 0.0
            
        except Exception as e:
            print(f"匹配意图失败: {str(e)}")
            return None, 0.0
    
    def get_intent_info(self, intent_id):
        """
        获取意图信息
        
        Args:
            intent_id: 意图ID
            
        Returns:
            意图信息字典
        """
        return self.intent_map.get(intent_id, None)
    
    def get_all_intents(self):
        """
        获取所有意图
        
        Returns:
            意图映射字典
        """
        return self.intent_map
    
    async def detect_orphan_nodes(self, mysql_config):
        """
        异步检测孤儿节点：查询FAQ表中的所有intent_id是否都存在于intents.yaml中
        
        Args:
            mysql_config: MySQL连接配置
            
        Returns:
            dict: 包含孤儿节点信息的字典
        """
        try:
            import pymysql
            
            # 异步连接MySQL
            loop = asyncio.get_event_loop()
            
            def connect_mysql():
                return pymysql.connect(
                    host=mysql_config['host'],
                    user=mysql_config['user'],
                    password=mysql_config['password'],
                    database=mysql_config['database'],
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor
                )
            
            connection = await loop.run_in_executor(None, connect_mysql)
            
            try:
                with connection.cursor() as cursor:
                    # 查询FAQ表中的所有intent_id
                    await loop.run_in_executor(None, cursor.execute, "SELECT DISTINCT intent_id FROM faq")
                    rows = await loop.run_in_executor(None, cursor.fetchall)
                    faq_intent_ids = [row['intent_id'] for row in rows]
                    
                    # 获取YAML中的所有intent_id
                    yaml_intent_ids = list(self.intent_map.keys())
                    
                    # 检测孤儿节点
                    orphan_nodes = [intent_id for intent_id in faq_intent_ids if intent_id not in yaml_intent_ids]
                    
                    print(f"FAQ表中共有 {len(faq_intent_ids)} 个不同的intent_id")
                    print(f"YAML文件中共有 {len(yaml_intent_ids)} 个不同的intent_id")
                    print(f"检测到 {len(orphan_nodes)} 个孤儿节点:")
                    for intent_id in orphan_nodes:
                        print(f"  - {intent_id}")
                    
                    return {
                        'total_faq_intents': len(faq_intent_ids),
                        'total_yaml_intents': len(yaml_intent_ids),
                        'orphan_nodes': orphan_nodes,
                        'has_orphans': len(orphan_nodes) > 0
                    }
            finally:
                connection.close()
                
        except Exception as e:
            print(f"检测孤儿节点失败: {str(e)}")
            return {
                'error': str(e),
                'has_orphans': False
            }

# 全局IntentManager实例
intent_manager = None

async def get_intent_manager():
    """
    异步获取IntentManager实例
    
    Returns:
        IntentManager实例
    """
    global intent_manager
    if intent_manager is None:
        intent_manager = IntentManager()
        await intent_manager.initialize()
    return intent_manager

if __name__ == "__main__":
    # 测试IntentManager
    async def test():
        manager = await get_intent_manager()
        
        # 测试意图匹配
        test_queries = [
            "我想退货",
            "如何查询快递",
            "会员积分怎么查",
            "你好"
        ]
        
        for query in test_queries:
            intent_id, score = await manager.match_intent(query)
            if intent_id:
                info = manager.get_intent_info(intent_id)
                print(f"查询: '{query}' -> 意图: {intent_id} ({info['name']}), 相似度: {score:.4f}")
            else:
                print(f"查询: '{query}' -> 未匹配到意图")
    
    asyncio.run(test())
