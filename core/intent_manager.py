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
from pymilvus import Collection, MilvusClient, connections, utility

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

# 加载环境变量
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
dashscope_model_id = os.getenv("DASHSCOPE_MODEL_ID")
INTENT_MILVUS_ALIAS = "intent_manager_kb"

class IntentManager:
    def __init__(self, yaml_path="config/intents.yaml", collection_name=None):
        """
        初始化意图管理器
        
        Args:
            yaml_path: 意图树YAML文件路径
            collection_name: Milvus集合名称
        """
        self.yaml_path = yaml_path
        self.collection_name = collection_name or os.getenv("INTENT_MILVUS_COLLECTION_NAME", "intent_index")
        self.intent_tree = {}
        self.intent_map = {}
        self.last_modified = 0
        self.embeddings = DashScopeEmbeddings(
            model=dashscope_model_id,
            dashscope_api_key=dashscope_api_key
        )
        self.vector_store = None
        self._initialized = False

    def _get_milvus_connection_args(self):
        raw_uri = os.getenv("MILVUS_URI", "http://localhost:19530")
        milvus_uri = raw_uri.strip()

        if milvus_uri.startswith("tcp://"):
            milvus_uri = f"http://{milvus_uri[len('tcp://'):]}"

        return {"uri": milvus_uri}

    def _ensure_milvus_orm_connection(self, connection_args):
        try:
            alias = MilvusClient(**connection_args)._using
            if not connections.has_connection(alias):
                connections.connect(alias=alias, **connection_args)
                print(f"Milvus ORM连接已建立: {alias} -> {connection_args['uri']}")
            return alias
        except Exception as e:
            print(f"建立Milvus ORM连接失败，将继续使用MilvusClient: {str(e)}")
            return None

    def _create_vector_store(self, connection_args):
        return Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args=connection_args,
            index_params={
                "metric_type": "COSINE"
            },
            auto_id=True
        )

    def _drop_intent_collection(self, active_alias):
        if not active_alias:
            return

        try:
            if utility.has_collection(self.collection_name, using=active_alias):
                utility.drop_collection(self.collection_name, using=active_alias)
                print(f"已删除现有集合: {self.collection_name}")
        except Exception as e:
            print(f"删除集合时出错: {str(e)}")

    def _refresh_vector_store_schema_cache(self, alias=None):
        if self.vector_store is None:
            return

        active_alias = alias or getattr(self.vector_store, "alias", None)
        if not active_alias:
            return

        try:
            self.vector_store.alias = active_alias
            self.vector_store.fields = []
            self.vector_store.col = None

            if utility.has_collection(self.collection_name, using=active_alias):
                collection = Collection(self.collection_name, using=active_alias)
                self.vector_store.col = collection
                self.vector_store.fields = [field.name for field in collection.schema.fields]
                print(f"已刷新意图Milvus字段缓存: {', '.join(self.vector_store.fields)}")
        except Exception as e:
            print(f"刷新意图Milvus字段缓存失败: {str(e)}")

    def _client_search_intents(self, query, k=3):
        if not self.vector_store:
            return []

        search_params = self.vector_store.search_params
        if isinstance(search_params, list):
            search_params = search_params[0] if search_params else None

        embedding = self.embeddings.embed_query(query)
        raw_results = self.vector_store.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field=self.vector_store._vector_field,
            search_params=search_params,
            limit=k,
            output_fields=["*"],
            timeout=self.vector_store.timeout
        )

        parsed_results = []
        for result in raw_results[0]:
            parsed_results.append((result.get("entity", {}), result.get("distance", 0.0)))

        return parsed_results
        
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
            connection_args = self._get_milvus_connection_args()
            milvus_alias = self._ensure_milvus_orm_connection(connection_args)
            
            # 创建向量存储
            loop = asyncio.get_event_loop()
            self.vector_store = await loop.run_in_executor(
                None,
                self._create_vector_store,
                connection_args
            )
            self._refresh_vector_store_schema_cache(alias=milvus_alias)
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
                connection_args = self._get_milvus_connection_args()
                active_alias = getattr(self.vector_store, "alias", INTENT_MILVUS_ALIAS)

                # langchain_milvus 的 drop_old 路径在部分版本中会触发 AsyncMilvusClient
                # 未 await 的 RuntimeWarning，因此这里使用 pymilvus 同步删除后重建。
                await loop.run_in_executor(None, self._drop_intent_collection, active_alias)
                self.vector_store = await loop.run_in_executor(
                    None,
                    self._create_vector_store,
                    connection_args
                )
                
                # 添加新文档
                await loop.run_in_executor(None, self.vector_store.add_documents, documents)
                self._refresh_vector_store_schema_cache(alias=active_alias)
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
                self._client_search_intents,
                query,
                k
            )
            
            if results:
                # 获取匹配度最高的结果
                best_result = max(results, key=lambda x: x[1])
                intent_id = best_result[0].get('intent_id')
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
                    port=int(mysql_config.get('port', 3306)),
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
                    await loop.run_in_executor(None, cursor.execute, "SELECT DISTINCT intent FROM faq")
                    rows = await loop.run_in_executor(None, cursor.fetchall)
                    faq_intent_ids = [row['intent'] for row in rows]
                    
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
