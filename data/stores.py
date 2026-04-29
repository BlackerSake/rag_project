
import time
import logging
import threading
from langchain_milvus import Milvus
from pymilvus import connections, utility, Collection

try:
    from langchain_elasticsearch import ElasticsearchStore
except ImportError:
    from langchain_community.vectorstores import ElasticsearchStore

logger = logging.getLogger(__name__)

class StoreManager:
    """Milvus 和 Elasticsearch 連接和配置管理"""

    def __init__(self, config, embeddings):
        self.config = config
        self.embeddings = embeddings
        self.vector_store = None
        self.elasticsearch_store = None
        self._init_lock = threading.Lock()
        self.initialized = threading.Event()

    def _is_milvus_connection_alive(self) -> bool:
        """檢查 Milvus 連接是否活躍"""
        if connections is None or utility is None:
            logger.warning("pymilvus 未安裝，跳過 Milvus 探活")
            return False

        try:
            if not connections.has_connection(self.config.milvus_alias):
                return False
            utility.list_collections(using=self.config.config._milvus_alias, timeout=3)
            return True
        except Exception as exc:
            logger.warning("Milvus 連接探活失败: %s", exc)
            return False
        
    def _connect_pymilvus(self) -> bool:
        """連接到Milvus"""
        if connections is None:
            logger.warning("pymilvus 未安裝，无法連接 Milvus")
            return False

        for attempt in range(self.config._milvus_connect_retries):
            try:
                if connections.has_connection(self.config.milvus_alias):
                    if self._is_milvus_connection_alive():
                        logger.info("Milvus alias %s 已可用，复用现有連接", self.config.milvus_alias)
                        return True
                    if self.config.vector_store is not None:
                        logger.warning(
                            "Milvus alias %s 探活失败，但 Milvus 对象仍存活，跳過重连以避免破坏业务对象",
                            self.config.milvus_alias,
                        )
                        return False
                logger.info("尝试連接 Milvus: %s (尝试 %s/%s)", self.config.milvus_uri, attempt + 1, self.config._milvus_connect_retries)
                connections.connect(alias=self.config.milvus_alias, uri=self.config.milvus_uri)
                if self._is_milvus_connection_alive():
                    logger.info("成功連接到 Milvus")
                    return True
            except Exception as exc:
                logger.error("建立 pymilvus 連接失败 (尝试 %s/%s): %s", attempt + 1, self.config._milvus_connect_retries, exc)

            if attempt < self.config._milvus_connect_retries - 1:
                time.sleep(min(2 ** attempt, 30))

        logger.error("建立 pymilvus 連接次数耗尽，最终失败")
        return False
    def _init_milvus_vector_store(self) -> None:
        """初始化Milvus向量存储"""
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                logger.info("尝试初始化 Milvus 向量库 (尝试 %s/%s)", attempt + 1, max_attempts)
                self.vector_store = Milvus(
                    embedding_function=self.config.embeddings,
                    collection_name=self.config._milvus_collection,
                    connection_args={
                        "uri": self.config.milvus_uri,
                        "alias": self.config.milvus_alias
                    },
                    index_params={
                        "metric_type": "COSINE"
                    },
                    auto_id=True
                )
                logger.info("成功連接到 Milvus 向量数据库")
                return
            except Exception as exc:
                logger.error("連接 Milvus 向量数据库失败 (尝试 %s/%s): %s", attempt + 1, max_attempts, exc)
                self.config.vector_store = None
                if attempt < max_attempts - 1:
                    time.sleep(min(2 ** attempt, 30))

        logger.error("連接 Milvus 向量数据库次数耗尽，最终失败")

    def _init_elasticsearch_store(self) -> None:
        """初始化Elasticsearch存储"""
        try:
            self.config.elasticsearch_store = ElasticsearchStore(
                es_url=self.config._es_url,
                index_name=self.config.es_index,
                embedding=self.config.embeddings
            )
            logger.info("成功連接到 Elasticsearch")
            self.config._initialize_elasticsearch_index()
        except Exception as exc:
            self.config.elasticsearch_store = None
            logger.error("連接 Elasticsearch 失败: %s", exc)

    def _is_connection_error(self, exc: Exception) -> bool:
            """判断是否为連接错误"""
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
    def ensure_connected(self):
        """确保連接到存储服务"""
        if self.config.vector_store is not None and self.config.elasticsearch_store is not None:
            return self.config.vector_store

        with self.config._init_lock:
            if self._initialized.is_set():
                return self.config.vector_store

            if self.config.vector_store is None:
                if self._is_milvus_connection_alive() or self.config._connect_pymilvus():
                    self.config._init_milvus_vector_store()
                else:
                    logger.error("Milvus 連接不可用，跳過向量库初始化")

            if self.config.elasticsearch_store is None:
                self.config._init_elasticsearch_store()

            if self.config.vector_store is not None and self.config.elasticsearch_store is not None:
                self.config._initialized.set()

        return self.config.vector_store

    def _initialize_elasticsearch_index(self):
        """初始化Elasticsearch索引"""
        try:
            es_client = self.config.elasticsearch_store.client
            index_name = self.config.es_index

            if not es_client.indices.exists(index=index_name):
                test_text = "test_dimension"
                test_embedding = self.config.embeddings.embed_query(test_text)
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
            logger.warning("pymilvus 未安裝，无法檢查  Milvus 集合")
            return summary

        collection_name = collection_name or self.config._milvus_collection
        if not self._is_milvus_connection_alive() and not self.config._connect_pymilvus():
            logger.warning("Milvus 連接不可用，无法檢查 集合: %s", collection_name)
            return summary

        collection = None
        try:
            if utility.has_collection(collection_name, using=self.config.milvus_alias):
                collection = Collection(collection_name, using=self.config.milvus_alias)
                collection.load()
                summary["collection_exists"] = True
                summary["entity_count"] = int(collection.num_entities)
                summary["fields"] = [field.name for field in collection.schema.fields]
        except Exception as e:
            logger.error("檢查  Milvus 集合失败: %s", e)
        finally:
            if collection is not None:
                try:
                    collection.release()
                except Exception as e:
                    logger.warning("释放 Milvus 集合失败: %s", e)

        return summary

    def _get_milvus_collection(self):
        """获取pymilvus原生Collection对象，可作为备用方案。"""
        if Collection is None or connections is None:
            return None
        try:
            if not connections.has_connection(self.config.milvus_alias):
                logger.warning("原生連接不存在，尝试建立...")
                connections.connect(alias=self.config.milvus_alias, uri=self.config.milvus_uri)
            col = Collection(self.config._milvus_collection, using=self.config.milvus_alias)
            col.load()
            return col
        except Exception as e:
            logger.error("获取原生Milvus集合失败: %s", e)
            return None




