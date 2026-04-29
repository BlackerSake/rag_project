import os
from dataclasses import dataclass

@dataclass(frozen = True)
class KBConfig:
    """知識庫類,負責從環境變量中讀取配置參數"""
    
    # dashscope配置
    dashscope_api_key:str | None
    dashscope_model_id:str | None

    # milvus配置
    milvus_uri:str
    milvus_collection:str
    milvus_alias:str
    milvus_connect_retries:int

    # elasticsearch配置
    es_url:str
    es_index:str

    # cache配置
    cache_maxsize:int
    cache_ttl:int

    # rerank配置
    rerank_enabled:bool
    rerank_top_k:int

    # evaluation配置
    retrieval_eval_enabled:bool
    retrieval_eval_top_k:int

    @classmethod
    def from_env(cls) -> "KBConfig":
        
        """從環境變量中創建KBconfig實例"""
        raw_uri = os.getenv("MILVUS_URI", "http://localhost:19530").strip()
        milvus_uri = f"http://{raw_uri[len('tcp://'):]}" if raw_uri.startswith("tcp://") else raw_uri
        
        return cls(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            dashscope_model_id=os.getenv("DASHSCOPE_MODEL_ID"),
            
            milvus_uri=milvus_uri,
            milvus_collection=os.getenv("MILVUS_COLLECTION_NAME", "customer_service"),
            milvus_alias="default",
            milvus_connect_retries=int(os.getenv("MILVUS_CONNECT_RETRIES", "3")),
            
            es_url=os.getenv("ES_URL", "http://localhost:9200"),
            es_index=os.getenv("ES_INDEX_NAME", "customer_service"),
            
            cache_maxsize=int(os.getenv("CACHE_MAXSIZE", "1000")),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            
            rerank_enabled=os.getenv("RERANK_ENABLED", "false").lower() == "true",
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "6")),
            retrieval_eval_enabled=os.getenv("RETRIEVAL_EVAL_ENABLED", "true").lower() == "true",
            retrieval_eval_top_k=int(os.getenv("RETRIEVAL_EVAL_TOP_K", "5")),
        )
