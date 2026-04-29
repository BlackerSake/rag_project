# 智能客服Agent（基于RAG技术）

Made by math_King Jumping
### 測試數據只120條
#### rrf_k = 60 & 比例: 1:1 (Qwen3-rerank)
|         方法          | Recall@3 |  MRR   | NDCG@3 | Precision@3 | Hit Rate@3 | Latency (ms) |
|----------------------|----------|--------|--------|-------------|------------|--------------|
| Vector               |  0.8111  | 0.8806 | 0.8176 |   0.3833    |   0.9000   |    503.75    |
| BM25                 |  0.5681  | 0.7000 | 0.5969 |   0.2333    |   0.7000   |     9.70     |
| Hybrid               |  0.8083  | 0.8833 | 0.8156 |   0.3806    |   0.9083   |    399.81    |
| Hybrid + Rerank      |  0.8111  | 0.9083 | 0.8253 |   0.3750    |   0.9250   |   2416.75    |
| Multi-query + Rerank |  0.8139  | 0.9069 | 0.8266 |   0.3778    |   0.9250   |   1254.80    |
| intent_filtered      |  0.8364  | 0.9404 | 0.8540 |   0.3945    |   0.9541   |   763.3969   |
## 快速开始

### 环境要求

- Python 3.10+
- Docker（用于运行Milvus和Elasticsearch）

### 安装依赖

1. **安装依赖**

```bash
pip install -r requirements.txt
```

### 配置环境变量

在项目根目录创建`.env`文件，配置以下参数：

```env
# DeepSeek API配置
deepseek_api_key="your-deepseek-api-key"
deepseek_model_id="deepseek-chat"
deepseek_base_url="https://api.deepseek.com"

# DashScope API配置
dashscope_api_key="your-dashscope-api-key"
dashscope_model_id="text-embedding-v2"

# 服务配置
MILVUS_URI="http://localhost:19530"
MILVUS_COLLECTION_NAME="customer_service"
ES_URL="http://localhost:9200"
ES_INDEX_NAME="customer_service"

# MySQL配置
MYSQL_HOST="127.0.0.1"
MYSQL_PORT="3306"
MYSQL_USER="root"
MYSQL_PASSWORD="your-mysql-password"
MYSQL_DATABASE="customer_service_db"

# 初始化配置
INIT_BATCH_SIZE="100"
INIT_RETRY_ATTEMPTS="3"
INIT_RETRY_INTERVAL_SECONDS="2"
INIT_FORCE_VECTOR_REIMPORT="false"
INIT_SKIP_VECTOR_IMPORT="false"

# 应用配置
APP_HOST="0.0.0.0"
APP_PORT="8000"
```

### 启动前置服务

#### 启动Milvus和Elasticsearch服务

使用Docker Compose启动所需的服务：

```bash
docker compose up -d
```

验证服务是否正常运行：

```bash
docker compose ps
```

#### 停止服务

当不需要服务时，可以停止并移除容器：

```bash
docker compose down
```

### 初始化知识库

1. **准备知识库文档**

在`data`目录下创建`documents`文件夹，并放入需要的文档。

2. **运行初始化脚本**

```bash
python init.py
```

`init.py` 会统一完成 FAQ 数据校验、意图树同步、MySQL FAQ 表同步、知识库连接检查，以及 MySQL FAQ 到 Milvus/Elasticsearch 的导入。兼容脚本仍可使用：

```bash
python scripts/insert_faq.py
python scripts/from_mysql_import_faq_to_milvus.py
python scripts/check_knowledge_base.py
```
### 启动应用

使用Uvicorn启动FastAPI应用：

```bash
uvicorn app:app --reload
```

应用将在`http://localhost:8000`运行。

## 技术栈

| 技术/框架 | 版本 | 用途 |
|----------|------|------|
| LangGraph | latest | 处理复杂的状态机，管理对话流程 |
| DeepSeek | latest | 大语言模型，提供智能对话能力 |
| LangChain | latest | 构建语言模型应用的框架 |
| Milvus | 2.3.0+ | 向量数据库，存储和检索知识库向量 |
| Elasticsearch | 8.0+ | 文本搜索引擎，执行BM25检索 |
| FastAPI | 0.100+ | 提供的API接口 |
| Uvicorn | 0.20+ | ASGI服务器，运行FastAPI应用 |
| DashScope | latest | 提供嵌入模型服务 |

## 项目结构

```
.
├── app.py              # FastAPI应用主文件
├── core/               # 核心模块
│   ├── models.py       # 模型定义
│   ├── nodes.py        # LangGraph节点
│   ├── state.py        # 状态定义
│   └── intent_manager.py # 意图管理
├── data/               # 数据相关
│   ├── knowledge_base.py # 知识库管理
│   └── documents/      # 文档存储
├── config/             # 配置文件
│   └── intents.yaml    # 意图配置
├── init.py             # 初始化脚本
├── requirements.txt    # 依赖配置
├── .env                # 环境变量
├── docker-compose.yml  # Docker Compose配置
└── README.md           # 项目说明
```
