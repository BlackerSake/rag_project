# 智能客服Agent（基于RAG技术）

Made by math_King Jumping

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
MILVUS_URI="tcp://localhost:19530"
ES_URL="http://localhost:9200"

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
```bash
python from_mysql_import_faq_to_milvus.py
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
