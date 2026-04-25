# 智能客服Agent 功能清单

## 1.知识增强与检索(RAG)
### 1.基于语义的向量检索
1. DashScpoeEmbedding:配置阿里向量模型将本地知识库(like FAQ)转化为向量,确保语义理解精准度
2. Milvus向量数据库:负责海量向量的高效存储与相似度检索
3. 动态阈值路由:利于余弦相似度进行打分,根据score大小选择不同回答模式

## 2.基于langgraph的状态机管理
1. 多节点拆分:
## 3.会话生命周期与长文本
1. 自动对话总结:监听对话轮数.采用摘要压缩法,超过15轮自动压缩,防止llm对话窗口溢出,节省token
2. 
## 4.API与后端
1. uvicorn+FastAPI构建异步ASGI架构,支持高并发流式请求
2. Docker-compose配置启动Milvus服务


## 知识库功能
- 使用DashScopeEmbedding模型进行向量嵌入
- 配置Milvus使用余弦相似度

- 从Excel自动导入FAQ（使用pandas和tqdm进度条）
- Pydantic校验FAQ数据（确保问题和答案存在）
- 知识库检索性能优化（缓存机制、结果去重）

