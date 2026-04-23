# 智能客服Agent

基于LangGraph、DeepSeek、LangChain、Milvus和FastAPI的智能客服系统。

## 技术栈

- **LangGraph**：处理复杂的状态机，解决客服中途跳话题的问题
- **DeepSeek**：大语言模型，提供智能对话能力
- **LangChain**：构建语言模型应用的框架
- **Milvus**：向量数据库，用于存储和检索知识库
- **FastAPI**：提供高性能的API接口
- **Uvicorn**：ASGI服务器，运行FastAPI应用
- **LangSmith**：监控和评估Agent的性能

## 功能特点

- 智能对话管理，支持话题切换
- 知识库检索，提供准确的信息
- 状态管理，保持对话上下文
- 高性能API接口
- 实时监控和评估

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置环境变量

在`.env`文件中配置以下参数：

```
deepseek_api_key="your-deepseek-api-key"
deepseek_model_id="deepseek"
deepseek_base_url="https://api.deepseek.com"
```

### 启动Milvus服务

确保Milvus服务正在运行：

### 热启动

```bash
uvicorn app:app --reload
```

## 项目结构

```
.
├── app.py              # FastAPI服务
├── knowledge_base.py   # 知识库管理
├── state_machine.py    # LangGraph状态机
├── init.py             # 初始化脚本
├── requirements.txt    # 依赖配置
├── .env                # 环境变量
└── README.md           # 项目说明
```


## 扩展建议

1. 添加更多的知识库文档
2. 优化提示词，提高回答质量
3. 添加多语言支持
4. 实现更复杂的状态管理逻辑
5. 添加用户身份验证

## 注意事项

- 确保DeepSeek API密钥有效
- 确保Milvus服务正在运行
- 对于生产环境，建议使用更强大的服务器和数据库
- 定期更新知识库，确保信息的准确性