import os
from dotenv import load_dotenv
from data.knowledge_base import KnowledgeBase

# 加载环境变量
load_dotenv('.env')

# 设置LangSmith环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("langchain_smith_api_key")

os.environ["LANGCHAIN_PROJECT"] = "智能客服Agent"

# 初始化知识库
kb = KnowledgeBase()

print("初始化完成！")

