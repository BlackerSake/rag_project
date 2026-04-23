import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logging_config import get_logger

# 加载环境变量
load_dotenv('.env')

# 获取日志记录器
logger = get_logger(__name__)

# 从环境变量获取配置
api_key = os.getenv("deepseek_api_key")
base_url = os.getenv("deepseek_base_url")
model_id = os.getenv("deepseek_model_id")

# 初始化模型
model = ChatOpenAI(
    model=model_id,
    api_key=api_key,
    base_url=base_url,
    streaming=True,
    temperature=0.5,
    timeout=30
)

logger.info(f"模型初始化完成: {model_id}")