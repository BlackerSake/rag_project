from pydantic import BaseModel
from typing import Optional

# 从PromptLoader加载prompt模板
from utils.prompt_loader import prompt_loader

# 定义prompt模板变量
topic_detection_prompt = None
direct_answer_prompt = None
clarify_question_prompt = None
fallback_response_prompt = None
summarization_prompt = None
chat_response_prompt = None

# 初始化prompt模板
def init_prompts():
    """初始化prompt模板"""
    global direct_answer_prompt, clarify_question_prompt, fallback_response_prompt, summarization_prompt, chat_response_prompt
    
    print("正在初始化prompt模板...")
    direct_answer_prompt = prompt_loader.get_prompt('direct_answer')
    clarify_question_prompt = prompt_loader.get_prompt('clarify_question')
    fallback_response_prompt = prompt_loader.get_prompt('fallback_response')
    summarization_prompt = prompt_loader.get_prompt('summarization')
    chat_response_prompt = prompt_loader.get_prompt('chat_response')
    
    print(f"direct_answer_prompt: {direct_answer_prompt is not None}")
    print(f"clarify_question_prompt: {clarify_question_prompt is not None}")
    print(f"fallback_response_prompt: {fallback_response_prompt is not None}")
    print(f"summarization_prompt: {summarization_prompt is not None}")
    print(f"chat_response_prompt: {chat_response_prompt is not None}")

# 初始化prompt模板
init_prompts()

# 定义聊天请求模型
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

# 定义聊天响应模型
class ChatResponse(BaseModel):
    response: str
    topic: str
    thread_id: str
