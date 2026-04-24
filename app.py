import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Any

# 先初始化 FastAPI
app = FastAPI(
    title="智能客服Agent API",
    description="基于LangGraph和DeepSeek的智能客服系统",
    version="1.0.0"
)

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 定义请求和响应模型
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    topic: str
    thread_id: str

# 初始化模板
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# 然后导入其他模块
import uvicorn
from utils.logging_config import get_logger
from state_machine import run_chat

# 获取日志记录器
logger = get_logger(__name__)

print(f"Templates directory: {templates_dir}")
print(f"Templates directory exists: {os.path.exists(templates_dir)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 直接读取并返回模板文件内容
    template_path = os.path.join(templates_dir, "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    return HTMLResponse(content=template_content)

# 聊天接口
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"收到聊天请求: message='{request.message}', thread_id='{request.thread_id}'")
        result = await run_chat(request.message, request.thread_id)
        logger.info(f"返回聊天响应: response='{result['response']}', topic='{result['topic']}', thread_id='{result['thread_id']}'")
        return ChatResponse(
            response=result["response"],
            topic=result["topic"],
            thread_id=result["thread_id"]
        )
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 流式聊天接口
from fastapi.responses import StreamingResponse
import asyncio

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest): # 改为 async
    try:
        logger.info(f"收到流式聊天请求: thread_id='{request.thread_id}'")
        
        from state_machine import compiled_graph # 确保导出已编译的图
        from langchain_core.messages import HumanMessage

        async def event_generator():
            try:
                # 使用 astream_events 获取事件流
                async for event in compiled_graph.astream_events(
                    {"messages": [HumanMessage(content=request.message)]},
                    config={"configurable": {"thread_id": request.thread_id}},
                    version="v2"
                ):
                    # 处理自定义事件（来自stream_multi_response）
                    if event["event"] == "on_custom_event":
                        data = event["data"]
                        if data:
                            logger.info(f"收到自定义事件: {data[:50]}...")
                            yield f"data: {data}\n\n"
                    # 处理聊天模型流式输出
                    elif event["event"] == "on_chat_model_stream":
                        node = event.get("metadata", {}).get("langgraph_node")
                        #白名单：只允许直接回答、澄清提问、兜底回复和闲聊回复节点的输出
                        if node in ["direct_answer", "clarify_question", "fallback_response", "chat_response"]:
                            content = event["data"]["chunk"].content
                            if content:
                                yield f"data: {content}\n\n"
                
                # 发送结束标记
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                logger.info(f"流式连接被取消: thread_id='{request.thread_id}'")
                # 客户端断开连接，优雅地处理取消操作
                return
            except Exception as e:
                logger.error(f"流式生成异常: {str(e)}")
                yield f"data: 抱歉，流式传输出现异常，请稍后再试。\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"流式异常: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)