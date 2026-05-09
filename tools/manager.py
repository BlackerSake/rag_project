import json
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool
from langchain.agents import create_agent

from utils.logging_config import get_logger

from .logistics import query_logistics


TOOLS: List[BaseTool] = [query_logistics]
TOOL_REGISTRY: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}
logger = get_logger(__name__)


def _preview_payload(payload: Any, max_length: int = 800) -> str:
    """壓縮 ReAct 日誌內容，避免 app.log 單行過長。"""
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        text = str(payload)

    text = " ".join(text.split())
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


class ReActToolLoggingCallback(BaseCallbackHandler):
    """監控 ReAct 模式的 action 與 tool 執行事件，統一寫入 app.log。"""

    def on_agent_action(self, action, **kwargs: Any) -> None:
        logger.info(
            "ReAct action: tool=%s, input=%s, thought=%s, run_id=%s",
            getattr(action, "tool", "unknown"),
            _preview_payload(getattr(action, "tool_input", "")),
            _preview_payload(getattr(action, "log", "")),
            kwargs.get("run_id"),
        )

    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        logger.info(
            "ReAct finish: output=%s, run_id=%s",
            _preview_payload(getattr(finish, "return_values", finish)),
            kwargs.get("run_id"),
        )

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        logger.info(
            "ReAct tool start: name=%s, input=%s, run_id=%s, parent_run_id=%s",
            serialized.get("name", "unknown"),
            _preview_payload(input_str),
            kwargs.get("run_id"),
            kwargs.get("parent_run_id"),
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        logger.info(
            "ReAct tool end: output=%s, run_id=%s, parent_run_id=%s",
            _preview_payload(output),
            kwargs.get("run_id"),
            kwargs.get("parent_run_id"),
        )

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.exception(
            "ReAct tool error: error=%s, run_id=%s, parent_run_id=%s",
            error,
            kwargs.get("run_id"),
            kwargs.get("parent_run_id"),
        )


def get_tools() -> List[BaseTool]:
    """返回可綁定到 LLM 的 LangChain 標準 tools。"""
    return list(TOOLS)


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """按 tool_call name 查找工具。"""
    return TOOL_REGISTRY.get(name)


def invoke_tool_call(tool_call: Dict[str, Any]) -> str:
    """執行單個 LangChain tool_call，並統一返回字串結果。"""
    tool_name = tool_call.get("name", "")
    tool = get_tool_by_name(tool_name)
    if tool is None:
        available = ", ".join(sorted(TOOL_REGISTRY)) or "無"
        return f"未找到工具：{tool_name}。可用工具：{available}"

    args = tool_call.get("args") or {}
    result = tool.invoke(args)
    return str(result)


def create_customer_service_react_agent(
    llm,
    system_prompt: Optional[str] = None,
    enable_logging: bool = True,
):
    """建立 ReAct agent，供需要 ReAct 鏈路的入口使用。"""
    tools = get_tools()
    agent = create_agent(llm, tools=tools, prompt=system_prompt)
    logger.info(
        "建立 ReAct agent: tools=%s, logging=%s, has_system_prompt=%s",
        [tool.name for tool in tools],
        enable_logging,
        bool(system_prompt),
    )

    if not enable_logging:
        return agent

    return agent.with_config({"callbacks": [ReActToolLoggingCallback()]})


async def load_mcp_tools(server_config: Optional[Dict[str, Dict[str, Any]]] = None) -> Iterable[BaseTool]:
    """從 MCP server 載入 tools。

    `langchain_mcp_adapters` 是 MCP 接入時才需要的可選依賴，避免普通客服
    啟動流程因未配置 MCP server 而失敗。
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise ImportError(
            "MCP 接入需要安裝 langchain-mcp-adapters，"
            "請先將該依賴加入環境後再啟用 MCP tools。"
        ) from exc

    config = server_config or {
        "logistics": {
            "command": "python",
            "args": ["tools/mcp_server.py"],
            "transport": "stdio",
        }
    }
    client = MultiServerMCPClient(config)
    return await client.get_tools()
