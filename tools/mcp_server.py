
from .logistics import query_logistics as query_logistics_tool

from mcp.server.fastmcp import FastMCP



mcp = FastMCP("customer-service-logistics")


@mcp.tool()
def query_logistics(tracking_number: str) -> str:
    """查詢物流單號的快遞狀態與最新軌跡。"""
    return query_logistics_tool.invoke({"tracking_number": tracking_number})


if __name__ == "__main__":
    mcp.run(transport="stdio")
