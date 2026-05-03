from langchain_core.tools import tool


@tool
def query_logistics(tracking_number: str) -> str:
    """查詢物流單號的快遞狀態與最新軌跡。用戶詢問快遞進度時調用。"""
    normalized_tracking_number = tracking_number.strip()
    if not normalized_tracking_number:
        return "物流單號缺失，請用戶提供需要查詢的快遞單號。"

    # Mock 工具：不連接真實物流 API，保持 tool calling 鏈路可重現。
    return (
        f"物流單號：{normalized_tracking_number}\n"
        "承運商：順豐速運\n"
        "當前狀態：運輸中\n"
        "最新軌跡：2026-05-03 14:20，包裹已從上海分撥中心發出，"
        "正在前往廣州轉運中心。\n"
        "預計送達：2026-05-04 18:00 前\n"
        "客服建議：若明晚仍未更新軌跡，可協助登記催件。"
    )
