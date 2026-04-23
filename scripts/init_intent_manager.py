
"""
初始化意图管理器脚本
在项目启动时运行，确保IntentManager被正确初始化
"""

import sys
import os
import asyncio
import time

# 添加根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent_manager import get_intent_manager

async def init_intent_manager():
    """初始化意图管理器"""
    try:
        print("开始初始化意图管理器...")
        
        # 获取IntentManager实例
        manager = await get_intent_manager()
        
        # 等待初始化完成
        time.sleep(2)
        
        # 验证初始化结果
        intent_map = manager.get_all_intents()
        print(f"意图管理器初始化成功，共加载 {len(intent_map)} 个意图")
        
        # 测试几个意图匹配
        test_queries = [
            "我想退货",
            "如何查询快递",
            "会员积分怎么查",
            "你好"
        ]
        
        print("\n测试意图匹配:")
        for query in test_queries:
            intent_id, score = await manager.match_intent(query)
            if intent_id:
                info = manager.get_intent_info(intent_id)
                print(f"查询: '{query}' -> 意图: {intent_id} ({info['name']}), 相似度: {score:.4f}")
            else:
                print(f"查询: '{query}' -> 未匹配到意图")
        
        return True
        
    except Exception as e:
        print(f"初始化意图管理器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(init_intent_manager())
