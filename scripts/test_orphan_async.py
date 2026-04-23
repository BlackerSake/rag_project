
"""
异步孤儿节点检测测试
"""

import sys
import io
import os
import asyncio
from typing import Dict, Any, List

# 设置编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')

# 添加根目录到Python路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

from core.intent_manager import get_intent_manager

class AsyncOrphanDetector:
    """异步孤儿节点检测器"""
    
    def __init__(self, mysql_config: Dict[str, str]):
        """
        初始化检测器
        
        Args:
            mysql_config: MySQL连接配置
        """
        self.mysql_config = mysql_config
        self.intent_manager = None
    
    async def initialize(self):
        """异步初始化"""
        if self.intent_manager is None:
            self.intent_manager = await get_intent_manager()
    
    async def detect_orphans(self) -> Dict[str, Any]:
        """
        异步检测孤儿节点
        
        Returns:
            检测结果字典
        """
        try:
            # 确保初始化完成
            if self.intent_manager is None:
                await self.initialize()
            
            # 运行异步的检测方法
            result = await self.intent_manager.detect_orphan_nodes(self.mysql_config)
            return result
        except Exception as e:
            return {
                'error': str(e),
                'has_orphans': False
            }

async def main():
    """主函数"""
    print("开始异步孤儿节点检测...")
    
    # MySQL连接配置
    mysql_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '133466',
        'database': 'customer_agent'
    }
    
    # 初始化检测器
    detector = AsyncOrphanDetector(mysql_config)
    
    # 执行检测
    result = await detector.detect_orphans()
    
    # 显示结果
    if 'error' in result:
        print(f"错误: {result['error']}")
    else:
        print(f"\n检测结果:")
        print(f"FAQ表中的intent_id数量: {result['total_faq_intents']}")
        print(f"YAML文件中的intent_id数量: {result['total_yaml_intents']}")
        print(f"是否存在孤儿节点: {'是' if result['has_orphans'] else '否'}")
        
        if result['orphan_nodes']:
            print(f"\n孤儿节点列表:")
            for intent_id in result['orphan_nodes']:
                print(f"  - {intent_id}")
        else:
            print("\n没有检测到孤儿节点，所有intent_id都在YAML文件中定义")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
