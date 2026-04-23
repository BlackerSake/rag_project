import os
import sys
import requests
from typing import Dict, Any

# 添加当前目录到 Python 路径，解决相对导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from base import BaseTool


class LogisticsTool(BaseTool):
    """物流查询工具"""
    
    def __init__(self):
        """初始化物流工具"""
        self.api_key = os.getenv('agent_tool_api')
        if not self.api_key:
            raise ValueError("agent_tool_api not found in environment variables")
        self.api_url = "https://uapis.cn/api/v1/misc/tracking/query"
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行物流查询
        
        Args:
            tracking_number: 物流单号
            
        Returns:
            Dict[str, Any]: 物流查询结果
        """
        try:
            # 验证参数
            tracking_number = "435115254779167"
            #tracking_number = kwargs.get("tracking_number")
            if not tracking_number:
                return self.format_result(False, message="物流单号缺失，请提供正确的物流单号")
            
            # 调用物流API
            logistics_data = self._call_logistics_api(tracking_number)
            
            
            """
            # 提炼关键信息
            processed_data = self._process_logistics_data(logistics_data)
            return self.format_result(True, data=processed_data, message="查询成功")
            """
            # 直接返回原始物流数据，不进行精炼
            return self.format_result(True, data=logistics_data, message="查询成功")
            
        except Exception as e:
            return self.format_result(False, message=f"查询失败: {str(e)}")
    
    def _process_logistics_data(self, logistics_data: Dict[str, Any]) -> str:
        """处理物流数据，提炼关键信息
        
        Args:
            logistics_data: 原始物流数据
            
        Returns:
            str: 处理后的物流信息
        """
        # 提取基本信息
        tracking_number = logistics_data.get('tracking_number', '未知')
        carrier = logistics_data.get('carrier_name', logistics_data.get('carrier', '未知'))
        status = logistics_data.get('status', '未知')
        
        # 提取最新轨迹
        latest = ""
        tracks = logistics_data.get('tracks', [])
        if tracks:
            latest_track = tracks[0]  # 假设第一条是最新的
            latest = f"{latest_track.get('time', '未知时间')} - {latest_track.get('location', '未知位置')}: {latest_track.get('description', '无描述')}"
        
        # 构建信息字符串
        info = f"快递单号：{tracking_number}\n"
        info += f"承运商：{carrier}\n"
        info += f"当前状态：{status}\n"
        info += f"最新轨迹：{latest}\n"
        
        # 提取其他可能的信息
        if 'estimated_delivery' in logistics_data:
            info += f"预计送达：{logistics_data['estimated_delivery']}\n"
        if 'current_location' in logistics_data:
            info += f"当前位置：{logistics_data['current_location']}\n"
        
        return info
    
    def validate_params(self, **kwargs) -> bool:
        """验证参数
        
        Args:
            **kwargs: 待验证参数
            
        Returns:
            bool: 参数是否有效
        """
        tracking_number = kwargs.get("tracking_number")
        if not tracking_number:
            return False
        return True
    
    def _call_logistics_api(self, tracking_number: str) -> Dict[str, Any]:
        """调用物流API
        
        Args:
            tracking_number: 物流单号
            
        Returns:
            Dict[str, Any]: 物流查询结果
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        params = {
            "tracking_number": tracking_number
        }
        
        response = requests.get(self.api_url, headers=headers, params=params)
        
        # 处理响应
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            error_data = response.json()
            raise Exception(f"暂无物流信息: {error_data.get('message', '未找到物流数据')}")
        else:
            response.raise_for_status()
            return response.json()

