from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseTool(ABC):
    """工具基类，定义统一接口"""
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具逻辑
        
        Args:
            **kwargs: 工具执行所需参数
            
        Returns:
            Dict[str, Any]: 执行结果，包含 success、data、message 字段
        """
        pass
    
    def validate_params(self, **kwargs) -> bool:
        """验证参数
        
        Args:
            **kwargs: 待验证参数
            
        Returns:
            bool: 参数是否有效
        """
        return True
    
    def format_result(self, success: bool, data: Optional[Any] = None, message: str = "") -> Dict[str, Any]:
        """格式化返回结果
        
        Args:
            success: 是否执行成功
            data: 执行结果数据
            message: 执行消息
            
        Returns:
            Dict[str, Any]: 格式化后的结果
        """
        return {
            "success": success,
            "data": data,
            "message": message
        }
