f"""print()
决策引擎基类 - 所有具体决策引擎的抽象接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class DecisionEngine(ABC):
    """决策引擎抽象基类"""
    
    @abstractmethod
    async def make_decision(self, decision_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        抽象决策方法，具体实现由子类提供
        
        Args:
            decision_request: 决策请求数据
            
        Returns:
            决策结果
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取决策引擎状态
        
        Returns:
            引擎状态信息
        """
        return {
            "status": "operational",
            "type": self.__class__.__name__
        }
