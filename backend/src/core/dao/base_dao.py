"""基础DAO类
提供通用的数据库操作方法
"""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseDAO(ABC):
    """基础DAO抽象类"""
    
    def __init__(self, model_class: type = None):
        """初始化DAO"""
        self.model_class = model_class
        self._data_store = {}
        logger.info(f"BaseDAO初始化，模型类: {model_class.__name__ if model_class else 'None'}")
    
    @abstractmethod
    def get_by_id(self, id: Any) -> Optional[Dict[str, Any]]:
        """根据ID获取数据"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Dict[str, Any]]:
        """获取所有数据"""
        pass
    
    @abstractmethod
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建数据"""
        pass
    
    @abstractmethod
    def update(self, id: Any, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新数据"""
        pass
    
    @abstractmethod
    def delete(self, id: Any) -> bool:
        """删除数据"""
        pass
    
    def _generate_id(self) -> int:
        """生成唯一ID"""
        if not self._data_store:
            return 1
        return max(self._data_store.keys()) + 1