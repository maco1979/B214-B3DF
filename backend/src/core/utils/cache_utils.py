"""
缓存工具模块
提供统一的缓存管理功能
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from functools import wraps


class CacheItem:
    """缓存项"""
    
    def __init__(self, value: Any, expires_at: Optional[float] = None):
        """初始化缓存项
        
        Args:
            value: 缓存值
            expires_at: 过期时间戳（秒），None表示永不过期
        """
        self.value = value
        self.expires_at = expires_at
        self.created_at = time.time()
    
    def is_expired(self) -> bool:
        """检查缓存项是否过期
        
        Returns:
            bool: 是否过期
        """
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def ttl(self) -> Optional[float]:
        """获取剩余过期时间
        
        Returns:
            Optional[float]: 剩余过期时间（秒），None表示永不过期
        """
        if self.expires_at is None:
            return None
        return max(0, self.expires_at - time.time())


class CacheManager:
    """缓存管理器"""
    
    def __init__(self, default_ttl: int = 300):
        """初始化缓存管理器
        
        Args:
            default_ttl: 默认过期时间（秒），默认为5分钟
        """
        self._cache: Dict[str, CacheItem] = {}
        self._default_ttl = default_ttl
        self._lock = threading.RLock()  # 可重入锁，支持嵌套调用
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值
        
        Args:
            key: 缓存键
        
        Returns:
            Optional[Any]: 缓存值，若不存在或已过期则返回None
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            item = self._cache[key]
            if item.is_expired():
                del self._cache[key]
                return None
            
            return item.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），None表示使用默认值，0表示永不过期
        """
        with self._lock:
            expires_at = None
            if ttl is not None:
                if ttl > 0:
                    expires_at = time.time() + ttl
            else:
                # 使用默认过期时间
                expires_at = time.time() + self._default_ttl
            
            self._cache[key] = CacheItem(value, expires_at)
    
    def delete(self, key: str) -> None:
        """删除缓存
        
        Args:
            key: 缓存键
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
    
    def exists(self, key: str) -> bool:
        """检查缓存是否存在
        
        Args:
            key: 缓存键
        
        Returns:
            bool: 是否存在且未过期
        """
        return self.get(key) is not None
    
    def get_or_set(self, key: str, func: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """获取缓存值，若不存在则执行func并缓存结果
        
        Args:
            key: 缓存键
            func: 生成缓存值的函数
            ttl: 过期时间（秒）
        
        Returns:
            Any: 缓存值
        """
        value = self.get(key)
        if value is not None:
            return value
        
        value = func()
        self.set(key, value, ttl)
        return value
    
    def size(self) -> int:
        """获取缓存大小
        
        Returns:
            int: 缓存项数量
        """
        with self._lock:
            # 清理过期项
            self._clean_expired()
            return len(self._cache)
    
    def _clean_expired(self) -> None:
        """清理过期的缓存项"""
        expired_keys = []
        for key, item in self._cache.items():
            if item.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            # 清理过期项
            self._clean_expired()
            
            total_size = len(self._cache)
            current_time = time.time()
            
            # 计算平均TTL
            total_ttl = 0
            expiring_items = 0
            for item in self._cache.values():
                if item.expires_at is not None:
                    total_ttl += item.expires_at - current_time
                    expiring_items += 1
            
            avg_ttl = total_ttl / expiring_items if expiring_items > 0 else None
            
            return {
                "total_size": total_size,
                "average_ttl": avg_ttl,
                "expiring_items": expiring_items,
                "permanent_items": total_size - expiring_items,
                "default_ttl": self._default_ttl
            }
    
    def set_default_ttl(self, ttl: int) -> None:
        """设置默认过期时间
        
        Args:
            ttl: 默认过期时间（秒）
        """
        self._default_ttl = ttl


# 全局缓存管理器实例
default_cache_manager = CacheManager()


# 便捷函数
def get_cache(key: str) -> Optional[Any]:
    """获取缓存值
    
    Args:
        key: 缓存键
    
    Returns:
        Optional[Any]: 缓存值
    """
    return default_cache_manager.get(key)


def set_cache(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """设置缓存值
    
    Args:
        key: 缓存键
        value: 缓存值
        ttl: 过期时间（秒）
    """
    default_cache_manager.set(key, value, ttl)


def delete_cache(key: str) -> None:
    """删除缓存
    
    Args:
        key: 缓存键
    """
    default_cache_manager.delete(key)


def clear_cache() -> None:
    """清空所有缓存"""
    default_cache_manager.clear()


def cache(ttl: Optional[int] = None, key_prefix: str = ""):
    """缓存装饰器
    
    示例用法：
    @cache(ttl=60, key_prefix="user_")
    def get_user(user_id):
        # 从数据库获取用户信息
        return db.query(User).filter(User.id == user_id).first()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """同步函数包装器"""
            # 生成缓存键
            key = f"{key_prefix}{func.__name__}:{args}:{tuple(sorted(kwargs.items()))}"
            
            # 尝试从缓存获取
            result = get_cache(key)
            if result is not None:
                return result
            
            # 执行函数获取结果
            result = func(*args, **kwargs)
            
            # 缓存结果
            set_cache(key, result, ttl)
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """异步函数包装器"""
            # 生成缓存键
            key = f"{key_prefix}{func.__name__}:{args}:{tuple(sorted(kwargs.items()))}"
            
            # 尝试从缓存获取
            result = get_cache(key)
            if result is not None:
                return result
            
            # 执行异步函数获取结果
            result = await func(*args, **kwargs)
            
            # 缓存结果
            set_cache(key, result, ttl)
            return result
        
        return async_wrapper if hasattr(func, "__await__") else wrapper
    
    return decorator


# 导出所有内容
__all__ = [
    "CacheManager",
    "CacheItem",
    "default_cache_manager",
    "get_cache",
    "set_cache",
    "delete_cache",
    "clear_cache",
    "cache"
]