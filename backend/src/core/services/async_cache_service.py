"""
异步缓存服务 - 提供高性能Redis缓存和内存缓存
支持异步调用、自动过期、LRU策略等高级功能
"""

import asyncio
import json
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """缓存条目"""
    value: T
    created_at: float
    expires_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class AsyncMemoryCache:
    """异步内存缓存 - 使用LRU策略"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """异步获取缓存值"""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            current_time = time.time()
            
            # 检查是否过期
            if current_time > entry.expires_at:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            # 更新访问信息（LRU）
            entry.access_count += 1
            entry.last_accessed = current_time
            
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """异步设置缓存值"""
        async with self._lock:
            ttl = ttl or self.default_ttl
            current_time = time.time()
            
            # 如果达到最大容量，删除最少使用的条目
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            self._cache[key] = CacheEntry(
                value=value,
                created_at=current_time,
                expires_at=current_time + ttl,
                access_count=1,
                last_accessed=current_time
            )
            return True
    
    async def delete(self, key: str) -> bool:
        """异步删除缓存"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self):
        """清除所有缓存"""
        async with self._lock:
            self._cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        async with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate": round(hit_rate * 100, 2)
            }


class AsyncRedisCache:
    """异步Redis缓存"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", default_ttl: int = 300):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self._client = None
        self._connected = False
        self._fallback_cache = AsyncMemoryCache()
    
    async def connect(self):
        """连接到Redis"""
        try:
            import redis.asyncio as aioredis
            self._client = await aioredis.from_url(self.redis_url)
            await self._client.ping()
            self._connected = True
            logger.info("Redis连接成功")
        except Exception as e:
            logger.warning(f"Redis连接失败，使用内存缓存: {e}")
            self._connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """异步获取缓存值"""
        if not self._connected:
            return await self._fallback_cache.get(key)
        
        try:
            data = await self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis读取失败: {e}")
            return await self._fallback_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """异步设置缓存值"""
        ttl = ttl or self.default_ttl
        
        if not self._connected:
            return await self._fallback_cache.set(key, value, ttl)
        
        try:
            await self._client.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.error(f"Redis写入失败: {e}")
            return await self._fallback_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """异步删除缓存"""
        if not self._connected:
            return await self._fallback_cache.delete(key)
        
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False
    
    async def get_or_set(self, key: str, factory: Callable, ttl: Optional[int] = None) -> Any:
        """获取缓存值，如果不存在则使用factory创建并缓存"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # 生成新值
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        await self.set(key, value, ttl)
        return value


class AsyncSystemMetrics:
    """异步系统指标收集器"""
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache = AsyncMemoryCache(max_size=100, default_ttl=5)
    
    async def get_cpu_percent(self) -> float:
        """异步获取CPU使用率"""
        cache_key = "cpu_percent"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        import psutil
        loop = asyncio.get_event_loop()
        value = await loop.run_in_executor(
            self._executor,
            lambda: psutil.cpu_percent(interval=0.1)
        )
        await self._cache.set(cache_key, value, ttl=2)
        return value
    
    async def get_memory_info(self) -> Dict[str, Any]:
        """异步获取内存信息"""
        cache_key = "memory_info"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        import psutil
        loop = asyncio.get_event_loop()
        memory = await loop.run_in_executor(
            self._executor,
            psutil.virtual_memory
        )
        value = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }
        await self._cache.set(cache_key, value, ttl=5)
        return value
    
    async def get_disk_info(self) -> Dict[str, Any]:
        """异步获取磁盘信息"""
        cache_key = "disk_info"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        import psutil
        loop = asyncio.get_event_loop()
        disk = await loop.run_in_executor(
            self._executor,
            lambda: psutil.disk_usage('/')
        )
        value = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
        await self._cache.set(cache_key, value, ttl=10)
        return value
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """异步并行获取所有系统指标"""
        # 并行获取所有指标
        cpu_task = self.get_cpu_percent()
        memory_task = self.get_memory_info()
        disk_task = self.get_disk_info()
        
        cpu, memory, disk = await asyncio.gather(
            cpu_task, memory_task, disk_task
        )
        
        return {
            "cpu": {"percent": cpu},
            "memory": memory,
            "disk": disk,
            "timestamp": datetime.now().isoformat()
        }


# 全局实例
memory_cache = AsyncMemoryCache()
redis_cache = AsyncRedisCache()
system_metrics = AsyncSystemMetrics()


def cache_result(ttl: int = 300, key_prefix: str = ""):
    """缓存结果装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            key_parts = [key_prefix or func.__name__]
            if args:
                key_parts.append(hashlib.md5(str(args).encode()).hexdigest()[:8])
            if kwargs:
                key_parts.append(hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8])
            cache_key = ":".join(key_parts)
            
            # 尝试从缓存获取
            cached = await memory_cache.get(cache_key)
            if cached is not None:
                return cached
            
            # 执行函数
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # 缓存结果
            await memory_cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator


class DecisionResultCache:
    """决策结果缓存 - 专门用于缓存AI决策结果"""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 60):
        self._cache = AsyncMemoryCache(max_size=max_size, default_ttl=default_ttl)
        self._decision_stats = {
            "total_decisions": 0,
            "cached_decisions": 0,
            "cache_hit_rate": 0.0
        }
    
    def _generate_decision_key(self, module: str, state: Dict, objective: str) -> str:
        """生成决策缓存键"""
        state_hash = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]
        return f"decision:{module}:{objective}:{state_hash}"
    
    async def get_cached_decision(self, module: str, state: Dict, objective: str) -> Optional[Dict]:
        """获取缓存的决策结果"""
        key = self._generate_decision_key(module, state, objective)
        result = await self._cache.get(key)
        
        self._decision_stats["total_decisions"] += 1
        if result is not None:
            self._decision_stats["cached_decisions"] += 1
        
        self._update_hit_rate()
        return result
    
    async def cache_decision(self, module: str, state: Dict, objective: str, 
                            decision: Dict, ttl: Optional[int] = None):
        """缓存决策结果"""
        key = self._generate_decision_key(module, state, objective)
        await self._cache.set(key, decision, ttl)
    
    def _update_hit_rate(self):
        """更新缓存命中率"""
        total = self._decision_stats["total_decisions"]
        if total > 0:
            self._decision_stats["cache_hit_rate"] = round(
                self._decision_stats["cached_decisions"] / total * 100, 2
            )
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取决策缓存统计"""
        cache_stats = await self._cache.get_stats()
        return {
            **self._decision_stats,
            "cache_stats": cache_stats
        }


# 全局决策缓存实例
decision_cache = DecisionResultCache()
