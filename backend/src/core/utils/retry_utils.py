"""
重试和容错工具模块
提供重试装饰器、错误处理和恢复机制
"""

import asyncio
import logging
import time
from typing import Callable, Any, Optional, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class RetryExhaustedError(Exception):
    """重试次数耗尽异常"""
    pass


class CircuitBreaker:
    """熔断器模式实现"""
    
    def __init__(self, 
                 failure_threshold: int = 5, 
                 recovery_timeout: int = 60, 
                 name: str = "circuit_breaker"):
        """初始化熔断器
        
        Args:
            failure_threshold: 失败次数阈值，超过则打开熔断器
            recovery_timeout: 恢复超时时间，超时后进入半开状态
            name: 熔断器名称
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        # 状态：closed（闭合）、open（打开）、half_open（半开）
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        
        logger.info(f"初始化熔断器: {name}, 失败阈值: {failure_threshold}, 恢复超时: {recovery_timeout}秒")
    
    def is_allowed(self) -> bool:
        """检查是否允许请求通过
        
        Returns:
            bool: 是否允许请求通过
        """
        current_time = time.time()
        
        if self.state == "open":
            # 检查是否可以进入半开状态
            if current_time - self.last_failure_time > self.recovery_timeout:
                logger.info(f"熔断器 {self.name} 从打开状态转为半开状态")
                self.state = "half_open"
                return True
            else:
                logger.warning(f"熔断器 {self.name} 处于打开状态，拒绝请求")
                return False
        
        return True
    
    def record_success(self):
        """记录成功请求"""
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= 3:  # 连续成功3次则闭合熔断器
                logger.info(f"熔断器 {self.name} 从半开状态转为闭合状态")
                self.state = "closed"
                self.failure_count = 0
                self.success_count = 0
        
        logger.debug(f"熔断器 {self.name} 记录成功，当前状态: {self.state}")
    
    def record_failure(self):
        """记录失败请求"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            logger.warning(f"熔断器 {self.name} 从闭合状态转为打开状态")
            self.state = "open"
        elif self.state == "half_open":
            logger.warning(f"熔断器 {self.name} 从半开状态转为打开状态")
            self.state = "open"
        
        logger.debug(f"熔断器 {self.name} 记录失败，当前状态: {self.state}, 失败次数: {self.failure_count}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取熔断器状态
        
        Returns:
            Dict: 熔断器状态信息
        """
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout
        }


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    circuit_breaker: Optional[CircuitBreaker] = None
) -> Callable:
    """重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 退避因子
        exceptions: 需要重试的异常类型
        circuit_breaker: 熔断器实例
    
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            current_delay = delay
            
            while attempt < max_retries:
                try:
                    # 检查熔断器是否允许请求
                    if circuit_breaker and not circuit_breaker.is_allowed():
                        raise RetryExhaustedError(f"熔断器 {circuit_breaker.name} 拒绝请求")
                    
                    # 执行函数
                    result = await func(*args, **kwargs)
                    
                    # 记录成功
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                except exceptions as e:
                    attempt += 1
                    
                    # 记录失败
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    if attempt < max_retries:
                        logger.warning(f"函数 {func.__name__} 执行失败 (尝试 {attempt}/{max_retries}): {e}，将在 {current_delay:.2f} 秒后重试")
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"函数 {func.__name__} 执行失败，已达到最大重试次数 ({max_retries}): {e}")
                        raise RetryExhaustedError(f"函数 {func.__name__} 执行失败，已达到最大重试次数") from e
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            attempt = 0
            current_delay = delay
            
            while attempt < max_retries:
                try:
                    # 检查熔断器是否允许请求
                    if circuit_breaker and not circuit_breaker.is_allowed():
                        raise RetryExhaustedError(f"熔断器 {circuit_breaker.name} 拒绝请求")
                    
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 记录成功
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    return result
                except exceptions as e:
                    attempt += 1
                    
                    # 记录失败
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    
                    if attempt < max_retries:
                        logger.warning(f"函数 {func.__name__} 执行失败 (尝试 {attempt}/{max_retries}): {e}，将在 {current_delay:.2f} 秒后重试")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger.error(f"函数 {func.__name__} 执行失败，已达到最大重试次数 ({max_retries}): {e}")
                        raise RetryExhaustedError(f"函数 {func.__name__} 执行失败，已达到最大重试次数") from e
        
        # 根据原始函数类型返回相应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class BackupManager:
    """数据备份和恢复管理器"""
    
    def __init__(self, backup_dir: str = "./backups"):
        """初始化备份管理器
        
        Args:
            backup_dir: 备份目录路径
        """
        import os
        self.backup_dir = backup_dir
        
        # 创建备份目录
        os.makedirs(backup_dir, exist_ok=True)
        
        logger.info(f"初始化备份管理器，备份目录: {backup_dir}")
    
    async def backup_data(self, data: Any, backup_name: str) -> str:
        """备份数据
        
        Args:
            data: 要备份的数据
            backup_name: 备份名称
        
        Returns:
            str: 备份文件路径
        """
        import json
        import os
        from datetime import datetime
        
        # 生成备份文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"{backup_name}_{timestamp}.json")
        
        try:
            # 保存备份
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"成功备份数据到 {backup_file}")
            return backup_file
        except Exception as e:
            logger.error(f"备份数据失败: {e}")
            raise
    
    async def restore_data(self, backup_file: str) -> Any:
        """从备份恢复数据
        
        Args:
            backup_file: 备份文件路径
        
        Returns:
            Any: 恢复的数据
        """
        import json
        
        try:
            # 读取备份
            with open(backup_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"成功从 {backup_file} 恢复数据")
            return data
        except Exception as e:
            logger.error(f"恢复数据失败: {e}")
            raise
    
    async def get_latest_backup(self, backup_name: str) -> Optional[str]:
        """获取最新的备份文件
        
        Args:
            backup_name: 备份名称
        
        Returns:
            Optional[str]: 最新备份文件路径，没有则返回None
        """
        import os
        
        try:
            # 获取所有匹配的备份文件
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith(backup_name) and filename.endswith(".json"):
                    backup_files.append(os.path.join(self.backup_dir, filename))
            
            if not backup_files:
                return None
            
            # 按修改时间排序，返回最新的
            backup_files.sort(key=os.path.getmtime, reverse=True)
            return backup_files[0]
        except Exception as e:
            logger.error(f"获取最新备份失败: {e}")
            return None
    
    async def cleanup_old_backups(self, backup_name: str, keep_days: int = 7) -> int:
        """清理旧备份
        
        Args:
            backup_name: 备份名称
            keep_days: 保留天数
        
        Returns:
            int: 清理的备份文件数量
        """
        import os
        from datetime import datetime, timedelta
        
        try:
            cutoff_time = datetime.now() - timedelta(days=keep_days)
            cutoff_timestamp = cutoff_time.timestamp()
            
            deleted_count = 0
            for filename in os.listdir(self.backup_dir):
                if filename.startswith(backup_name) and filename.endswith(".json"):
                    file_path = os.path.join(self.backup_dir, filename)
                    if os.path.getmtime(file_path) < cutoff_timestamp:
                        os.remove(file_path)
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 个过期备份")
            
            return deleted_count
        except Exception as e:
            logger.error(f"清理旧备份失败: {e}")
            return 0


# 全局实例
backup_manager = BackupManager()


# 熔断器实例
circuit_breakers = {
    "database": CircuitBreaker(name="database", failure_threshold=5, recovery_timeout=60),
    "external_api": CircuitBreaker(name="external_api", failure_threshold=3, recovery_timeout=30),
    "cache": CircuitBreaker(name="cache", failure_threshold=5, recovery_timeout=30)
}


# 便捷的重试装饰器，使用默认的熔断器
retry_database = retry(max_retries=3, delay=1.0, backoff_factor=2.0, circuit_breaker=circuit_breakers["database"])
retry_external = retry(max_retries=3, delay=2.0, backoff_factor=2.0, circuit_breaker=circuit_breakers["external_api"])
retry_cache = retry(max_retries=2, delay=0.5, backoff_factor=2.0, circuit_breaker=circuit_breakers["cache"])