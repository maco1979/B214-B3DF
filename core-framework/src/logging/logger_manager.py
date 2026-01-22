"""
日志系统模块
提供结构化的日志记录，支持多级别日志、日志轮转和分布式追踪
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class LoggerWrapper:
    """日志包装器，提供统一的日志接口"""
    
    def __init__(self, name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        self.extra = extra or {}
        self._logger = logger.bind(module=name, **self.extra)
    
    def debug(self, message: str, **kwargs) -> None:
        """记录调试级别日志"""
        self._logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """记录信息级别日志"""
        self._logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """记录警告级别日志"""
        self._logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """记录错误级别日志"""
        self._logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """记录严重级别日志"""
        self._logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """记录异常信息"""
        self._logger.exception(message, **kwargs)
    
    def with_fields(self, **fields) -> 'LoggerWrapper':
        """创建带有额外字段的新日志器"""
        new_extra = {**self.extra, **fields}
        return LoggerWrapper(self.name, new_extra)


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "30 days",
    compression: Optional[str] = "gz",
    serialize: bool = True
) -> None:
    """
    设置日志系统
    
    Args:
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式 (json, simple)
        log_file: 日志文件路径
        rotation: 日志轮转大小
        retention: 日志保留时间
        compression: 日志压缩格式
        serialize: 是否序列化日志
    """
    # 移除默认的日志处理器
    logger.remove()
    
    # 配置日志格式
    if log_format == "json":
        format_str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"
    else:
        format_str = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # 添加控制台日志处理器
    logger.add(
        sys.stderr,
        level=log_level,
        format=format_str,
        colorize=True if log_format != "json" else False,
        backtrace=True,
        diagnose=True
    )
    
    # 添加文件日志处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            level=log_level,
            format=format_str,
            rotation=rotation,
            retention=retention,
            compression=compression,
            serialize=serialize,
            backtrace=True,
            diagnose=True
        )
    
    # 配置序列化（JSON格式）
    if serialize and log_format == "json":
        def serialize_record(record):
            record["extra"]["serialized"] = json.dumps({
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["name"],
                "function": record["function"],
                "line": record["line"],
                "extra": record["extra"]
            })
        
        logger = logger.patch(serialize_record)
    
    # 设置全局日志级别
    logger.level(log_level)
    
    # 记录日志系统启动信息
    logger.info(
        "日志系统已初始化",
        log_level=log_level,
        log_format=log_format,
        log_file=log_file
    )


def get_logger(name: str, **kwargs) -> LoggerWrapper:
    """
    获取指定名称的日志器
    
    Args:
        name: 日志器名称
        **kwargs: 额外字段
        
    Returns:
        LoggerWrapper实例
    """
    return LoggerWrapper(name, kwargs)


# 全局日志配置
_logging_config: Optional[Dict[str, Any]] = None


def configure_logging(config: Dict[str, Any]) -> None:
    """
    根据配置字典设置日志系统
    
    Args:
        config: 日志配置字典
    """
    global _logging_config
    _logging_config = config
    
    setup_logging(
        log_level=config.get("log_level", "INFO"),
        log_format=config.get("log_format", "json"),
        log_file=config.get("log_file"),
        rotation=config.get("rotation", "10 MB"),
        retention=config.get("retention", "30 days"),
        compression=config.get("compression", "gz"),
        serialize=config.get("serialize", True)
    )


def get_logging_config() -> Optional[Dict[str, Any]]:
    """获取当前日志配置"""
    return _logging_config


# 预定义的日志器
app_logger = get_logger("application")
db_logger = get_logger("database")
api_logger = get_logger("api")
auth_logger = get_logger("auth")
service_logger = get_logger("service")