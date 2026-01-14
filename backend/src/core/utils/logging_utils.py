"""
统一日志管理模块
提供统一的日志配置和记录方法
"""

import logging
import logging.config
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

# 默认日志配置（先使用detailed格式化器，后续会替换为JSON格式化器）
DEFAULT_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "detailed",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8"
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": "logs/error.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file", "error_file"],
            "level": "INFO",
            "propagate": True
        },
        "uvicorn": {
            "handlers": ["console", "file", "error_file"],
            "level": "WARNING",
            "propagate": False
        },
        "fastapi": {
            "handlers": ["console", "file", "error_file"],
            "level": "WARNING",
            "propagate": False
        },
        "sqlalchemy": {
            "handlers": ["console", "file", "error_file"],
            "level": "WARNING",
            "propagate": False
        },
        "websockets": {
            "handlers": ["console", "file", "error_file"],
            "level": "WARNING",
            "propagate": False
        },
        "flax_patch": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}


class JSONFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def format(self, record):
        """将日志记录格式化为JSON字符串"""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
            "module": record.module,
            "process": record.process,
            "thread": record.thread,
            "exception": record.exc_text if record.exc_info else None
        }
        
        # 添加额外的上下文信息
        if hasattr(record, "context"):
            log_data["context"] = record.context
        
        return json.dumps(log_data, ensure_ascii=False)


class LoggerManager:
    """日志管理器"""
    
    def __init__(self):
        """初始化日志管理器"""
        self._loggers = {}
        self._context_stack = []
        
        # 确保日志目录存在
        os.makedirs("logs", exist_ok=True)
        
        # 配置日志（使用临时的detailed格式化器）
        logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
        
        # 获取配置的logger
        self._root_logger = logging.getLogger()
        
        # 替换所有处理器的格式化器为JSON格式化器
        json_formatter = JSONFormatter()
        for handler in self._root_logger.handlers:
            handler.setFormatter(json_formatter)
        
        # 记录初始化信息
        self._root_logger.info("日志系统初始化完成")
    
    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """获取日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        
        Returns:
            logging.Logger: 日志记录器实例
        """
        if name not in self._loggers:
            logger = logging.getLogger(name)
            if level:
                logger.setLevel(getattr(logging, level.upper()))
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_global_level(self, level: str):
        """设置全局日志级别
        
        Args:
            level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
        """
        self._root_logger.setLevel(getattr(logging, level.upper()))
        
    def push_context(self, context: Dict[str, Any]):
        """推送日志上下文
        
        Args:
            context: 上下文信息
        """
        self._context_stack.append(context)
    
    def pop_context(self) -> Optional[Dict[str, Any]]:
        """弹出日志上下文
        
        Returns:
            Optional[Dict[str, Any]]: 弹出的上下文信息
        """
        if self._context_stack:
            return self._context_stack.pop()
        return None
    
    def get_current_context(self) -> Dict[str, Any]:
        """获取当前上下文
        
        Returns:
            Dict[str, Any]: 当前上下文信息
        """
        if self._context_stack:
            return self._context_stack[-1]
        return {}
    
    def clear_context(self):
        """清除所有上下文"""
        self._context_stack.clear()


# 全局日志管理器实例
logger_manager = LoggerManager()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """便捷函数：获取日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
    
    Returns:
        logging.Logger: 日志记录器实例
    """
    return logger_manager.get_logger(name, level)


def log_context(context: Dict[str, Any]):
    """日志上下文装饰器
    
    示例用法：
    @log_context({"service": "user_service", "version": "1.0.0"})
    def get_user(user_id):
        logger = get_logger("user_service")
        logger.info(f"获取用户信息: {user_id}")
        # 业务逻辑
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """同步函数包装器"""
            try:
                # 推送上下文
                logger_manager.push_context(context)
                
                # 执行函数
                return func(*args, **kwargs)
            finally:
                # 弹出上下文
                logger_manager.pop_context()
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """异步函数包装器"""
            try:
                # 推送上下文
                logger_manager.push_context(context)
                
                # 执行函数
                return await func(*args, **kwargs)
            finally:
                # 弹出上下文
                logger_manager.pop_context()
        
        return async_wrapper if hasattr(func, "__await__") else wrapper
    
    return decorator


# 便捷的日志记录函数
def debug(logger: logging.Logger, message: str, **kwargs):
    """记录DEBUG级别的日志"""
    logger.debug(message, extra={"context": {**logger_manager.get_current_context(), **kwargs}})

def info(logger: logging.Logger, message: str, **kwargs):
    """记录INFO级别的日志"""
    logger.info(message, extra={"context": {**logger_manager.get_current_context(), **kwargs}})

def warning(logger: logging.Logger, message: str, **kwargs):
    """记录WARNING级别的日志"""
    logger.warning(message, extra={"context": {**logger_manager.get_current_context(), **kwargs}})

def error(logger: logging.Logger, message: str, **kwargs):
    """记录ERROR级别的日志"""
    logger.error(message, extra={"context": {**logger_manager.get_current_context(), **kwargs}})

def critical(logger: logging.Logger, message: str, **kwargs):
    """记录CRITICAL级别的日志"""
    logger.critical(message, extra={"context": {**logger_manager.get_current_context(), **kwargs}})


# 导出所有内容
__all__ = [
    "LoggerManager",
    "JSONFormatter",
    "logger_manager",
    "get_logger",
    "log_context",
    "debug",
    "info",
    "warning",
    "error",
    "critical"
]