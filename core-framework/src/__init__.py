"""
AI农业微服务系统核心框架层

提供通用的基础功能和工具库，包括配置管理、日志记录、错误处理、数据验证等核心功能。
"""

from .config import Config
from .logging import setup_logging, get_logger
from .errors import AppError, ErrorCode, ErrorHandler
from .validation import Validator, SchemaValidator
from .utils import DateTimeUtils, StringUtils, CryptoUtils
from .middleware import ErrorMiddleware, LoggingMiddleware

__version__ = "1.0.0"
__all__ = [
    "Config",
    "setup_logging",
    "get_logger",
    "AppError",
    "ErrorCode",
    "ErrorHandler",
    "Validator",
    "SchemaValidator",
    "DateTimeUtils",
    "StringUtils",
    "CryptoUtils",
    "ErrorMiddleware",
    "LoggingMiddleware",
]