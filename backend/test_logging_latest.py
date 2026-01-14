#!/usr/bin/env python3
"""
测试最新的日志配置
"""

import sys
import os
import logging.config

# 添加项目根目录到Python路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

print("=== 测试最新日志配置 ===")

# 先配置日志
from src.core.utils.logging_utils import DEFAULT_LOGGING_CONFIG
logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)

# 再初始化日志管理器
from src.core.utils.logging_utils import LoggerManager
logger_manager = LoggerManager()

# 获取日志记录器
logger = logging.getLogger("test_latest")

# 测试日志记录
print("\n=== 开始记录日志 ===")
logger.debug("这是一条DEBUG日志")
logger.info("这是一条INFO日志")
logger.warning("这是一条WARNING日志")
logger.error("这是一条ERROR日志")
logger.critical("这是一条CRITICAL日志")

print("=== 测试完成 ===")
