#!/usr/bin/env python3
"""
测试日志配置是否正常工作
"""

import logging
import sys
import os

# 添加项目根目录到Python路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

# 导入日志配置
from src.core.utils.logging_utils import get_logger, logger_manager, JSONFormatter

print("=== 测试日志配置 ===")

# 打印当前日志配置
root_logger = logging.getLogger()
print(f"\n1. 根日志器级别: {logging.getLevelName(root_logger.level)}")
print(f"2. 根日志器处理器数量: {len(root_logger.handlers)}")

# 打印每个处理器的信息
for i, handler in enumerate(root_logger.handlers):
    print(f"\n处理器 {i+1}:")
    print(f"   类型: {type(handler).__name__}")
    print(f"   级别: {logging.getLevelName(handler.level)}")
    print(f"   格式化器: {type(handler.formatter).__name__}")
    if hasattr(handler, 'baseFilename'):
        print(f"   文件名: {handler.baseFilename}")
    if hasattr(handler, 'maxBytes'):
        print(f"   最大字节数: {handler.maxBytes}")
    if hasattr(handler, 'backupCount'):
        print(f"   备份数量: {handler.backupCount}")

# 测试日志记录
print("\n=== 测试日志记录 ===")
logger = get_logger("test_logger")

logger.debug("这是一条DEBUG级别的日志")
logger.info("这是一条INFO级别的日志")
logger.warning("这是一条WARNING级别的日志")
logger.error("这是一条ERROR级别的日志")
logger.critical("这是一条CRITICAL级别的日志")

# 测试JSON格式化器
print("\n=== 测试JSON格式化器直接输出 ===")
formatter = JSONFormatter()
record = logging.LogRecord(
    name="test_json",
    level=logging.INFO,
    pathname="test_logging.py",
    lineno=42,
    msg="测试JSON格式化器",
    args=(),
    exc_info=None
)
json_output = formatter.format(record)
print(f"JSON输出: {json_output}")

print("\n=== 测试完成 ===")
