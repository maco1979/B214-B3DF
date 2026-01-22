#!/usr/bin/env python3
"""
快速测试日志配置是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

# 导入日志配置
from src.core.utils.logging_utils import get_logger

print("=== 快速测试日志系统 ===")

# 测试日志记录
logger = get_logger("quick_test")
logger.info("这是一条快速测试的INFO日志")
logger.warning("这是一条快速测试的WARNING日志")

print("=== 测试完成 ===")
