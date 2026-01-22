#!/usr/bin/env python3
"""
测试Flax兼容性补丁的日志输出
"""

import sys
import os

# 添加项目根目录到Python路径
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

print("=== 测试Flax兼容性补丁日志 ===")

# 配置日志（先于Flax补丁加载）
import logging.config
from src.core.utils.logging_utils import DEFAULT_LOGGING_CONFIG, LoggerManager
logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)
logger_manager = LoggerManager()

# 应用Flax兼容性补丁
from src.core.utils.flax_patch import apply_flax_patch
result = apply_flax_patch()

print(f"\nFlax补丁应用结果: {result}")

# 查看日志文件中是否有相关记录
print("\n=== 查看最新日志 ===")
import subprocess
result = subprocess.run(["Get-Content", "-Path", "d:/1.6/1.5/backend/logs/app.log", "-Tail", "20"], 
                       capture_output=True, text=True, shell=True)

# 过滤与flax_patch相关的日志
flax_logs = [line for line in result.stdout.split('\n') if 'flax' in line.lower()]
if flax_logs:
    print("找到Flax补丁相关日志:")
    for log in flax_logs:
        print(f"  {log}")
else:
    print("未找到Flax补丁相关日志")
    print("\n最新日志（全部）:")
    for line in result.stdout.split('\n')[-10:]:
        print(f"  {line}")

print("\n=== 测试完成 ===")
