#!/usr/bin/env python3
"""
后端服务主入口
"""

import uvicorn
import sys
import os
import importlib.util

# 将项目根目录添加到Python路径
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
sys.path.append(project_root)

# 应用Flax兼容性补丁
try:
    flax_patch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "utils", "flax_patch.py")
    spec = importlib.util.spec_from_file_location("flax_patch", flax_patch_path)
    flax_patch_module = importlib.util.module_from_spec(spec)
    sys.modules["flax_patch"] = flax_patch_module
    spec.loader.exec_module(flax_patch_module)
    flax_patch_module.apply_flax_patch()
    print("Flax兼容性补丁应用成功")
except Exception as e:
    print(f"Flax兼容性补丁应用失败: {e}")
    # 继续执行，即使补丁应用失败

from src.api import create_app

# 创建FastAPI应用
app = create_app()

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
