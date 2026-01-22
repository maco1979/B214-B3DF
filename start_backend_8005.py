#!/usr/bin/env python3
"""
在8005端口启动后端服务
用于解决WebSocket连接到ws://127.0.0.1:8005/api/camera/ws/frame的问题
"""

import uvicorn
import sys
import os
import importlib.util

# 将项目根目录添加到Python路径
src_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(src_dir, "backend", "src")
sys.path.insert(0, os.path.join(src_dir, "backend"))
sys.path.insert(0, backend_dir)

# 应用Flax兼容性补丁
try:
    flax_patch_path = os.path.join(backend_dir, "core", "utils", "flax_patch.py")
    if os.path.exists(flax_patch_path):
        spec = importlib.util.spec_from_file_location("flax_patch", flax_patch_path)
        flax_patch_module = importlib.util.module_from_spec(spec)
        sys.modules["flax_patch"] = flax_patch_module
        spec.loader.exec_module(flax_patch_module)
        flax_patch_module.apply_flax_patch()
        print("Flax兼容性补丁应用成功")
    else:
        print("Flax补丁文件不存在，跳过")
except Exception as e:
    print(f"Flax兼容性补丁应用失败: {e}")

from src.api import create_app

# 创建FastAPI应用
app = create_app()

if __name__ == "__main__":
    # 在8005端口启动服务
    print("启动后端服务在端口8005...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        reload=False,  # 生产模式下关闭自动重载
        log_level="info"
    )