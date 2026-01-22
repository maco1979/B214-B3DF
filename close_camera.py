#!/usr/bin/env python3
"""
关闭摄像头的简单脚本
"""

from backend.src.core.services.camera_controller import CameraController

# 创建摄像头控制器实例
camera = CameraController()

print("=== 关闭摄像头 ===")

# 检查摄像头状态
is_open = camera.is_camera_open()
print(f"当前摄像头状态: {'已打开' if is_open else '已关闭'}")

if is_open:
    # 关闭摄像头
    result = camera.close_camera()
    print(f"关闭摄像头结果: {result}")
    
    # 再次检查状态
    new_status = camera.is_camera_open()
    print(f"关闭后状态: {'已关闭' if not new_status else '仍在打开'}")
else:
    print("摄像头已关闭，无需操作")

print("=== 操作完成 ===")