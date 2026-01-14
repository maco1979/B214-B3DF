#!/usr/bin/env python3
"""
简单的人脸识别测试脚本
"""

from backend.src.core.services.camera_controller import CameraController
import time

# 创建摄像头控制器
camera = CameraController()

print("=== 人脸识别测试 ===")

# 1. 打开真实摄像头
print("1. 打开真实摄像头...")
result = camera.open_camera(0)
print(f"   结果: {result}")

if not camera.is_camera_open():
    print("真实摄像头打开失败，测试终止")
    exit(1)

print(f"   摄像头状态: 已打开")

# 2. 启动视觉识别
print("\n2. 启动视觉识别...")
result = camera.start_visual_recognition()
print(f"   结果: {result}")

# 3. 等待2秒，让识别稳定
print("\n3. 等待2秒，让识别稳定...")
time.sleep(2)

# 4. 获取识别状态
print("\n4. 获取识别状态...")
status = camera.get_recognition_status()
print(f"   识别启用: {status['recognizing_enabled']}")
print(f"   识别物体数量: {status['recognized_objects_count']}")
print(f"   识别物体: {status['recognized_objects']}")

# 5. 停止识别
print("\n5. 停止识别...")
result = camera.stop_visual_recognition()
print(f"   结果: {result}")

# 6. 关闭摄像头
print("\n6. 关闭摄像头...")
result = camera.close_camera()
print(f"   结果: {result}")

print("\n=== 测试完成 ===")