#!/usr/bin/env python3
"""
测试摄像头状态的脚本
"""

from backend.src.core.services.camera_controller import CameraController

# 创建摄像头控制器实例
camera = CameraController()

print("=== 摄像头状态测试 ===")

# 1. 检查摄像头基本状态
print("1. 检查摄像头状态...")
is_open = camera.is_camera_open()
print(f"   摄像头是否打开: {'是' if is_open else '否'}")

# 2. 检查识别和跟踪状态
print("\n2. 检查识别和跟踪状态...")
# 使用公共方法获取状态
recognition_status = camera.get_recognition_status()
tracking_status = camera.get_tracking_status()

print(f"   识别是否启用: {'是' if recognition_status['recognizing_enabled'] else '否'}")
print(f"   跟踪是否启用: {'是' if tracking_status['tracking_enabled'] else '否'}")

# 3. 如果摄像头打开，关闭它
if is_open:
    print("\n3. 关闭摄像头...")
    result = camera.close_camera()
    print(f"   关闭结果: {result}")
    
    # 4. 再次检查状态
    new_status = camera.is_camera_open()
    print(f"   关闭后状态: {'是' if new_status else '否'}")
else:
    print("\n3. 摄像头已关闭，无需操作")

print("\n=== 测试完成 ===")