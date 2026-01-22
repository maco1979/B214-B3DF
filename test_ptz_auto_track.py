#!/usr/bin/env python3
"""
PTZ自动跟踪测试脚本
测试摄像头跟踪与PTZ控制的集成
"""

import asyncio
import logging
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def test_ptz_auto_track():
    """测试PTZ自动跟踪功能"""
    print("=== PTZ自动跟踪测试 ===")
    
    # 1. 初始化摄像头控制器
    camera_controller = CameraController()
    print("摄像头控制器初始化完成")
    
    # 2. 打开模拟摄像头
    print("\n1. 打开模拟摄像头...")
    result = camera_controller.open_camera(999)  # 使用模拟摄像头
    print(f"打开摄像头结果: {result}")
    
    if not result["success"]:
        print("无法打开摄像头，测试终止")
        return
    
    # 3. 初始化PTZ控制器
    print("\n2. 初始化PTZ控制器...")
    ptz_controller = PTZCameraController(
        protocol=PTZProtocol.HTTP_API,
        connection_type='http',
        base_url='http://localhost:8001',
        username='admin',
        password='admin'
    )
    
    # 4. 连接PTZ
    print("\n3. 连接PTZ控制器...")
    result = await ptz_controller.connect()
    print(f"连接PTZ结果: {result}")
    print(f"PTZ连接状态: {ptz_controller.is_connected}")
    
    if not ptz_controller.is_connected:
        print("PTZ连接失败，测试终止")
        return
    
    # 5. 启动视觉识别和跟踪
    print("\n4. 启动视觉识别...")
    result = camera_controller.start_visual_recognition(model_type='haar')
    print(f"启动识别结果: {result}")
    
    print("\n5. 启动视觉跟踪...")
    result = camera_controller.start_visual_tracking(tracker_type='MIL')
    print(f"启动跟踪结果: {result}")
    
    # 6. 测试PTZ自动跟踪
    print("\n6. 测试PTZ自动跟踪...")
    print("等待5秒，观察PTZ自动跟踪效果...")
    
    for i in range(5):
        # 获取跟踪状态
        tracking_status = camera_controller.get_tracking_status()
        print(f"\n第{i+1}秒:")
        print(f"  跟踪状态: {tracking_status}")
        
        # 模拟跟踪对象移动并测试PTZ自动跟踪
        if tracking_status["tracking_enabled"] and tracking_status["tracked_object"]:
            bbox = tracking_status["tracked_object"]
            frame_size = (640, 480)  # 模拟摄像头尺寸
            
            # 调用PTZ自动跟踪
            result = await ptz_controller.auto_track_object(bbox, frame_size)
            print(f"  PTZ自动跟踪结果: {result}")
            
            # 获取PTZ状态
            ptz_status = ptz_controller.get_status()
            print(f"  PTZ当前状态: {ptz_status}")
        
        await asyncio.sleep(1)
    
    # 7. 停止跟踪和识别
    print("\n7. 停止跟踪和识别...")
    result = camera_controller.stop_visual_tracking()
    print(f"停止跟踪结果: {result}")
    
    result = camera_controller.stop_visual_recognition()
    print(f"停止识别结果: {result}")
    
    # 8. 断开PTZ连接
    print("\n8. 断开PTZ连接...")
    result = await ptz_controller.disconnect()
    print(f"断开PTZ结果: {result}")
    
    # 9. 关闭摄像头
    print("\n9. 关闭摄像头...")
    result = camera_controller.close_camera()
    print(f"关闭摄像头结果: {result}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_ptz_auto_track())