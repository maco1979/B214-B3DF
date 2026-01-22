#!/usr/bin/env python3
"""
人脸识别和PTZ控制综合测试脚本
"""

import asyncio
import logging
from backend.src.core.services.camera_controller import CameraController
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_face_tracking_ptz():
    """测试人脸识别和PTZ控制"""
    print("=== 人脸识别和PTZ控制综合测试 ===")
    
    # 1. 初始化摄像头控制器
    camera_controller = CameraController()
    print("\n1. 摄像头控制器初始化完成")
    
    # 2. 打开摄像头
    print("\n2. 打开摄像头...")
    result = camera_controller.open_camera(0)  # 尝试打开真实摄像头
    if not result["success"]:
        print(f"真实摄像头打开失败: {result['message']}")
        print("尝试打开模拟摄像头...")
        result = camera_controller.open_camera(999)  # 打开模拟摄像头
    
    print(f"打开摄像头结果: {result}")
    if not result["success"]:
        print("摄像头打开失败，测试终止")
        return
    
    # 3. 启动视觉识别
    print("\n3. 启动视觉识别...")
    result = camera_controller.start_visual_recognition(model_type='haar')
    print(f"启动识别结果: {result}")
    
    # 4. 启动视觉跟踪
    print("\n4. 启动视觉跟踪...")
    result = camera_controller.start_visual_tracking(tracker_type='MIL')
    print(f"启动跟踪结果: {result}")
    
    # 5. 初始化并连接PTZ
    print("\n5. 初始化PTZ控制器...")
    ptz_controller = PTZCameraController(
        protocol=PTZProtocol.HTTP_API,
        connection_type='http',
        base_url='http://localhost:8001',
        username='admin',
        password='admin'
    )
    
    print("\n6. 连接PTZ...")
    result = await ptz_controller.connect()
    print(f"连接结果: {result}")
    print(f"连接状态: {ptz_controller.is_connected}")
    
    # 6. 测试人脸识别
    print("\n7. 测试人脸识别...")
    print("等待3秒，观察人脸检测结果...")
    await asyncio.sleep(3)
    
    # 获取识别状态
    recognition_status = camera_controller.get_recognition_status()
    print(f"\n人脸识别结果:")
    print(f"  识别启用: {recognition_status['recognizing_enabled']}")
    print(f"  识别物体数量: {recognition_status['recognized_objects_count']}")
    print(f"  识别物体: {recognition_status['recognized_objects']}")
    
    # 7. 测试PTZ控制
    print("\n8. 测试PTZ控制...")
    if ptz_controller.is_connected:
        print("  测试右转...")
        result = await ptz_controller.execute_action(PTZAction.PAN_RIGHT, speed=50)
        print(f"    右转结果: {result}")
        
        await asyncio.sleep(1)
        
        print("  测试左转...")
        result = await ptz_controller.execute_action(PTZAction.PAN_LEFT, speed=50)
        print(f"    左转结果: {result}")
        
        await asyncio.sleep(1)
        
        print("  测试向上转...")
        result = await ptz_controller.execute_action(PTZAction.TILT_UP, speed=50)
        print(f"    向上转结果: {result}")
        
        await asyncio.sleep(1)
        
        print("  测试向下转...")
        result = await ptz_controller.execute_action(PTZAction.TILT_DOWN, speed=50)
        print(f"    向下转结果: {result}")
        
        await asyncio.sleep(1)
        
        print("  测试停止...")
        result = await ptz_controller.execute_action(PTZAction.STOP, speed=0)
        print(f"    停止结果: {result}")
        
        # 8. 测试PTZ自动跟踪
        print("\n9. 测试PTZ自动跟踪...")
        print("等待5秒，观察PTZ自动跟踪效果...")
        
        for i in range(5):
            # 获取跟踪状态
            tracking_status = camera_controller.get_tracking_status()
            print(f"\n  第{i+1}秒:")
            print(f"    跟踪状态: {tracking_status}")
            
            if tracking_status["tracking_enabled"] and tracking_status["tracked_object"]:
                bbox = tracking_status["tracked_object"]
                frame_size = (640, 480)  # 假设帧大小
                
                # 调用PTZ自动跟踪
                result = await ptz_controller.auto_track_object(bbox, frame_size)
                print(f"    PTZ自动跟踪结果: {result}")
                
                # 获取PTZ状态
                ptz_status = ptz_controller.get_status()
                print(f"    PTZ当前位置: pan={ptz_status['position']['pan']:.1f}°, tilt={ptz_status['position']['tilt']:.1f}°, zoom={ptz_status['position']['zoom']:.1f}x")
            
            await asyncio.sleep(1)
    
    # 9. 清理资源
    print("\n10. 清理资源...")
    
    # 停止跟踪
    result = camera_controller.stop_visual_tracking()
    print(f"停止跟踪结果: {result}")
    
    # 停止识别
    result = camera_controller.stop_visual_recognition()
    print(f"停止识别结果: {result}")
    
    # 断开PTZ
    if ptz_controller.is_connected:
        result = await ptz_controller.disconnect()
        print(f"断开PTZ结果: {result}")
    
    # 关闭摄像头
    result = camera_controller.close_camera()
    print(f"关闭摄像头结果: {result}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    asyncio.run(test_face_tracking_ptz())