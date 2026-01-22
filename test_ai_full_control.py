#!/usr/bin/env python3
"""AI完整控制测试脚本"""

import requests
import time
import random

BASE_URL = "http://localhost:8001"
PTZ_CONFIG = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://192.168.1.1",
    "username": "admin",
    "password": "admin"
}

def test_ai_full_control():
    """测试AI完整控制流程"""
    print("=== AI完整控制测试 ===")
    
    # 1. 关闭现有摄像头
    print("1. 关闭现有摄像头...")
    try:
        close_result = requests.post(f"{BASE_URL}/api/camera/close").json()
        print(f"   关闭结果: {close_result}")
    except Exception as e:
        print(f"   关闭摄像头错误: {e}")
    
    # 2. 打开摄像头
    print("2. 打开摄像头...")
    open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
    print(f"   打开结果: {open_result}")
    
    if not open_result.get("success"):
        print("   ❌ 摄像头打开失败，测试终止")
        return False
    
    # 3. 断开现有PTZ连接
    print("3. 断开现有PTZ连接...")
    try:
        disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
        print(f"   断开结果: {disconnect_result}")
    except Exception as e:
        print(f"   断开连接错误: {e}")
    
    # 4. 连接PTZ
    print("4. 连接PTZ...")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=PTZ_CONFIG).json()
    print(f"   连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print("   ❌ PTZ连接失败，测试终止")
        return False
    
    # 5. 启动视觉识别
    print("5. 启动视觉识别...")
    recognition_result = requests.post(f"{BASE_URL}/api/camera/recognition/start", 
                                     json={"model_type": "haar"}).json()
    print(f"   启动结果: {recognition_result}")
    
    # 6. 模拟AI决策和控制
    print("6. 模拟AI决策和控制...")
    
    # 获取初始状态
    status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    current_pan = status["data"]["position"]["pan"]
    current_tilt = status["data"]["position"]["tilt"]
    
    print(f"   初始位置: pan={current_pan:.1f}, tilt={current_tilt:.1f}")
    
    # 测试3次AI控制
    for i in range(3):
        print(f"\n7. AI控制轮次 {i+1}/3:")
        
        # 模拟AI检测到目标位置
        target_pan = random.uniform(-180, 180)
        target_tilt = random.uniform(-90, 90)
        
        print(f"   AI检测到目标，需要移动到: pan={target_pan:.1f}, tilt={target_tilt:.1f}")
        
        # 调用PTZ移动API
        move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                                  json={"pan": target_pan, "tilt": target_tilt, "speed": 90})
        
        print(f"   移动命令响应: {move_result.json()}")
        
        # 等待摄像头移动完成
        time.sleep(4)
        
        # 获取当前位置
        status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
        new_pan = status["data"]["position"]["pan"]
        new_tilt = status["data"]["position"]["tilt"]
        
        print(f"   移动后位置: pan={new_pan:.1f}, tilt={new_tilt:.1f}")
        
        # 计算移动角度差
        pan_diff = abs(new_pan - current_pan)
        tilt_diff = abs(new_tilt - current_tilt)
        
        print(f"   移动角度差: pan={pan_diff:.1f}°, tilt={tilt_diff:.1f}°")
        
        # 验证是否有明显移动
        if pan_diff > 5 or tilt_diff > 5:
            print(f"   ✅ AI控制成功！云台发生了明显移动")
        else:
            print(f"   ⚠️  云台移动不明显，可能存在问题")
        
        # 更新当前位置
        current_pan = new_pan
        current_tilt = new_tilt
    
    # 8. 停止视觉识别
    print("\n8. 停止视觉识别...")
    stop_result = requests.post(f"{BASE_URL}/api/camera/recognition/stop").json()
    print(f"   停止结果: {stop_result}")
    
    # 9. 断开PTZ连接
    print("9. 断开PTZ连接...")
    disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print(f"   断开结果: {disconnect_result}")
    
    # 10. 关闭摄像头
    print("10. 关闭摄像头...")
    close_result = requests.post(f"{BASE_URL}/api/camera/close").json()
    print(f"   关闭结果: {close_result}")
    
    return True

if __name__ == "__main__":
    test_ai_full_control()