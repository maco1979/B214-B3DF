#!/usr/bin/env python3
"""AI 100°移动测试脚本"""

import requests
import time

BASE_URL = "http://localhost:8001"
PTZ_CONFIG = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://192.168.1.1",
    "username": "admin",
    "password": "admin"
}

def test_ai_100deg_move():
    """测试开机后AI控制云台移动超过100°"""
    print("=== AI 100°移动测试 ===")
    
    # 1. 确保摄像头已打开
    print("1. 确保摄像头已打开...")
    open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
    print(f"   摄像头状态: {open_result}")
    
    if not open_result.get("success"):
        print("   ❌ 摄像头打开失败，测试终止")
        return False
    
    # 2. 确保PTZ已连接
    print("2. 确保PTZ已连接...")
    try:
        disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
        print(f"   断开现有连接: {disconnect_result}")
    except Exception as e:
        print(f"   断开连接错误: {e}")
    
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=PTZ_CONFIG).json()
    print(f"   PTZ连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print("   ❌ PTZ连接失败，测试终止")
        return False
    
    # 3. 移动到初始位置 (确保起始点为0°)
    print("3. 移动到初始位置 (0°, 0°)...")
    init_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 0, "tilt": 0, "speed": 100})
    print(f"   初始位置设置: {init_result.json()}")
    time.sleep(3)
    
    # 4. 获取初始位置状态
    print("4. 获取初始位置状态...")
    init_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    init_pan = init_status["data"]["position"]["pan"]
    init_tilt = init_status["data"]["position"]["tilt"]
    print(f"   初始位置: pan={init_pan:.1f}°, tilt={init_tilt:.1f}°")
    
    # 5. AI控制: 移动到目标位置 (120°, 0°) - 超过100°
    print("5. AI控制: 移动到目标位置 (120°, 0°)...")
    print("   🔥 测试目标: 水平移动120°，超过100°要求")
    
    move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 120, "tilt": 0, "speed": 90})
    print(f"   移动命令响应: {move_result.json()}")
    
    # 等待摄像头移动完成
    print("   ⏳ 等待摄像头移动完成...")
    time.sleep(5)
    
    # 6. 获取移动后位置
    print("6. 获取移动后位置...")
    final_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    final_pan = final_status["data"]["position"]["pan"]
    final_tilt = final_status["data"]["position"]["tilt"]
    print(f"   移动后位置: pan={final_pan:.1f}°, tilt={final_tilt:.1f}°")
    
    # 7. 计算实际移动角度
    pan_movement = abs(final_pan - init_pan)
    tilt_movement = abs(final_tilt - init_tilt)
    
    print(f"\n=== 测试结果 ===")
    print(f"🔹 初始位置: pan={init_pan:.1f}°")
    print(f"🔹 目标位置: pan=120.0°")
    print(f"🔹 实际位置: pan={final_pan:.1f}°")
    print(f"🔹 实际移动角度: {pan_movement:.1f}°")
    
    # 8. 验证是否超过100°
    if pan_movement > 100:
        print(f"✅ 测试成功！AI控制云台移动了 {pan_movement:.1f}°，超过了100°要求")
        success = True
    else:
        print(f"❌ 测试失败！AI控制云台只移动了 {pan_movement:.1f}°，未达到100°要求")
        success = False
    
    # 9. 移动回初始位置
    print("\n7. 移动回初始位置...")
    back_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 0, "tilt": 0, "speed": 100})
    print(f"   回归初始位置: {back_result.json()}")
    time.sleep(3)
    
    print(f"\n=== 测试完成 ===")
    return success

if __name__ == "__main__":
    test_ai_100deg_move()