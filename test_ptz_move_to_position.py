#!/usr/bin/env python3
"""测试move_to_position方法的PTZ控制脚本"""

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

def test_move_to_position():
    """测试move_to_position方法"""
    print("=== 测试move_to_position方法 ===")
    
    # 1. 断开现有连接
    print("1. 断开现有连接...")
    disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print(f"   断开结果: {disconnect_result}")
    
    # 2. 连接PTZ
    print("2. 连接PTZ...")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=PTZ_CONFIG).json()
    print(f"   连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print("   ❌ PTZ连接失败，测试终止")
        return False
    
    # 3. 测试不同位置的move_to_position
    test_positions = [
        ("初始位置", 0, 0, 1.0),
        ("向右120度", 120, 0, 1.0),
        ("向左120度", -120, 0, 1.0),
        ("向上60度", 0, 60, 1.0),
        ("向下60度", 0, -60, 1.0),
        ("复合位置", 90, 45, 1.5)
    ]
    
    for desc, pan, tilt, zoom in test_positions:
        print(f"3. 测试{desc} (pan={pan}, tilt={tilt}, zoom={zoom})...")
        
        # 调用move_to_position API
        result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": pan, "tilt": tilt, "zoom": zoom, "speed": 70})
        
        print(f"   请求结果: {result.json()}")
        time.sleep(3)  # 等待摄像头移动完成
        
        # 获取当前状态
        status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
        print(f"   当前状态: {status}")
        time.sleep(1)
    
    # 4. 断开连接
    print("4. 断开连接...")
    disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print(f"   断开结果: {disconnect_result}")
    
    return True

if __name__ == "__main__":
    test_move_to_position()