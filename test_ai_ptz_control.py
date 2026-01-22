#!/usr/bin/env python3
"""AI PTZ控制测试脚本"""

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

def test_ai_ptz_control():
    """测试AI PTZ控制"""
    print("=== AI PTZ控制测试 ===")
    
    # 1. 断开现有连接
    print("1. 断开现有连接...")
    try:
        disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
        print(f"   断开结果: {disconnect_result}")
    except Exception as e:
        print(f"   断开连接错误: {e}")
    
    # 2. 连接PTZ
    print("2. 连接PTZ...")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=PTZ_CONFIG).json()
    print(f"   连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print("   ❌ PTZ连接失败，测试终止")
        return False
    
    # 3. 模拟AI决策过程
    print("3. 模拟AI决策过程...")
    
    # 模拟AI检测到目标并计算需要移动的位置
    def simulate_ai_decision(current_pan, current_tilt):
        """模拟AI决策"""
        # 随机生成目标位置（模拟AI检测到不同位置的目标）
        target_pan = random.uniform(-180, 180)
        target_tilt = random.uniform(-90, 90)
        
        print(f"   AI决策: 从当前位置(pan={current_pan:.1f}, tilt={current_tilt:.1f}) 移动到目标位置(pan={target_pan:.1f}, tilt={target_tilt:.1f})")
        return target_pan, target_tilt
    
    # 获取初始状态
    status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    current_pan = status["data"]["position"]["pan"]
    current_tilt = status["data"]["position"]["tilt"]
    
    # 模拟5次AI决策和PTZ控制
    for i in range(5):
        print(f"\n4. AI决策轮次 {i+1}/5:")
        
        # AI决策
        target_pan, target_tilt = simulate_ai_decision(current_pan, current_tilt)
        
        # 调用PTZ控制API
        print(f"   调用PTZ控制API移动到目标位置...")
        result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": target_pan, "tilt": target_tilt, "speed": 80})
        
        print(f"   API响应: {result.json()}")
        
        # 等待摄像头移动完成
        time.sleep(4)
        
        # 更新当前位置
        status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
        current_pan = status["data"]["position"]["pan"]
        current_tilt = status["data"]["position"]["tilt"]
        
        print(f"   移动完成，当前位置: pan={current_pan:.1f}, tilt={current_tilt:.1f}")
        
        # 检查是否达到目标位置（允许±5°误差）
        if abs(current_pan - target_pan) < 5 and abs(current_tilt - target_tilt) < 5:
            print(f"   ✅ AI控制成功，位置误差在允许范围内")
        else:
            print(f"   ⚠️  位置误差较大: pan误差={abs(current_pan - target_pan):.1f}°, tilt误差={abs(current_tilt - target_tilt):.1f}°")
    
    # 5. 断开连接
    print("\n5. 断开连接...")
    disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print(f"   断开结果: {disconnect_result}")
    
    return True

if __name__ == "__main__":
    test_ai_ptz_control()