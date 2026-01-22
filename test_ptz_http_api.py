#!/usr/bin/env python3
"""
测试PTZ HTTP API功能
"""

import requests
import time

BASE_URL = "http://localhost:8001/api"

def test_ptz_http_api():
    """
    测试PTZ HTTP API功能
    """
    print("=== 测试PTZ HTTP API功能 ===")
    
    # 1. 测试PTZ连接
    print("\n1. 测试PTZ连接...")
    try:
        payload = {
            "protocol": "http",
            "connection_type": "http",
            "base_url": "http://localhost:8001",
            "username": "admin",
            "password": "admin"
        }
        response = requests.post(f"{BASE_URL}/camera/ptz/connect", json=payload, timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   连接PTZ失败: {e}")
    
    # 2. 测试PTZ动作
    print("\n2. 测试PTZ动作...")
    
    test_actions = [
        {"action": "pan_left", "name": "向左转"},
        {"action": "pan_right", "name": "向右转"},
        {"action": "tilt_up", "name": "向上转"},
        {"action": "tilt_down", "name": "向下转"},
        {"action": "stop", "name": "停止"}
    ]
    
    for test_action in test_actions:
        action = test_action["action"]
        name = test_action["name"]
        
        print(f"   测试 {name}...")
        try:
            payload = {
                "action": action,
                "speed": 50
            }
            response = requests.post(f"{BASE_URL}/camera/ptz/action", json=payload, timeout=10)
            result = response.json()
            print(f"     状态码: {response.status_code}")
            print(f"     结果: {result}")
            time.sleep(1)
        except Exception as e:
            print(f"     错误: {e}")
    
    # 3. 获取PTZ状态
    print("\n3. 获取PTZ状态...")
    try:
        response = requests.get(f"{BASE_URL}/camera/ptz/status", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   获取PTZ状态失败: {e}")
    
    # 4. 断开PTZ连接
    print("\n4. 断开PTZ连接...")
    try:
        response = requests.post(f"{BASE_URL}/camera/ptz/disconnect", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   断开PTZ连接失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_ptz_http_api()
