#!/usr/bin/env python3
"""
简单测试摄像头打开和关闭功能
"""

import requests
import time

BASE_URL = "http://localhost:8001/api"

def test_simple_camera():
    """
    简单测试摄像头打开和关闭功能
    """
    print("=== 简单测试摄像头功能 ===")
    
    # 1. 检查服务器状态
    print("\n1. 检查服务器状态...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
        print("   ✅ 服务器正常运行")
    except Exception as e:
        print(f"   服务器连接失败: {e}")
        return False
    
    # 2. 打开摄像头
    print("\n2. 打开摄像头...")
    try:
        response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=5)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
        return response.json()["success"]
    except Exception as e:
        print(f"   打开摄像头失败: {e}")
        return False

if __name__ == "__main__":
    test_simple_camera()
