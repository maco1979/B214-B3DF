#!/usr/bin/env python3
"""
检查服务器中正在运行的摄像头实例
"""

import requests

BASE_URL = "http://localhost:8001/api"

print("=== 检查服务器摄像头状态 ===")

# 检查摄像头状态
try:
    response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
    print(f"1. 摄像头API状态响应:")
    print(f"   状态码: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   成功: {data['success']}")
        print(f"   消息: {data['message']}")
        print(f"   数据: {data['data']}")
        is_open = data['data']['is_open']
        print(f"   摄像头是否打开: {'是' if is_open else '否'}")
        if is_open:
            print("\n2. 关闭摄像头...")
            close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
            print(f"   关闭状态码: {close_response.status_code}")
            print(f"   关闭响应: {close_response.json()}")
            
            # 再次检查状态
            check_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
            check_data = check_response.json()
            print(f"\n3. 关闭后状态:")
            print(f"   摄像头是否打开: {'是' if check_data['data']['is_open'] else '否'}")
except Exception as e:
    print(f"请求失败: {e}")
    
print("\n=== 检查完成 ===")