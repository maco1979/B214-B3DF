#!/usr/bin/env python3
"""
直接测试关闭摄像头功能
"""

import requests
import time

BASE_URL = "http://localhost:8001/api"

print("=== 直接测试关闭摄像头功能 ===")

# 1. 检查当前状态
print("\n1. 检查当前摄像头状态...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
    status_response.raise_for_status()
    status_result = status_response.json()
    print(f"   状态码: {status_response.status_code}")
    print(f"   响应: {status_result}")
    is_open = status_result['data']['is_open']
    print(f"   当前摄像头状态: {'打开' if is_open else '关闭'}")
except Exception as e:
    print(f"   获取状态失败: {e}")
    exit(1)

# 2. 尝试关闭摄像头
print("\n2. 尝试关闭摄像头...")
try:
    close_response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
    close_response.raise_for_status()
    close_result = close_response.json()
    print(f"   状态码: {close_response.status_code}")
    print(f"   关闭响应: {close_result}")
except Exception as e:
    print(f"   关闭摄像头失败: {e}")
    exit(1)

# 3. 等待一会儿，确保线程有时间关闭
print("\n3. 等待1秒...")
time.sleep(1)

# 4. 再次检查状态
print("\n4. 再次检查摄像头状态...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
    status_response.raise_for_status()
    status_result = status_response.json()
    print(f"   状态码: {status_response.status_code}")
    print(f"   响应: {status_result}")
    is_open = status_result['data']['is_open']
    print(f"   关闭后摄像头状态: {'打开' if is_open else '关闭'}")
    
    if not is_open:
        print("   ✅ 修复成功！摄像头已成功关闭")
    else:
        print("   ❌ 修复失败！摄像头仍然打开")
except Exception as e:
    print(f"   获取状态失败: {e}")
    exit(1)

# 5. 测试重新打开后关闭
print("\n5. 测试重新打开后关闭...")
try:
    # 打开摄像头
    open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=10)
    open_response.raise_for_status()
    open_result = open_response.json()
    print(f"   打开摄像头: {open_result['message']}")
    
    # 等待1秒
    time.sleep(1)
    
    # 再次关闭
    close_response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
    close_response.raise_for_status()
    close_result = close_response.json()
    print(f"   再次关闭摄像头: {close_result['message']}")
    
    # 等待1秒
    time.sleep(1)
    
    # 检查最终状态
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
    status_result = status_response.json()
    is_open = status_result['data']['is_open']
    print(f"   最终摄像头状态: {'打开' if is_open else '关闭'}")
    
    if not is_open:
        print("   ✅ 完整测试通过！摄像头可以正常关闭")
    else:
        print("   ❌ 完整测试失败！摄像头仍然打开")
        
except Exception as e:
    print(f"   测试失败: {e}")
    exit(1)

print("\n=== 测试完成 ===")
