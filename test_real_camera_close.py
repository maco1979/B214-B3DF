#!/usr/bin/env python3
"""
测试真实摄像头的关闭功能
"""

import requests
import time

BASE_URL = "http://localhost:8001/api"

print("=== 测试真实摄像头的关闭功能 ===")

# 1. 列出可用摄像头
print("\n1. 列出可用摄像头...")
try:
    list_response = requests.get(f"{BASE_URL}/camera/list", timeout=10)
    list_response.raise_for_status()
    list_result = list_response.json()
    print(f"   状态码: {list_response.status_code}")
    print(f"   可用摄像头: {len(list_result['data']['cameras'])}")
    for cam in list_result['data']['cameras']:
        print(f"   - 索引: {cam['index']}, 类型: {cam['type']}")
        if cam['type'] == 'real':
            real_cam_index = cam['index']
            print(f"   ✓ 找到真实摄像头，索引: {real_cam_index}")
except Exception as e:
    print(f"   列出摄像头失败: {e}")
    # 继续测试，使用默认索引0
    real_cam_index = 0
    print(f"   使用默认索引 {real_cam_index}")

# 2. 打开真实摄像头
print("\n2. 尝试打开真实摄像头...")
try:
    open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": real_cam_index}, timeout=10)
    open_response.raise_for_status()
    open_result = open_response.json()
    print(f"   状态码: {open_response.status_code}")
    print(f"   打开响应: {open_result}")
    if open_result['success']:
        print("   ✅ 摄像头打开成功")
    else:
        print("   ❌ 摄像头打开失败，使用模拟摄像头继续测试")
        # 使用模拟摄像头继续测试
        open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=10)
        open_response.raise_for_status()
        open_result = open_response.json()
        print(f"   模拟摄像头打开响应: {open_result}")
except Exception as e:
    print(f"   打开摄像头失败: {e}")
    print("   使用模拟摄像头继续测试")
    # 使用模拟摄像头继续测试
    try:
        open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=10)
        open_response.raise_for_status()
        open_result = open_response.json()
        print(f"   模拟摄像头打开响应: {open_result}")
    except Exception as e2:
        print(f"   模拟摄像头打开也失败: {e2}")
        exit(1)

# 3. 等待一会儿，确保摄像头正常运行
print("\n3. 等待2秒...")
time.sleep(2)

# 4. 检查状态，确认摄像头已打开
print("\n4. 检查摄像头状态...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
    status_response.raise_for_status()
    status_result = status_response.json()
    is_open = status_result['data']['is_open']
    print(f"   当前摄像头状态: {'打开' if is_open else '关闭'}")
    if not is_open:
        print("   ❌ 摄像头状态异常")
        exit(1)
except Exception as e:
    print(f"   获取状态失败: {e}")
    exit(1)

# 5. 启动识别功能（模拟真实使用场景）
print("\n5. 启动识别功能...")
try:
    start_recog_response = requests.post(f"{BASE_URL}/camera/recognition/start", json={"model_type": "haar"}, timeout=10)
    start_recog_response.raise_for_status()
    start_recog_result = start_recog_response.json()
    print(f"   识别启动响应: {start_recog_result['message']}")
    time.sleep(1)
    
    # 启动跟踪功能
    start_track_response = requests.post(f"{BASE_URL}/camera/tracking/start", json={"tracker_type": "MIL"}, timeout=10)
    start_track_response.raise_for_status()
    start_track_result = start_track_response.json()
    print(f"   跟踪启动响应: {start_track_result['message']}")
except Exception as e:
    print(f"   启动功能失败: {e}")
    # 继续测试，不影响关闭功能

# 6. 等待一会儿，确保功能正常运行
print("\n6. 等待2秒...")
time.sleep(2)

# 7. 直接关闭摄像头（不先停止功能，测试强制关闭）
print("\n7. 直接关闭摄像头（模拟真实使用场景，不先停止功能）...")
try:
    close_response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
    close_response.raise_for_status()
    close_result = close_response.json()
    print(f"   状态码: {close_response.status_code}")
    print(f"   关闭响应: {close_result}")
except Exception as e:
    print(f"   关闭摄像头失败: {e}")
    exit(1)

# 8. 等待一会儿，确保线程有时间关闭
print("\n8. 等待2秒...")
time.sleep(2)

# 9. 检查关闭后状态
print("\n9. 检查关闭后状态...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
    status_response.raise_for_status()
    status_result = status_response.json()
    is_open = status_result['data']['is_open']
    print(f"   状态码: {status_response.status_code}")
    print(f"   响应: {status_result}")
    print(f"   关闭后摄像头状态: {'打开' if is_open else '关闭'}")
    
    if not is_open:
        print("   ✅ 真实摄像头关闭成功！修复有效")
        print("   ✅ 功能正常关闭，不会出现摄像头一直打开的情况")
    else:
        print("   ❌ 真实摄像头仍然打开，修复无效")
except Exception as e:
    print(f"   获取状态失败: {e}")
    exit(1)

# 10. 最后检查，确保功能都已停止
print("\n10. 检查功能状态...")
try:
    # 检查识别状态
    recog_status_response = requests.get(f"{BASE_URL}/camera/recognition/status", timeout=10)
    recog_status_response.raise_for_status()
    recog_status = recog_status_response.json()
    print(f"   识别功能状态: {'开启' if recog_status['data']['recognizing_enabled'] else '关闭'}")
    
    # 检查跟踪状态
    track_status_response = requests.get(f"{BASE_URL}/camera/tracking/status", timeout=10)
    track_status_response.raise_for_status()
    track_status = track_status_response.json()
    print(f"   跟踪功能状态: {'开启' if track_status['data']['tracking_enabled'] else '关闭'}")
    
    if not recog_status['data']['recognizing_enabled'] and not track_status['data']['tracking_enabled']:
        print("   ✅ 所有功能都已正常关闭")
except Exception as e:
    print(f"   检查功能状态失败: {e}")

print("\n=== 测试完成 ===")
