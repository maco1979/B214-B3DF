#!/usr/bin/env python3
"""
测试真实摄像头的打开和关闭功能
"""

import requests
import time

BASE_URL = "http://localhost:8001/api"

def test_real_camera():
    """
    测试真实摄像头的打开和关闭功能
    """
    print("=== 测试真实摄像头功能 ===")
    
    # 1. 先关闭摄像头
    print("\n1. 关闭摄像头...")
    try:
        response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
        print(f"   状态码: {response.status_code}")
        print(f"   响应: {response.json()}")
    except Exception as e:
        print(f"   关闭摄像头失败: {e}")
    time.sleep(1)
    
    # 2. 测试真实摄像头打开（使用默认索引0）
    print("\n2. 打开真实摄像头（索引0）...")
    try:
        response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 0}, timeout=10)
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   响应: {result}")
        
        if result["success"]:
            print("   ✅ 真实摄像头打开成功")
            print(f"   摄像头类型: {result['data']['camera_info']['type']}")
            
            # 3. 等待摄像头初始化
            print("\n3. 等待摄像头初始化...")
            time.sleep(2)
            
            # 4. 检查摄像头状态
            print("\n4. 检查摄像头状态...")
            response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
            status_result = response.json()
            print(f"   状态码: {response.status_code}")
            print(f"   响应: {status_result}")
            print(f"   摄像头是否打开: {'是' if status_result['data']['is_open'] else '否'}")
            
            # 5. 获取当前帧（验证摄像头是否真的工作）
            print("\n5. 获取当前帧...")
            response = requests.get(f"{BASE_URL}/camera/frame", timeout=10)
            print(f"   状态码: {response.status_code}")
            frame_result = response.json()
            print(f"   帧获取结果: {'成功' if frame_result['success'] else '失败'}")
            
            # 6. 关闭摄像头
            print("\n6. 关闭摄像头...")
            response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
            print(f"   状态码: {response.status_code}")
            close_result = response.json()
            print(f"   响应: {close_result}")
            
            # 7. 再次检查状态
            print("\n7. 再次检查摄像头状态...")
            response = requests.get(f"{BASE_URL}/camera/status", timeout=10)
            status_result = response.json()
            print(f"   状态码: {response.status_code}")
            print(f"   摄像头是否打开: {'是' if status_result['data']['is_open'] else '否'}")
        else:
            print("   ❌ 真实摄像头打开失败")
            print(f"   错误信息: {result['message']}")
            
            # 8. 测试另一个真实摄像头索引
            print("\n8. 尝试打开真实摄像头（索引1）...")
            response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 1}, timeout=10)
            print(f"   状态码: {response.status_code}")
            result = response.json()
            print(f"   响应: {result}")
            if result["success"]:
                print("   ✅ 真实摄像头（索引1）打开成功")
                print(f"   摄像头类型: {result['data']['camera_info']['type']}")
                
                # 关闭摄像头
                response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
                print(f"   关闭结果: {response.json()}")
    except Exception as e:
        print(f"   测试真实摄像头失败: {e}")
    
    # 9. 测试模拟摄像头打开（明确指定999）
    print("\n9. 测试模拟摄像头打开（索引999）...")
    try:
        response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=10)
        print(f"   状态码: {response.status_code}")
        result = response.json()
        print(f"   响应: {result}")
        if result["success"]:
            print("   ✅ 模拟摄像头打开成功")
            print(f"   摄像头类型: {result['data']['camera_info']['type']}")
            
            # 关闭模拟摄像头
            response = requests.post(f"{BASE_URL}/camera/close", timeout=10)
            print(f"   关闭结果: {response.json()}")
    except Exception as e:
        print(f"   测试模拟摄像头失败: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_real_camera()
