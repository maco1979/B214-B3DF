#!/usr/bin/env python3
"""
简单自动跟踪测试脚本
直接测试人脸识别和跟踪的基本功能
"""

import requests
import time

BASE_URL = "http://localhost:8001/api"

def test_basic_flow():
    """测试基本流程"""
    print("=== 简单自动跟踪测试 ===")
    
    # 1. 关闭摄像头（确保初始状态）
    print("\n1. 关闭摄像头...")
    try:
        r = requests.post(f"{BASE_URL}/camera/close", timeout=5)
        print(f"   结果: {r.json()}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 2. 打开摄像头
    print("\n2. 打开摄像头...")
    try:
        r = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=5)
        print(f"   结果: {r.json()}")
        if not r.json()["success"]:
            print("   摄像头打开失败，测试终止")
            return
    except Exception as e:
        print(f"   错误: {e}")
        return
    
    # 3. 获取摄像头状态
    print("\n3. 获取摄像头状态...")
    try:
        r = requests.get(f"{BASE_URL}/camera/status", timeout=5)
        print(f"   结果: {r.json()}")
    except Exception as e:
        print(f"   错误: {e}")
    
    # 4. 启动人脸识别
    print("\n4. 启动人脸识别...")
    try:
        r = requests.post(f"{BASE_URL}/camera/recognition/start", json={"model_type": "haar"}, timeout=5)
        print(f"   结果: {r.json()}")
        if not r.json()["success"]:
            print("   人脸识别启动失败，测试终止")
            requests.post(f"{BASE_URL}/camera/close", timeout=5)
            return
    except Exception as e:
        print(f"   错误: {e}")
        requests.post(f"{BASE_URL}/camera/close", timeout=5)
        return
    
    # 5. 循环检查人脸识别结果
    print("\n5. 检查人脸识别结果...")
    face_detected = False
    for i in range(5):
        try:
            r = requests.get(f"{BASE_URL}/camera/recognition/status", timeout=5)
            result = r.json()
            print(f"   第{i+1}次检查: {result}")
            
            if result["data"]["recognized_objects_count"] > 0:
                face_detected = True
                print("   ✅ 检测到人脸！")
                break
        except Exception as e:
            print(f"   错误: {e}")
        time.sleep(1)
    
    if not face_detected:
        print("   ❌ 未检测到人脸，测试终止")
        requests.post(f"{BASE_URL}/camera/recognition/stop", timeout=5)
        requests.post(f"{BASE_URL}/camera/close", timeout=5)
        return
    
    # 6. 手动启动跟踪
    print("\n6. 手动启动跟踪...")
    try:
        # 使用检测到的人脸作为初始边界框
        r = requests.get(f"{BASE_URL}/camera/recognition/status", timeout=5)
        face = r.json()["data"]["recognized_objects"][0]
        initial_bbox = face["bbox"]
        
        r = requests.post(f"{BASE_URL}/camera/tracking/start", 
                         json={"tracker_type": "MIL", "initial_bbox": initial_bbox}, timeout=5)
        print(f"   结果: {r.json()}")
        if not r.json()["success"]:
            print("   跟踪启动失败，测试终止")
            requests.post(f"{BASE_URL}/camera/recognition/stop", timeout=5)
            requests.post(f"{BASE_URL}/camera/close", timeout=5)
            return
    except Exception as e:
        print(f"   错误: {e}")
        requests.post(f"{BASE_URL}/camera/recognition/stop", timeout=5)
        requests.post(f"{BASE_URL}/camera/close", timeout=5)
        return
    
    # 7. 检查跟踪状态
    print("\n7. 检查跟踪状态...")
    for i in range(10):
        try:
            r = requests.get(f"{BASE_URL}/camera/tracking/status", timeout=5)
            result = r.json()
            print(f"   第{i+1}秒: {result}")
        except Exception as e:
            print(f"   错误: {e}")
        time.sleep(1)
    
    # 8. 清理资源
    print("\n8. 清理资源...")
    try:
        r = requests.post(f"{BASE_URL}/camera/tracking/stop", timeout=5)
        print(f"   停止跟踪: {r.json()}")
        
        r = requests.post(f"{BASE_URL}/camera/recognition/stop", timeout=5)
        print(f"   停止人脸识别: {r.json()}")
        
        r = requests.post(f"{BASE_URL}/camera/close", timeout=5)
        print(f"   关闭摄像头: {r.json()}")
    except Exception as e:
        print(f"   错误: {e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_basic_flow()
