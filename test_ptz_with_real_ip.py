#!/usr/bin/env python3
"""使用真实IP的PTZ控制测试"""

import requests
import time

BASE_URL = "http://localhost:8001"

# 测试不同的PTZ配置
PTZ_CONFIGS = [
    {
        "name": "当前配置（路由器IP）",
        "config": {
            "protocol": "http",
            "connection_type": "http",
            "base_url": "http://192.168.1.1",
            "username": "admin",
            "password": "admin"
        }
    },
    {
        "name": "网络设备1（可能是摄像头）",
        "config": {
            "protocol": "http",
            "connection_type": "http",
            "base_url": "http://192.168.1.33",
            "username": "admin",
            "password": "admin"
        }
    }
]

def test_ptz_config(config_name, ptz_config):
    """测试单个PTZ配置"""
    print(f"\n=== 测试 {config_name} ===")
    
    # 断开现有连接
    try:
        requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    except Exception as e:
        print(f"   断开连接错误: {e}")
    
    # 连接PTZ
    print(f"1. 连接PTZ...")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config).json()
    print(f"   连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print(f"   ❌ PTZ连接失败")
        return False
    
    # 测试PTZ移动
    print(f"2. 测试PTZ移动...")
    move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 30, "tilt": 15, "speed": 50})
    
    print(f"   移动命令响应: {move_result.json()}")
    
    # 等待移动完成
    time.sleep(2)
    
    # 获取当前状态
    print(f"3. 获取当前状态...")
    status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    print(f"   当前状态: {status}")
    
    return True

def main():
    """主测试函数"""
    print("=== 多IP PTZ控制测试 ===")
    
    # 确保摄像头已打开
    print("1. 确保摄像头已打开...")
    open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
    print(f"   摄像头状态: {open_result}")
    
    if not open_result.get("success"):
        print("   ❌ 摄像头打开失败，测试终止")
        return False
    
    # 测试所有PTZ配置
    for config_info in PTZ_CONFIGS:
        test_ptz_config(config_info["name"], config_info["config"])
    
    # 清理资源
    print("\n=== 清理资源 ===")
    requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print("✅ 测试完成")

if __name__ == "__main__":
    main()