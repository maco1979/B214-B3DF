#!/usr/bin/env python3
"""使用正确IP的PTZ控制测试"""

import requests
import time

BASE_URL = "http://localhost:8001"

# 使用网络中发现的设备IP
test_ips = [
    "192.168.1.33",  # 网络设备
    "192.168.1.1"    # 路由器IP（作为对比）
]

def test_ptz_with_ip(ip_address):
    """使用指定IP测试PTZ控制"""
    print(f"\n=== 使用IP {ip_address} 测试PTZ控制 ===")
    
    # 构建PTZ配置
    ptz_config = {
        "protocol": "http",
        "connection_type": "http",
        "base_url": f"http://{ip_address}",
        "username": "admin",
        "password": "admin"
    }
    
    # 1. 检查摄像头是否已打开
    print(f"1. 检查摄像头状态")
    status = requests.get(f"{BASE_URL}/api/camera/status").json()
    if not status.get("data", {}).get("is_open", False):
        print(f"   摄像头未打开，正在打开...")
        open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
        print(f"   打开结果: {open_result}")
    
    # 2. 断开现有PTZ连接
    print(f"2. 断开现有PTZ连接")
    try:
        requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    except Exception as e:
        print(f"   断开连接错误: {e}")
    
    # 3. 连接PTZ
    print(f"3. 连接PTZ")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config).json()
    print(f"   连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print(f"   ❌ PTZ连接失败")
        return False
    
    # 4. 测试PTZ移动
    print(f"4. 测试PTZ移动")
    move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 60.0, "tilt": 30.0, "speed": 70})
    
    print(f"   移动命令结果: {move_result.json()}")
    
    # 5. 等待移动完成
    print(f"5. 等待移动完成 (3秒)")
    time.sleep(3)
    
    # 6. 检查状态
    print(f"6. 检查PTZ状态")
    status_result = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    print(f"   状态结果: {status_result}")
    
    # 7. 断开连接
    print(f"7. 断开PTZ连接")
    requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    
    return True

def main():
    """主函数"""
    print("=== PTZ控制测试 ===")
    print("使用不同IP地址测试PTZ控制功能")
    
    for ip in test_ips:
        test_ptz_with_ip(ip)
    
    print(f"\n=== 测试完成 ===")
    print(f"\n💡 提示:")
    print(f"   • 网络设备IP: 192.168.1.33")
    print(f"   • 路由器IP: 192.168.1.1")
    print(f"   • 建议使用网络设备IP进行PTZ控制测试")

if __name__ == "__main__":
    main()