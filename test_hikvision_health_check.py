#!/usr/bin/env python3
"""海康威视摄像头健康检查脚本"""

import requests
import time

def check_hikvision_camera(ip_address="192.168.1.1"):
    """检查海康威视摄像头是否可达及API状态"""
    print(f"=== 海康威视摄像头健康检查 ({ip_address}) ===")
    
    # 1. 测试网络连接
    print("1. 测试网络连接...")
    try:
        response = requests.get(f"http://{ip_address}", timeout=5)
        print(f"   ✅ 网络可达，状态码: {response.status_code}")
        print(f"   📝 响应头: {dict(response.headers)}")
    except Exception as e:
        print(f"   ❌ 网络连接失败: {e}")
        return False
    
    # 2. 检查海康威视ISAPI状态
    print("2. 检查ISAPI接口...")
    try:
        isapi_url = f"http://{ip_address}/ISAPI"
        response = requests.get(isapi_url, timeout=5)
        print(f"   ✅ ISAPI可达，状态码: {response.status_code}")
        print(f"   📝 响应内容: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ ISAPI接口失败: {e}")
    
    # 3. 检查PTZ控制接口
    print("3. 检查PTZ控制接口...")
    try:
        ptz_url = f"http://{ip_address}/ISAPI/PTZCtrl/channels/1/status"
        response = requests.get(ptz_url, auth=("admin", "admin"), timeout=5)
        print(f"   📡 PTZ状态接口响应: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ PTZ接口正常，响应: {response.text[:300]}...")
        else:
            print(f"   ❌ PTZ接口失败，响应: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ PTZ接口请求失败: {e}")
    
    # 4. 测试PTZ控制命令
    print("4. 测试PTZ控制命令...")
    try:
        # 测试停止命令
        stop_url = f"http://{ip_address}/ISAPI/PTZCtrl/channels/1/continuous?PanLeft=0&PanRight=0&TiltUp=0&TiltDown=0&ZoomIn=0&ZoomOut=0"
        response = requests.get(stop_url, auth=("admin", "admin"), timeout=5)
        print(f"   🛑 停止命令响应: {response.status_code}")
        print(f"   📝 响应: {response.text[:200]}...")
    except Exception as e:
        print(f"   ❌ 停止命令失败: {e}")
    
    print("\n=== 检查完成 ===")
    return True

import sys

if __name__ == "__main__":
    # 检查命令行参数
    ip_address = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.1"
    check_hikvision_camera(ip_address)
    
    # 提示用户可能的问题
    if ip_address == "192.168.1.1":
        print("\n📌 注意：192.168.1.1通常是路由器IP，不是摄像头IP！")
        print("建议测试海康威视常见默认IP：")
        print("- python test_hikvision_health_check.py 192.168.1.64")
        print("- python test_hikvision_health_check.py 192.168.1.100")
    print("\n您可以通过以下方式获取摄像头IP：")
    print("1. 查看摄像头标签上的默认IP")
    print("2. 使用海康威视SADP工具扫描网络")
    print("3. 登录路由器查看设备列表")
    print("\n修改方式：")
    print("- 修改测试脚本中的PTZ_CONFIG['base_url']")
    print("- 或直接运行: python test_hikvision_health_check.py [正确IP]")
