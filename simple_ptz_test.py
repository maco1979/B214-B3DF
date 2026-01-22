#!/usr/bin/env python3
"""简单PTZ控制测试脚本"""

import requests
import time

# =======================================
# 配置区域 - 请在此填写真实的摄像头信息
# =======================================
BASE_URL = "http://localhost:8001"

PTZ_CONFIG = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://192.168.1.1",  # 替换为摄像头真实IP
    "username": "admin",                 # 替换为摄像头真实用户名
    "password": "admin"                  # 替换为摄像头真实密码
}

# 测试参数
TEST_PAN = 60.0       # 测试水平移动角度（-180到180）
TEST_TILT = 30.0       # 测试垂直移动角度（-90到90）
TEST_SPEED = 70        # 测试移动速度（0到100）

# =======================================
# 测试代码 - 无需修改
# =======================================

def test_ptz_simple():
    """简单PTZ控制测试"""
    print("=== 简单PTZ控制测试 ===")
    print(f"配置信息:")
    print(f"  摄像头IP: {PTZ_CONFIG['base_url']}")
    print(f"  用户名: {PTZ_CONFIG['username']}")
    print(f"  密码: {'*' * len(PTZ_CONFIG['password'])}")
    print(f"  测试参数: pan={TEST_PAN}°, tilt={TEST_TILT}°, speed={TEST_SPEED}%")
    print()
    
    # 1. 打开摄像头
    print("1. 打开摄像头...")
    open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
    print(f"   结果: {open_result}")
    
    if not open_result.get("success"):
        print("   ❌ 摄像头打开失败")
        return False
    
    # 2. 断开现有PTZ连接
    print("2. 断开现有PTZ连接...")
    try:
        disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
        print(f"   结果: {disconnect_result}")
    except Exception as e:
        print(f"   警告: 断开连接错误 - {e}")
    
    # 3. 连接PTZ
    print("3. 连接PTZ...")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=PTZ_CONFIG).json()
    print(f"   结果: {connect_result}")
    
    if not connect_result.get("success"):
        print("   ❌ PTZ连接失败")
        return False
    
    # 4. 移动到初始位置
    print("4. 移动到初始位置 (0°, 0°)...")
    init_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 0, "tilt": 0, "speed": 100})
    print(f"   结果: {init_result.json()}")
    time.sleep(2)
    
    # 5. 获取初始位置
    print("5. 获取初始位置...")
    init_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    init_pan = init_status["data"]["position"]["pan"]
    init_tilt = init_status["data"]["position"]["tilt"]
    print(f"   初始位置: pan={init_pan:.1f}°, tilt={init_tilt:.1f}°")
    
    # 6. 测试PTZ移动
    print(f"6. 测试移动到位置 (pan={TEST_PAN}°, tilt={TEST_TILT}°)...")
    move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": TEST_PAN, "tilt": TEST_TILT, "speed": TEST_SPEED})
    
    print(f"   移动命令结果: {move_result.json()}")
    
    # 7. 等待移动完成
    print(f"7. 等待移动完成 ({5}秒)...")
    time.sleep(5)
    
    # 8. 获取移动后位置
    print("8. 获取移动后位置...")
    final_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    final_pan = final_status["data"]["position"]["pan"]
    final_tilt = final_status["data"]["position"]["tilt"]
    print(f"   移动后位置: pan={final_pan:.1f}°, tilt={final_tilt:.1f}°")
    
    # 9. 计算移动角度
    pan_diff = abs(final_pan - init_pan)
    tilt_diff = abs(final_tilt - init_tilt)
    
    print(f"9. 计算移动角度...")
    print(f"   水平移动: {pan_diff:.1f}° (目标: {abs(TEST_PAN - init_pan):.1f}°)")
    print(f"   垂直移动: {tilt_diff:.1f}° (目标: {abs(TEST_TILT - init_tilt):.1f}°)")
    
    # 10. 测试结果
    print("\n=== 测试结果 ===")
    if pan_diff > 5 or tilt_diff > 5:
        print(f"✅ 测试成功！云台发生了明显移动")
        print(f"   AI能够控制摄像头移动到指定位置")
        success = True
    else:
        print(f"❌ 测试失败！云台没有明显移动")
        print(f"   可能的原因:")
        print(f"   1. 摄像头IP地址错误")
        print(f"   2. 用户名或密码错误")
        print(f"   3. 摄像头不支持该PTZ协议")
        print(f"   4. 网络连接问题")
        success = False
    
    # 11. 清理资源
    print("\n10. 清理资源...")
    
    # 移动回初始位置
    print(f"   移动回初始位置...")
    requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                 json={"pan": 0, "tilt": 0, "speed": 100})
    time.sleep(2)
    
    # 断开PTZ连接
    print(f"   断开PTZ连接...")
    requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    
    # 关闭摄像头
    print(f"   关闭摄像头...")
    requests.post(f"{BASE_URL}/api/camera/close").json()
    
    print(f"\n=== 测试完成 ===")
    return success

if __name__ == "__main__":
    test_ptz_simple()