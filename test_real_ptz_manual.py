#!/usr/bin/env python3
"""
真实PTZ硬件手动测试脚本
直接发送PTZ命令到物理摄像头，验证物理控制
"""

import requests
import time

# 配置
BASE_URL = "http://localhost:8001"

# 海康威视摄像头配置
HIKVISION_CONFIG = {
    "base_url": "http://192.168.1.64",  # 摄像头IP地址
    "username": "admin",  # 摄像头用户名
    "password": "admin",  # 摄像头密码
    "protocol": "http",
    "connection_type": "http"
}

# 测试配置
TEST_CONFIG = {
    "speed": 50,          # PTZ动作速度
    "action_duration": 5,  # 每个动作持续时间（秒）
    "pause_duration": 2    # 动作间隔时间（秒）
}

def connect_ptz():
    """连接PTZ摄像头"""
    print("连接到PTZ摄像头...")
    
    # 断开现有连接
    disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print(f"断开现有连接: {disconnect_result}")
    
    # 连接新PTZ
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=HIKVISION_CONFIG).json()
    print(f"连接结果: {connect_result}")
    
    return connect_result.get("success", False)

def get_ptz_status():
    """获取PTZ状态"""
    status_result = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    if status_result.get("success"):
        position = status_result["data"]["position"]
        print(f"当前PTZ位置: pan={position['pan']:.1f}°, tilt={position['tilt']:.1f}°, zoom={position['zoom']:.1f}x")
        return position
    else:
        print(f"获取PTZ状态失败: {status_result}")
        return None

def send_ptz_action(action):
    """发送PTZ动作命令"""
    print(f"\n发送动作: {action}")
    
    request_data = {
        "action": action,
        "speed": TEST_CONFIG["speed"]
    }
    
    result = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=request_data).json()
    print(f"动作结果: {result}")
    
    return result.get("success", False)

def test_ptz_movement():
    """测试PTZ运动"""
    print("=== 真实PTZ硬件手动测试 ===")
    print("\n配置信息:")
    print(f"- 摄像头IP: {HIKVISION_CONFIG['base_url']}")
    print(f"- 动作速度: {TEST_CONFIG['speed']}%")
    print(f"- 动作持续时间: {TEST_CONFIG['action_duration']}秒")
    print(f"- 动作间隔时间: {TEST_CONFIG['pause_duration']}秒")
    
    # 0. 检查并开启摄像头
    print("\n=== 检查摄像头状态 ===")
    camera_status = requests.get(f"{BASE_URL}/api/camera/status").json()
    print(f"摄像头状态: {camera_status}")
    
    if not camera_status.get("success") or not camera_status.get("data", {}).get("is_open"):
        print("\n摄像头未开启，正在开启摄像头...")
        open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
        print(f"摄像头开启结果: {open_result}")
        
        # 再次检查摄像头状态
        camera_status = requests.get(f"{BASE_URL}/api/camera/status").json()
        if not camera_status.get("success") or not camera_status.get("data", {}).get("is_open"):
            print("❌ 摄像头开启失败，退出测试")
            return False
    
    print("✅ 摄像头已成功开启")
    
    # 1. 连接PTZ
    if not connect_ptz():
        print("PTZ连接失败，退出测试")
        return False
    
    # 2. 获取初始状态
    print("\n=== 初始状态 ===")
    initial_position = get_ptz_status()
    if not initial_position:
        return False
    
    # 3. 测试水平运动
    print("\n=== 测试水平运动 ===")
    
    # 向右转动
    print(f"\n1. 向右转动 {TEST_CONFIG['action_duration']} 秒...")
    if send_ptz_action("pan_right"):
        time.sleep(TEST_CONFIG['action_duration'])
        send_ptz_action("stop")
        time.sleep(TEST_CONFIG['pause_duration'])
        get_ptz_status()
    else:
        print("向右转动命令发送失败")
    
    # 向左转动
    print(f"\n2. 向左转动 {TEST_CONFIG['action_duration']} 秒...")
    if send_ptz_action("pan_left"):
        time.sleep(TEST_CONFIG['action_duration'])
        send_ptz_action("stop")
        time.sleep(TEST_CONFIG['pause_duration'])
        get_ptz_status()
    else:
        print("向左转动命令发送失败")
    
    # 4. 测试垂直运动
    print("\n=== 测试垂直运动 ===")
    
    # 向上转动
    print(f"\n3. 向上转动 {TEST_CONFIG['action_duration']} 秒...")
    if send_ptz_action("tilt_up"):
        time.sleep(TEST_CONFIG['action_duration'])
        send_ptz_action("stop")
        time.sleep(TEST_CONFIG['pause_duration'])
        get_ptz_status()
    else:
        print("向上转动命令发送失败")
    
    # 向下转动
    print(f"\n4. 向下转动 {TEST_CONFIG['action_duration']} 秒...")
    if send_ptz_action("tilt_down"):
        time.sleep(TEST_CONFIG['action_duration'])
        send_ptz_action("stop")
        time.sleep(TEST_CONFIG['pause_duration'])
        get_ptz_status()
    else:
        print("向下转动命令发送失败")
    
    # 5. 测试大角度运动
    print("\n=== 测试大角度运动 ===")
    
    # 持续向左转动10秒（大角度）
    print(f"\n5. 持续向左转动 10 秒...")
    if send_ptz_action("pan_left"):
        time.sleep(10)
        send_ptz_action("stop")
        time.sleep(TEST_CONFIG['pause_duration'])
        get_ptz_status()
    else:
        print("大角度向左转动命令发送失败")
    
    # 6. 恢复初始位置（如果需要）
    print("\n=== 测试完成 ===")
    
    # 获取最终状态
    final_position = get_ptz_status()
    
    if initial_position and final_position:
        # 计算位置变化
        pan_change = abs(final_position['pan'] - initial_position['pan'])
        tilt_change = abs(final_position['tilt'] - initial_position['tilt'])
        
        print(f"\n=== 测试结果 ===")
        print(f"初始位置: pan={initial_position['pan']:.1f}°, tilt={initial_position['tilt']:.1f}°")
        print(f"最终位置: pan={final_position['pan']:.1f}°, tilt={final_position['tilt']:.1f}°")
        print(f"位置变化: pan={pan_change:.1f}°, tilt={tilt_change:.1f}°")
        
        if pan_change > 5.0 or tilt_change > 5.0:
            print("✅ PTZ物理控制验证成功！")
            print("   位置发生了明显变化，说明物理控制有效")
        else:
            print("❌ PTZ物理控制验证失败")
            print("   位置变化不明显，可能的原因：")
            print("   1. 摄像头网络连接问题")
            print("   2. 摄像头用户名/密码错误")
            print("   3. 摄像头HTTP API未启用")
            print("   4. 摄像头PTZ功能未启用")
            print("   5. 摄像头与电脑不在同一网段")
            print("   6. 摄像头硬件故障")
    
    return True

def main():
    """主函数"""
    try:
        test_ptz_movement()
    except KeyboardInterrupt:
        print("\n\n=== 测试已中断 ===")
    except Exception as e:
        print(f"\n\n=== 测试发生错误: {e} ===")
        import traceback
        traceback.print_exc()
    finally:
        # 发送停止命令
        print("\n发送最终停止命令...")
        requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
        print("测试结束")

if __name__ == "__main__":
    main()
