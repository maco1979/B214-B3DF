#!/usr/bin/env python3
"""
海康威视摄像头PTZ控制测试脚本
直接连接到海康威视摄像头并发送PTZ命令，验证物理控制
"""

import requests
import time

# 海康威视摄像头配置
# 请根据实际情况修改以下配置
HIKVISION_CONFIG = {
    "base_url": "http://192.168.1.64",  # 摄像头IP地址
    "username": "admin",  # 摄像头用户名
    "password": "admin",  # 摄像头密码
    "channel": 1  # 通道号，通常为1
}

# 海康威视PTZ控制API路径
PTZ_API_PATH = f"/ISAPI/PTZCtrl/channels/{HIKVISION_CONFIG['channel']}/continuous"

# PTZ动作列表
PTZ_ACTIONS = {
    "pan_left": "PanLeft",
    "pan_right": "PanRight",
    "tilt_up": "TiltUp",
    "tilt_down": "TiltDown",
    "zoom_in": "ZoomIn",
    "zoom_out": "ZoomOut"
}

def send_ptz_command(action, speed=50):
    """
    发送PTZ命令到海康威视摄像头
    
    Args:
        action: PTZ动作 (pan_left, pan_right, tilt_up, tilt_down, zoom_in, zoom_out)
        speed: 动作速度 (0-100)
    
    Returns:
        dict: 命令执行结果
    """
    try:
        # 构建请求URL
        full_url = f"{HIKVISION_CONFIG['base_url']}{PTZ_API_PATH}"
        
        # 构建PTZ命令参数
        params = {
            "PanLeft": 0,
            "PanRight": 0,
            "TiltUp": 0,
            "TiltDown": 0,
            "ZoomIn": 0,
            "ZoomOut": 0
        }
        
        # 设置动作速度
        if action in PTZ_ACTIONS:
            params[PTZ_ACTIONS[action]] = speed
        else:
            return {"success": False, "message": f"不支持的动作: {action}"}
        
        print(f"发送PTZ命令: {action} @ {speed}%")
        print(f"请求URL: {full_url}")
        print(f"请求参数: {params}")
        
        # 发送HTTP GET请求
        response = requests.get(
            full_url,
            auth=(HIKVISION_CONFIG['username'], HIKVISION_CONFIG['password']),
            params=params,
            timeout=5
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            return {"success": True, "message": f"PTZ命令成功执行: {action}"}
        else:
            return {"success": False, "message": f"PTZ命令执行失败: {response.status_code} - {response.text}"}
    
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": f"无法连接到摄像头: {HIKVISION_CONFIG['base_url']}"}
    except requests.exceptions.Timeout:
        return {"success": False, "message": f"请求超时: {HIKVISION_CONFIG['base_url']}"}
    except Exception as e:
        return {"success": False, "message": f"请求异常: {str(e)}"}

def test_ptz_control():
    """
    测试海康威视摄像头PTZ控制
    """
    print("=== 海康威视摄像头PTZ控制测试 ===")
    print(f"摄像头IP: {HIKVISION_CONFIG['base_url']}")
    print(f"用户名: {HIKVISION_CONFIG['username']}")
    print(f"通道号: {HIKVISION_CONFIG['channel']}")
    print()
    
    # 测试序列：左→右→上→下→停止
    test_sequence = [
        ("pan_left", 50),
        ("pan_right", 50),
        ("tilt_up", 50),
        ("tilt_down", 50)
    ]
    
    for action, speed in test_sequence:
        print(f"\n=== 测试 {action} ===")
        result = send_ptz_command(action, speed)
        print(f"结果: {'✅ 成功' if result['success'] else '❌ 失败'}")
        print(f"消息: {result['message']}")
        
        # 持续执行2秒
        time.sleep(2)
        
        # 发送停止命令
        print(f"发送停止命令...")
        stop_result = send_ptz_command("pan_left", 0)  # 所有速度设为0
        print(f"停止结果: {'✅ 成功' if stop_result['success'] else '❌ 失败'}")
        
        # 等待1秒
        time.sleep(1)
    
    print("\n=== 测试完成 ===")
    print("请观察摄像头是否有相应动作。")
    print("如果摄像头没有动作，请检查以下事项：")
    print("1. 摄像头IP地址是否正确")
    print("2. 用户名和密码是否正确")
    print("3. 摄像头网络连接是否正常")
    print("4. 摄像头是否支持HTTP API控制")
    print("5. 摄像头是否已启用PTZ控制")

if __name__ == "__main__":
    test_ptz_control()
