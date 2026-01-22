import requests
import json

# 测试PTZ云台控制API
BASE_URL = "http://localhost:8005"

def test_ptz_status():
    """测试获取PTZ状态"""
    try:
        response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
        print("PTZ状态:", response.json())
        return response.json()
    except Exception as e:
        print(f"获取PTZ状态失败: {e}")
        return None

def test_ptz_connect():
    """测试连接PTZ云台"""
    try:
        payload = {
            "protocol": "pelco_d",
            "connection_type": "serial",
            "port": "COM1",  # Windows上典型的串口
            "baudrate": 9600,
            "address": 1
        }
        response = requests.post(
            f"{BASE_URL}/api/camera/ptz/connect", 
            json=payload
        )
        print("PTZ连接结果:", response.json())
        return response.json()
    except Exception as e:
        print(f"PTZ连接失败: {e}")
        return None

def test_ptz_action(action, speed=50):
    """测试PTZ动作"""
    try:
        payload = {
            "action": action,
            "speed": speed
        }
        response = requests.post(
            f"{BASE_URL}/api/camera/ptz/action", 
            json=payload
        )
        print(f"PTZ {action} 动作结果:", response.json())
        return response.json()
    except Exception as e:
        print(f"PTZ {action} 动作失败: {e}")
        return None

def test_ptz_move(pan, tilt, zoom=None, speed=50):
    """测试PTZ移动到指定位置"""
    try:
        payload = {
            "pan": pan,
            "tilt": tilt,
            "speed": speed
        }
        if zoom is not None:
            payload["zoom"] = zoom
            
        response = requests.post(
            f"{BASE_URL}/api/camera/ptz/move", 
            json=payload
        )
        print(f"PTZ移动到({pan}, {tilt})结果:", response.json())
        return response.json()
    except Exception as e:
        print(f"PTZ移动失败: {e}")
        return None

if __name__ == "__main__":
    print("开始测试PTZ云台控制...")
    
    # 1. 检查当前状态
    print("\n1. 检查PTZ云台状态:")
    status = test_ptz_status()
    
    # 2. 尝试连接云台
    print("\n2. 尝试连接PTZ云台:")
    connect_result = test_ptz_connect()
    
    # 3. 如果连接成功，测试一些动作
    if connect_result and connect_result.get("success"):
        print("\n3. 测试PTZ动作:")
        
        # 测试向右转
        test_ptz_action("pan_right", 30)
        
        # 测试向上转
        test_ptz_action("tilt_up", 30)
        
        # 测试停止
        test_ptz_action("stop")
        
        # 测试移动到指定位置
        print("\n4. 测试移动到指定位置:")
        test_ptz_move(pan=45.0, tilt=30.0, zoom=2.0)
    
    print("\nPTZ云台控制测试完成")