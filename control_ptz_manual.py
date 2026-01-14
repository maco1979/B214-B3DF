import requests
import time

# PTZ手动控制脚本
print("=== PTZ云台手动控制 ===")
print("使用方向键控制云台运动，按q退出")
print("\n控制说明:")
print("  ← → : 水平转动")
print("  ↑ ↓ : 垂直转动")
print("  s   : 停止所有动作")
print("  q   : 退出程序")
print("\n速度控制:")
print("  1-9 : 设置速度级别 (1=最慢, 9=最快)")

# 配置
BASE_URL = "http://localhost:8001"
camera_index = 0

# 速度映射
SPEED_MAP = {
    '1': 10,
    '2': 20,
    '3': 30,
    '4': 40,
    '5': 50,
    '6': 60,
    '7': 70,
    '8': 80,
    '9': 90
}

current_speed = 50  # 默认速度

# 检查摄像头状态
def check_camera_status():
    response = requests.get(f"{BASE_URL}/api/camera/status")
    return response.json()

# 打开摄像头
def open_camera():
    response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": camera_index})
    return response.json()

# 检查PTZ状态
def check_ptz_status():
    response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
    return response.json()

# 连接PTZ
def connect_ptz():
    ptz_config = {
        "protocol": "http",
        "connection_type": "http",
        "base_url": "http://localhost:8001",
        "username": "admin",
        "password": "admin"
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
    return response.json()

# 发送PTZ命令
def send_ptz_command(action):
    request_data = {
        "action": action,
        "speed": current_speed
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=request_data)
    return response.json()

# 主程序
try:
    # 1. 检查并打开摄像头
    print("\n1. 检查摄像头状态...")
    camera_status = check_camera_status()
    print(f"摄像头状态: {camera_status}")
    
    if not camera_status.get("success") or not camera_status.get("data", {}).get("is_open"):
        print("正在打开摄像头...")
        open_result = open_camera()
        print(f"打开摄像头结果: {open_result}")
    
    # 2. 检查并连接PTZ
    print("\n2. 检查PTZ状态...")
    ptz_status = check_ptz_status()
    print(f"PTZ状态: {ptz_status}")
    
    if not ptz_status.get("success") or not ptz_status.get("data", {}).get("connected"):
        print("正在连接PTZ...")
        connect_result = connect_ptz()
        print(f"连接PTZ结果: {connect_result}")
    
    # 3. 开始手动控制
    print("\n3. 开始PTZ手动控制")
    print("按方向键控制，按q退出")
    print(f"当前速度: {current_speed}")
    
    while True:
        # 读取键盘输入
        key = input("\n输入命令: ")
        
        if key == 'q':
            print("退出控制程序")
            break
        
        elif key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            current_speed = SPEED_MAP[key]
            print(f"速度已设置为: {current_speed}")
        
        elif key == 's':
            print("发送停止命令...")
            result = send_ptz_command("stop")
            print(f"结果: {result}")
        
        elif key == '\x1b[A':  # 上箭头
            print("发送向上转动命令...")
            result = send_ptz_command("tilt_up")
            print(f"结果: {result}")
        
        elif key == '\x1b[B':  # 下箭头
            print("发送向下转动命令...")
            result = send_ptz_command("tilt_down")
            print(f"结果: {result}")
        
        elif key == '\x1b[C':  # 右箭头
            print("发送向右转动命令...")
            result = send_ptz_command("pan_right")
            print(f"结果: {result}")
        
        elif key == '\x1b[D':  # 左箭头
            print("发送向左转动命令...")
            result = send_ptz_command("pan_left")
            print(f"结果: {result}")
        
        else:
            print("未知命令，请重新输入")
            print("可用命令: 方向键(←→↑↓), s(停止), 1-9(速度), q(退出)")
            
except KeyboardInterrupt:
    print("\n\n程序已停止")
except Exception as e:
    print(f"\n\n发生错误: {e}")
finally:
    # 停止所有动作
    print("\n发送最终停止命令...")
    try:
        send_ptz_command("stop")
    except:
        pass
    print("PTZ控制程序已结束")
