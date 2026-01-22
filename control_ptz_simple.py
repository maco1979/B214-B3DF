import requests
import time

# 简单PTZ控制脚本
print("=== 简单PTZ云台控制 ===")
print("使用字母命令控制云台运动")
print("\n控制命令:")
print("  a : 向左转")
print("  d : 向右转")
print("  w : 向上转")
print("  s : 向下转")
print("  x : 停止所有动作")
print("  q : 退出程序")
print("  1-9 : 设置速度 (1=最慢, 9=最快)")

# 配置
BASE_URL = "http://localhost:8001"

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

current_speed = 50

# 发送PTZ命令
def send_ptz(action):
    request_data = {
        "action": action,
        "speed": current_speed
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=request_data)
    return response.json()

# 主程序
print("\n=== 开始控制 ===")
print(f"当前速度: {current_speed}")

# 停止所有动作
send_ptz("stop")

while True:
    cmd = input("\n请输入命令: ").lower()
    
    if cmd == 'q':
        print("退出控制")
        send_ptz("stop")
        break
    
    elif cmd == 'x':
        print("停止所有动作")
        result = send_ptz("stop")
        print(f"结果: {result}")
    
    elif cmd == 'a':
        print("向左转动")
        result = send_ptz("pan_left")
        print(f"结果: {result}")
    
    elif cmd == 'd':
        print("向右转动")
        result = send_ptz("pan_right")
        print(f"结果: {result}")
    
    elif cmd == 'w':
        print("向上转动")
        result = send_ptz("tilt_up")
        print(f"结果: {result}")
    
    elif cmd == 's':
        print("向下转动")
        result = send_ptz("tilt_down")
        print(f"结果: {result}")
    
    elif cmd in SPEED_MAP:
        current_speed = SPEED_MAP[cmd]
        print(f"速度已设置为: {current_speed}")
    
    else:
        print("无效命令，请重新输入")
        print("可用命令: a(左), d(右), w(上), s(下), x(停止), q(退出), 1-9(速度)")

print("\nPTZ控制结束")
