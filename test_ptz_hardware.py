import requests
import time

# 测试配置
BASE_URL = "http://localhost:8001"

# 测试真实硬件PTZ动作
print("=== 真实硬件PTZ动作测试 ===")

# 1. 打开摄像头
print("1. 打开摄像头...")
response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0})
print(f"摄像头打开结果: {response.json()}")

# 2. 连接真实PTZ硬件
# 根据实际硬件情况修改以下配置
print("\n2. 连接真实PTZ硬件...")

# 配置选项1：网络TCP PTZ（真实网络PTZ设备）
ptz_config = {
    "protocol": "pelco_d",
    "connection_type": "http",
    "base_url": "http://localhost:8001",  # 使用本地服务器处理PTZ命令
    "username": "admin",
    "password": "admin"
}

# 配置选项2：串口PTZ（直接连接电脑串口）
# ptz_config = {
#     "protocol": "pelco_d",
#     "connection_type": "serial",
#     "port": "COM3",  # 替换为实际串口号
#     "baudrate": 9600,
#     "address": 1
# }

# 配置选项3：网络转串口PTZ
# ptz_config = {
#     "protocol": "pelco_d",
#     "connection_type": "network",
#     "host": "192.168.1.100",  # 替换为网络转串口设备IP
#     "port": 5000,  # 替换为实际端口
#     "address": 1
# }

response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
print(f"PTZ连接结果: {response.json()}")

# 3. 测试PTZ动作
print("\n3. 测试真实PTZ动作...")

# 定义测试动作
actions = [
    ("pan_left", "向左转"),
    ("stop", "停止"),
    ("pan_right", "向右转"),
    ("stop", "停止"),
    ("tilt_up", "向上转"),
    ("stop", "停止"),
    ("tilt_down", "向下转"),
    ("stop", "停止")
]

for action, desc in actions:
    print(f"  执行{desc}...")
    ptz_request = {
        "action": action,
        "speed": 30
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=ptz_request)
    result = response.json()
    print(f"  结果: {result}")
    
    # 检查是否成功执行
    if result.get("success"):
        print(f"  ✓ {desc}命令已发送到硬件")
    else:
        print(f"  ✗ {desc}命令执行失败")
    
    time.sleep(0.5)  # 等待0.5秒

# 4. 检查PTZ状态
print("\n4. 检查PTZ状态...")
response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
status = response.json()
print(f"PTZ状态: {status}")

if status.get("success"):
    data = status.get("data", {})
    if data.get("connected"):
        print(f"  ✓ PTZ已连接")
        print(f"  ✓ 协议: {data.get('protocol')}")
        print(f"  ✓ 连接类型: {data.get('connection_type')}")
        print(f"  ✓ 当前位置: pan={data.get('position', {}).get('pan')}, tilt={data.get('position', {}).get('tilt')}, zoom={data.get('position', {}).get('zoom')}")
    else:
        print(f"  ✗ PTZ未连接")

# 5. 断开PTZ连接
print("\n5. 断开PTZ连接...")
response = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect")
print(f"PTZ断开结果: {response.json()}")

# 6. 关闭摄像头
print("\n6. 关闭摄像头...")
response = requests.post(f"{BASE_URL}/api/camera/close")
print(f"摄像头关闭结果: {response.json()}")

print("\n=== 测试完成 ===")
print("\n故障排查建议:")
print("1. 检查PTZ设备是否正确连接电源和信号线")
print("2. 确认设备IP/串口/波特率等参数是否正确")
print("3. 验证设备是否支持所选协议（Pelco-D/Pelco-P等）")
print("4. 检查设备地址是否设置为1")
print("5. 尝试使用设备自带的控制软件测试")
