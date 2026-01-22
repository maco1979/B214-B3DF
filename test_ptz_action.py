import requests
import time

# 测试配置
BASE_URL = "http://localhost:8001"

# 测试步骤
print("=== 测试PTZ动作执行 ===")

# 1. 打开摄像头
print("1. 打开摄像头...")
response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0})
print(f"摄像头打开结果: {response.json()}")

# 2. 连接PTZ
print("\n2. 连接PTZ...")
ptz_config = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://localhost:8001",
    "username": "admin",
    "password": "admin"
}
response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
print(f"PTZ连接结果: {response.json()}")

# 3. 测试PTZ动作
print("\n3. 测试PTZ动作...")

# 定义测试动作
actions = [
    ("pan_left", "向左转"),
    ("pan_right", "向右转"),
    ("tilt_up", "向上转"),
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
    print(f"  结果: {response.json()}")
    time.sleep(1)  # 等待1秒

# 4. 测试不同速度
print("\n4. 测试不同速度...")
speeds = [10, 50, 90]
for speed in speeds:
    print(f"  执行向右转动，速度{speed}...")
    ptz_request = {
        "action": "pan_right",
        "speed": speed
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=ptz_request)
    print(f"  结果: {response.json()}")
    time.sleep(0.5)  # 等待0.5秒

# 停止
requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})

# 5. 检查PTZ状态
print("\n5. 检查PTZ状态...")
response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
print(f"PTZ状态: {response.json()}")

# 6. 断开PTZ连接
print("\n6. 断开PTZ连接...")
response = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect")
print(f"PTZ断开结果: {response.json()}")

# 7. 关闭摄像头
print("\n7. 关闭摄像头...")
response = requests.post(f"{BASE_URL}/api/camera/close")
print(f"摄像头关闭结果: {response.json()}")

print("\n=== 测试完成 ===")
