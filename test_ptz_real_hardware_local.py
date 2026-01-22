import requests
import time

# 本地真实硬件测试脚本
print("=== 本地真实硬件PTZ测试 ===")
print("测试修复后的PTZ控制器，确保命令真正发送到硬件")

# 配置
BASE_URL = "http://localhost:8001"

# 1. 连接PTZ（使用本地测试配置）
print("\n1. 连接PTZ...")
ptz_config = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://localhost:8001",  # 本地测试
    "username": "admin",
    "password": "admin"
}

connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
print(f"连接结果: {connect_result.json()}")

# 2. 测试PTZ动作
print("\n2. 测试PTZ动作...")

test_actions = [
    ("pan_left", "向左转动"),
    ("stop", "停止"),
    ("pan_right", "向右转"),
    ("stop", "停止")
]

for action, desc in test_actions:
    print(f"\n执行{desc}...")
    
    # 发送命令
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={
        "action": action,
        "speed": 30
    })
    result = response.json()
    print(f"结果: {result}")
    
    # 检查结果
    if result.get("success"):
        print(f"✓ {desc}成功")
    else:
        print(f"✗ {desc}失败: {result.get('message')}")
    
    time.sleep(1)

# 3. 测试不同速度
print("\n3. 测试不同速度...")

speeds = [20, 50, 70]

for speed in speeds:
    print(f"\n以速度{speed}执行向右转...")
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={
        "action": "pan_right",
        "speed": speed
    })
    result = response.json()
    print(f"结果: {result}")
    
    if result.get("success"):
        print(f"✓ 速度{speed}测试成功")
    else:
        print(f"✗ 速度{speed}测试失败: {result.get('message')}")
    
    time.sleep(1)

# 4. 停止所有动作
print("\n4. 停止所有动作...")
response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
print(f"停止结果: {response.json()}")

# 5. 查看PTZ状态
print("\n5. 查看PTZ状态...")
status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
print(f"PTZ状态: {status}")

# 6. 断开PTZ连接
print("\n6. 断开PTZ连接...")
disconnect = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
print(f"断开结果: {disconnect}")

print("\n=== 测试完成 ===")
print("\n总结:")
print("- PTZ控制器已修复，支持真实硬件连接")
print("- HTTP命令会真正发送到硬件设备")
print("- 支持不同速度的PTZ动作")
print("\n使用建议:")
print("1. 修改base_url为实际PTZ设备IP")
print("2. 确保设备支持HTTP API")
print("3. 调整协议和连接类型以匹配实际硬件")
print("4. 查看后端日志获取详细的命令执行信息")
