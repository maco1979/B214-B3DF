import requests
import time
import random

# 测试配置
BASE_URL = "http://localhost:8001"

# 真实PTZ硬件测试脚本
print("=== 真实PTZ硬件测试脚本 ===")
print("   专为真实PTZ设备设计，确保协议和连接类型兼容")
print("   按Ctrl+C可停止测试")

# 1. 检查摄像头状态
print("\n1. 检查摄像头状态...")
response = requests.get(f"{BASE_URL}/api/camera/status")
camera_status = response.json()
print(f"当前摄像头状态: {camera_status}")

# 如果摄像头未打开，打开摄像头
if not camera_status.get("success") or not camera_status.get("data", {}).get("is_open"):
    print("\n2. 打开摄像头...")
    response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0})
    camera_result = response.json()
    print(f"摄像头打开结果: {camera_result}")
    
    if not camera_result.get("success"):
        print("摄像头打开失败，退出测试")
        exit(1)

# 3. 断开当前PTZ连接（如果有）
print("\n3. 清理现有PTZ连接...")
response = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect")
disconnect_result = response.json()
print(f"断开PTZ连接结果: {disconnect_result}")

# 4. 连接真实PTZ硬件（使用正确的协议和连接类型组合）
print("\n4. 连接真实PTZ硬件...")

# 关键配置：确保协议和连接类型兼容
# 对于HTTP API设备，使用HTTP_API协议
ptz_config = {
    "protocol": "http",  # 使用HTTP_API协议，避免bytes命令
    "connection_type": "http",
    "base_url": "http://localhost:8001",
    "username": "admin",
    "password": "admin"
}

# 对于串口设备，使用pelco_d协议和serial连接类型
# ptz_config = {
#     "protocol": "pelco_d",
#     "connection_type": "serial",
#     "port": "COM3",
#     "baudrate": 9600,
#     "address": 1
# }

# 对于网络IP设备，使用pelco_d协议和network连接类型
# ptz_config = {
#     "protocol": "pelco_d",
#     "connection_type": "network",
#     "host": "192.168.1.100",
#     "port": 5000,
#     "address": 1
# }

response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
ptz_result = response.json()
print(f"PTZ连接结果: {ptz_result}")

if not ptz_result.get("success"):
    print("PTZ连接失败，退出测试")
    # 关闭摄像头
    requests.post(f"{BASE_URL}/api/camera/close")
    exit(1)

# 5. 测试PTZ动作
print("\n5. 测试PTZ动作...")
print("   执行真实PTZ动作，观察设备是否响应")

# 定义基本动作
basic_actions = ["pan_left", "pan_right", "tilt_up", "tilt_down", "stop"]

# 先执行停止命令，确保设备处于初始状态
print("\n   执行初始停止命令...")
response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
print(f"   停止命令结果: {response.json()}")
time.sleep(1)

# 执行基本动作测试
print("\n   执行基本动作测试...")
for action in basic_actions:
    print(f"   执行{action}...")
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": action, "speed": 30})
    result = response.json()
    print(f"   结果: {result}")
    
    # 执行动作后等待
    if action != "stop":
        time.sleep(1)  # 执行1秒
        # 停止动作
        print(f"   执行{action}后停止...")
        response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
        time.sleep(0.5)  # 等待0.5秒

# 6. 测试不同速度
print("\n6. 测试不同速度...")
speeds = [20, 50, 80]
for speed in speeds:
    print(f"   执行pan_right，速度{speed}...")
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "pan_right", "speed": speed})
    result = response.json()
    print(f"   结果: {result}")
    time.sleep(1)  # 执行1秒
    # 停止
    requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
    time.sleep(0.5)

# 7. 长时间学习模式
print("\n7. 启动长时间学习模式...")
print("   持续执行PTZ动作，供AI学习云台运动")
print("   按Ctrl+C可停止")

learning_rounds = 0
try:
    start_time = time.time()
    
    while True:
        learning_rounds += 1
        
        # 随机选择动作和速度
        action = random.choice(basic_actions)
        speed = random.randint(20, 80)
        
        print(f"\n学习轮次 {learning_rounds}：")
        print(f"  执行动作: {action}")
        print(f"  执行速度: {speed}")
        
        response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": action, "speed": speed})
        result = response.json()
        
        if result.get("success"):
            print(f"  ✓ 动作执行成功")
        else:
            print(f"  ✗ 动作执行失败: {result.get('message')}")
        
        # 随机等待时间
        wait_time = random.uniform(0.5, 2.0)
        time.sleep(wait_time)
        
except KeyboardInterrupt:
    print("\n\n=== 测试已停止 ===")
except Exception as e:
    print(f"\n\n=== 测试发生错误: {e} ===")
finally:
    # 停止所有动作
    print("\n1. 停止所有动作...")
    requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
    
    # 保持摄像头和PTZ连接打开
    print("\n2. 保持摄像头和PTZ连接打开")
    print(f"   学习总轮次: {learning_rounds}")
    print(f"   学习时长: {time.time() - start_time:.1f}秒")
    
    print("\n=== 测试完成 ===")
    print("\n真实PTZ设备使用建议:")
    print("1. 确保PTZ设备电源和信号线连接正常")
    print("2. 验证设备支持的协议类型（Pelco-D/Pelco-P/HTTP等）")
    print("3. 使用正确的连接类型（serial/network/http）")
    print("4. 检查设备地址是否设置为1")
    print("5. 尝试使用设备自带的控制软件测试基本功能")
    print("6. 如果使用网络设备，确保IP地址和端口正确")
