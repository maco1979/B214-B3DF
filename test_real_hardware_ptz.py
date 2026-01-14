import requests
import time

# 真实硬件PTZ测试脚本
print("=== 真实硬件PTZ测试 ===")
print("确保命令真正发送到硬件设备")
print("\n注意: 请确保已将脚本中的IP地址和端口更改为实际PTZ设备的地址")

# 配置 - 根据实际硬件修改
BASE_URL = "http://localhost:8001"

# 真实硬件PTZ配置
REAL_PTZ_CONFIG = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://192.168.1.100",  # 替换为实际PTZ设备IP
    "username": "admin",
    "password": "admin"
}

# 检查依赖
try:
    import httpx
    print("✓ httpx库已安装")
except ImportError:
    print("✗ httpx库未安装，正在安装...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx"])
    print("✓ httpx库安装成功")

# 1. 检查摄像头状态
print("\n1. 检查摄像头状态...")
camera_status = requests.get(f"{BASE_URL}/api/camera/status").json()
print(f"摄像头状态: {camera_status}")

# 2. 打开摄像头
if not camera_status.get("success") or not camera_status.get("data", {}).get("is_open"):
    print("\n2. 打开摄像头...")
    open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
    print(f"摄像头打开结果: {open_result}")

# 3. 断开现有PTZ连接
print("\n3. 断开现有PTZ连接...")
disconnect_result = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
print(f"断开PTZ连接结果: {disconnect_result}")

# 4. 连接真实PTZ硬件
print("\n4. 连接真实PTZ硬件...")
connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=REAL_PTZ_CONFIG).json()
print(f"连接PTZ硬件结果: {connect_result}")

if not connect_result.get("success"):
    print("\nPTZ连接失败，退出测试")
    exit(1)

# 5. 测试基本动作
print("\n5. 测试基本PTZ动作...")

basic_actions = [
    ("pan_left", "向左转动"),
    ("pan_right", "向右转"),
    ("tilt_up", "向上转"),
    ("tilt_down", "向下转"),
    ("stop", "停止")
]

speed = 30

try:
    for action, desc in basic_actions:
        print(f"\n执行{desc} (速度: {speed})...")
        
        # 发送PTZ命令
        ptz_request = {
            "action": action,
            "speed": speed
        }
        response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=ptz_request)
        result = response.json()
        
        print(f"结果: {result}")
        
        if result.get("success"):
            print(f"✓ {desc}命令发送成功")
        else:
            print(f"✗ {desc}命令发送失败")
            print(f"错误: {result.get('message')}")
        
        # 等待动作执行
        if action != "stop":
            time.sleep(1.0)  # 执行1秒
            # 发送停止命令
            stop_response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
            print(f"停止命令结果: {stop_response.json()}")
            time.sleep(0.5)  # 等待停止

    # 6. 检查最终状态
    print("\n6. 检查最终PTZ状态...")
    final_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    print(f"最终PTZ状态: {final_status}")
    
    if final_status.get("success"):
        position = final_status.get("data", {}).get("position", {})
        print(f"最终位置: pan={position.get('pan')}, tilt={position.get('tilt')}, zoom={position.get('zoom')}")

    print("\n=== 测试完成 ===")
    print("\n故障排查建议:")
    print("1. 检查PTZ设备电源和网线是否连接正常")
    print("2. 确认PTZ设备IP地址是否正确")
    print("3. 验证PTZ设备是否支持HTTP API")
    print("4. 检查PTZ设备的用户名和密码是否正确")
    print("5. 确认PTZ设备的HTTP API路径是否与脚本兼容")
    print("6. 查看PTZ设备的系统日志，确认是否收到命令")
    
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
