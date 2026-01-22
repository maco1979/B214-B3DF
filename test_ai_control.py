import requests
import json

# 测试AI控制API
BASE_URL = "http://localhost:8001"

print("===== 测试AI控制API =====")

# 测试1: 获取设备列表
print("\n1. 测试获取设备列表...")
response = requests.get(f"{BASE_URL}/ai-control/devices")
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# 测试2: 测试AI主控开关
print("\n2. 测试AI主控开关...")
response = requests.post(f"{BASE_URL}/ai-control/master-control", json={"activate": True})
print(f"激活主控状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# 测试3: 获取AI主控状态
print("\n3. 测试获取AI主控状态...")
response = requests.get(f"{BASE_URL}/ai-control/master-control/status")
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# 测试4: 控制设备
print("\n4. 测试控制设备...")
response = requests.post(f"{BASE_URL}/ai-control/device/1", json={"action": "get_status"})
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# 测试5: 获取设备状态
print("\n5. 测试获取设备状态...")
response = requests.get(f"{BASE_URL}/ai-control/device/1/status")
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

# 测试6: 切换设备连接状态
print("\n6. 测试切换设备连接状态...")
response = requests.post(f"{BASE_URL}/ai-control/device/4/connection", json={"connect": True})
print(f"状态码: {response.status_code}")
print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

print("\n===== AI控制API测试完成 =====")
