import requests
import time

# 后端API地址
BASE_URL = "http://localhost:8001/api"

# 测试步骤：
# 1. 打开摄像头
# 2. 启动视觉跟踪
# 3. 获取跟踪状态
# 4. 停止跟踪

print("=== 摄像头跟踪功能测试 ===")

# 1. 打开摄像头
print("\n1. 打开摄像头...")
camera_open_url = f"{BASE_URL}/camera/open"
response = requests.post(camera_open_url, json={"camera_index": 999})  # 使用模拟摄像头
if response.status_code == 200:
    result = response.json()
    print(f"   状态: {'成功' if result['success'] else '失败'}")
    print(f"   消息: {result['message']}")
    if result['success']:
        print(f"   摄像头索引: {result['data']['camera_index']}")
else:
    print(f"   错误: {response.status_code} - {response.text}")

# 2. 启动视觉跟踪
print("\n2. 启动视觉跟踪...")
tracking_start_url = f"{BASE_URL}/camera/tracking/start"
response = requests.post(tracking_start_url, json={"tracker_type": "MIL"})
if response.status_code == 200:
    result = response.json()
    print(f"   状态: {'成功' if result['success'] else '失败'}")
    print(f"   消息: {result['message']}")
    if result['success']:
        print(f"   跟踪器类型: {result['data']['tracker_type']}")
        print(f"   初始边界框: {result['data']['initial_bbox']}")
else:
    print(f"   错误: {response.status_code} - {response.text}")

# 3. 获取跟踪状态
print("\n3. 获取跟踪状态...")
tracking_status_url = f"{BASE_URL}/camera/tracking/status"
for i in range(3):
    response = requests.get(tracking_status_url)
    if response.status_code == 200:
        result = response.json()
        print(f"   状态: {'成功' if result['success'] else '失败'}")
        print(f"   消息: {result['message']}")
        if result['success']:
            print(f"   跟踪启用: {result['data']['tracking_enabled']}")
            print(f"   跟踪器类型: {result['data']['tracker_type']}")
            print(f"   跟踪对象: {result['data']['tracked_object']}")
            print(f"   跟踪结果数: {result['data']['tracking_results_count']}")
    else:
        print(f"   错误: {response.status_code} - {response.text}")
    time.sleep(1)

# 4. 停止跟踪
print("\n4. 停止跟踪...")
tracking_stop_url = f"{BASE_URL}/camera/tracking/stop"
response = requests.post(tracking_stop_url)
if response.status_code == 200:
    result = response.json()
    print(f"   状态: {'成功' if result['success'] else '失败'}")
    print(f"   消息: {result['message']}")
else:
    print(f"   错误: {response.status_code} - {response.text}")

# 5. 关闭摄像头
print("\n5. 关闭摄像头...")
camera_close_url = f"{BASE_URL}/camera/close"
response = requests.post(camera_close_url)
if response.status_code == 200:
    result = response.json()
    print(f"   状态: {'成功' if result['success'] else '失败'}")
    print(f"   消息: {result['message']}")
else:
    print(f"   错误: {response.status_code} - {response.text}")

print("\n=== 测试完成 ===")
