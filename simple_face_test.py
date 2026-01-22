import requests
import time

# 后端API地址
BASE_URL = "http://localhost:8001/api"

print("=== 简单人脸检测测试 ===")

# 1. 先检查摄像头状态并关闭
print("\n1. 检查并关闭摄像头...")
try:
    close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
    print(f"   关闭结果: {close_response.json()}")
except Exception as e:
    print(f"   关闭摄像头失败: {e}")
    print("   继续测试...")

# 2. 打开摄像头
print("\n2. 打开摄像头...")
try:
    open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=5)
    open_result = open_response.json()
    print(f"   打开结果: {open_result}")
    if not open_result['success']:
        print("   摄像头打开失败，退出测试")
        exit(1)
except Exception as e:
    print(f"   打开摄像头失败: {e}")
    exit(1)

# 3. 启动人脸识别
print("\n3. 启动人脸识别...")
try:
    start_response = requests.post(f"{BASE_URL}/camera/recognition/start", json={"model_type": "haar"}, timeout=5)
    start_result = start_response.json()
    print(f"   启动结果: {start_result}")
    if not start_result['success']:
        print("   人脸识别启动失败，退出测试")
        # 关闭摄像头
        requests.post(f"{BASE_URL}/camera/close", timeout=5)
        exit(1)
except Exception as e:
    print(f"   启动人脸识别失败: {e}")
    # 关闭摄像头
    requests.post(f"{BASE_URL}/camera/close", timeout=5)
    exit(1)

# 4. 等待并检查识别结果
print("\n4. 等待5秒后检查识别结果...")
time.sleep(5)

try:
    status_response = requests.get(f"{BASE_URL}/camera/recognition/status", timeout=5)
    status_result = status_response.json()
    print(f"   识别状态: {status_result}")
    
    # 检查模拟帧内容
    print("\n5. 检查模拟帧内容...")
    frame_response = requests.get(f"{BASE_URL}/camera/frame", timeout=5)
    frame_result = frame_response.json()
    if frame_result['success']:
        print("   成功获取帧数据")
        print(f"   帧数据大小: {len(frame_result['data']['frame_base64'])} 字节")
    else:
        print(f"   获取帧失败: {frame_result['message']}")
        
except Exception as e:
    print(f"   获取状态或帧失败: {e}")

# 6. 停止识别并关闭摄像头
print("\n6. 清理资源...")
try:
    requests.post(f"{BASE_URL}/camera/recognition/stop", timeout=5)
    requests.post(f"{BASE_URL}/camera/close", timeout=5)
    print("   资源清理完成")
except Exception as e:
    print(f"   资源清理失败: {e}")

print("\n=== 测试完成 ===")

print("\n=== 人脸检测结果分析 ===")
print("1. 模拟摄像头生成的帧是简单图形，包含文字和几何图形，没有真实人脸")
print("2. Haar级联人脸检测器只能检测真实人脸特征，所以检测结果为0是正常的")
print("3. 当使用真实摄像头且画面中有人脸时，应该能检测到人脸")
print("4. 你可以尝试以下操作：")
print("   - 使用真实摄像头（camera_index=0）")
print("   - 确保画面中有清晰可见的人脸")
print("   - 调整摄像头角度和光线条件")
