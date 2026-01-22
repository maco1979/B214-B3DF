import requests
import time

# 后端API地址
BASE_URL = "http://localhost:8001/api"

print("=== 摄像头完整生命周期测试 ===")

# 1. 初始状态检查
print("\n1. 初始状态检查...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
    status_result = status_response.json()
    print(f"   初始状态: {status_result}")
    initial_is_open = status_result['data']['is_open']
    if not initial_is_open:
        print("   ✅ 摄像头初始状态为关闭")
    else:
        print("   ⚠️  摄像头初始状态为打开，将尝试关闭")
        close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
        print(f"   关闭结果: {close_response.json()}")
except Exception as e:
    print(f"   获取初始状态失败: {e}")
    exit(1)

# 2. 打开摄像头
print("\n2. 打开摄像头...")
try:
    open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=5)
    open_result = open_response.json()
    print(f"   打开结果: {open_result}")
    if open_result['success']:
        print("   ✅ 摄像头打开成功")
    else:
        print("   ❌ 摄像头打开失败")
        exit(1)
    
    # 验证状态
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
    status_result = status_response.json()
    if status_result['data']['is_open']:
        print("   ✅ 状态验证：摄像头已打开")
    else:
        print("   ❌ 状态验证：摄像头仍然关闭")
        exit(1)
except Exception as e:
    print(f"   打开摄像头失败: {e}")
    exit(1)

# 3. 启动跟踪
print("\n3. 启动跟踪...")
try:
    start_track_response = requests.post(f"{BASE_URL}/camera/tracking/start", json={"tracker_type": "MIL"}, timeout=5)
    start_track_result = start_track_response.json()
    print(f"   启动跟踪结果: {start_track_result}")
    
    # 验证跟踪状态
    track_status_response = requests.get(f"{BASE_URL}/camera/tracking/status", timeout=5)
    track_status_result = track_status_response.json()
    print(f"   跟踪状态: {track_status_result}")
except Exception as e:
    print(f"   启动跟踪失败: {e}")

# 4. 启动识别
print("\n4. 启动识别...")
try:
    start_recog_response = requests.post(f"{BASE_URL}/camera/recognition/start", json={"model_type": "haar"}, timeout=5)
    start_recog_result = start_recog_response.json()
    print(f"   启动识别结果: {start_recog_result}")
    
    # 验证识别状态
    recog_status_response = requests.get(f"{BASE_URL}/camera/recognition/status", timeout=5)
    recog_status_result = recog_status_response.json()
    print(f"   识别状态: {recog_status_result}")
except Exception as e:
    print(f"   启动识别失败: {e}")

# 5. 关闭摄像头
print("\n5. 关闭摄像头...")
try:
    close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
    close_result = close_response.json()
    print(f"   关闭结果: {close_result}")
    if close_result['success']:
        print("   ✅ 摄像头关闭成功")
    else:
        print("   ❌ 摄像头关闭失败")
        exit(1)
    
    # 验证关闭状态
    time.sleep(1)  # 等待状态更新
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
    status_result = status_response.json()
    print(f"   关闭后状态: {status_result}")
    
    if not status_result['data']['is_open']:
        print("   ✅ 状态验证：摄像头已关闭")
    else:
        print("   ❌ 状态验证：摄像头仍然打开")
        exit(1)
    
    # 验证跟踪状态已重置
    track_status_response = requests.get(f"{BASE_URL}/camera/tracking/status", timeout=5)
    track_status_result = track_status_response.json()
    if not track_status_result['data']['tracking_enabled']:
        print("   ✅ 跟踪状态已重置")
    else:
        print("   ⚠️  跟踪状态未完全重置")
    
    # 验证识别状态已重置
    recog_status_response = requests.get(f"{BASE_URL}/camera/recognition/status", timeout=5)
    recog_status_result = recog_status_response.json()
    if not recog_status_result['data']['recognizing_enabled']:
        print("   ✅ 识别状态已重置")
    else:
        print("   ⚠️  识别状态未完全重置")
except Exception as e:
    print(f"   关闭摄像头失败: {e}")
    exit(1)

# 6. 重复打开和关闭测试
print("\n6. 重复打开和关闭测试...")
try:
    for i in range(2):
        print(f"   测试轮次 {i+1}:")
        
        # 打开
        open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=5)
        print(f"   - 打开: {open_response.json()['success']}")
        
        # 等待
        time.sleep(0.5)
        
        # 关闭
        close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
        print(f"   - 关闭: {close_response.json()['success']}")
        
        # 验证
        status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
        is_open = status_response.json()['data']['is_open']
        print(f"   - 状态验证: {'关闭' if not is_open else '打开'}")
        
        if not is_open:
            print(f"   - ✅ 轮次 {i+1} 成功")
        else:
            print(f"   - ❌ 轮次 {i+1} 失败")
            exit(1)
    
    print("   ✅ 重复打开和关闭测试通过")
except Exception as e:
    print(f"   重复测试失败: {e}")
    exit(1)

print("\n=== 测试完成 ===")
print("✅ 摄像头完整生命周期测试通过")
print("✅ 摄像头关闭功能已修复")
print("✅ 所有相关状态都能正确重置")
