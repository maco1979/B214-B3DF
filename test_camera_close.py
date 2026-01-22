import requests
import time

# 后端API地址
BASE_URL = "http://localhost:8001/api"

print("=== 摄像头关闭功能测试 ===")

# 1. 先获取摄像头状态
print("\n1. 获取当前摄像头状态...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
    status_result = status_response.json()
    print(f"   当前状态: {status_result}")
except Exception as e:
    print(f"   获取状态失败: {e}")

# 2. 尝试关闭摄像头
print("\n2. 尝试关闭摄像头...")
try:
    close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
    close_result = close_response.json()
    print(f"   关闭结果: {close_result}")
except Exception as e:
    print(f"   关闭失败: {e}")

# 3. 再次检查状态
print("\n3. 再次检查摄像头状态...")
try:
    status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
    status_result = status_response.json()
    print(f"   关闭后状态: {status_result}")
    is_open = status_result['data']['is_open']
    if not is_open:
        print("   ✅ 摄像头已成功关闭")
    else:
        print("   ❌ 摄像头仍然打开")
        
        # 4. 尝试强制关闭（打开后立即关闭）
        print("\n4. 尝试强制关闭...")
        try:
            open_response = requests.post(f"{BASE_URL}/camera/open", json={"camera_index": 999}, timeout=5)
            print(f"   重新打开: {open_response.json()}")
            
            # 立即关闭
            time.sleep(0.5)
            close_response = requests.post(f"{BASE_URL}/camera/close", timeout=5)
            close_result = close_response.json()
            print(f"   强制关闭: {close_result}")
            
            # 再次检查
            time.sleep(0.5)
            status_response = requests.get(f"{BASE_URL}/camera/status", timeout=5)
            status_result = status_response.json()
            print(f"   强制关闭后状态: {status_result}")
            if not status_result['data']['is_open']:
                print("   ✅ 强制关闭成功")
            else:
                print("   ❌ 强制关闭失败")
                print("   可能有其他客户端正在重新打开摄像头")
        except Exception as e:
            print(f"   强制关闭失败: {e}")
except Exception as e:
    print(f"   检查状态失败: {e}")

# 5. 检查WebSocket连接
print("\n5. 检查WebSocket连接...")
print("   如果有WebSocket客户端连接到 /api/camera/ws/frame，")
print("   摄像头可能会保持打开状态。请检查前端代码是否有自动重新连接逻辑。")

print("\n=== 测试完成 ===")
