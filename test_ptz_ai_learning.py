import requests
import time
import random

# 测试配置
BASE_URL = "http://localhost:8001"

# AI学习PTZ控制测试
print("=== AI学习PTZ控制测试 ===")

# 1. 打开摄像头
print("1. 打开摄像头...")
response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0})
camera_result = response.json()
print(f"摄像头打开结果: {camera_result}")

if not camera_result.get("success"):
    print("摄像头打开失败，退出测试")
    exit(1)

# 2. 连接PTZ
print("\n2. 连接PTZ...")
ptz_config = {
    "protocol": "pelco_d",
    "connection_type": "http",
    "base_url": "http://localhost:8001",
    "username": "admin",
    "password": "admin"
}
response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
ptz_result = response.json()
print(f"PTZ连接结果: {ptz_result}")

if not ptz_result.get("success"):
    print("PTZ连接失败，退出测试")
    # 关闭摄像头
    requests.post(f"{BASE_URL}/api/camera/close")
    exit(1)

# 3. AI学习阶段：执行多样化的PTZ动作
print("\n3. AI学习阶段 - 执行多样化PTZ动作...")
print("   此阶段将执行各种PTZ动作，供AI学习云台运动规律")
print("   按Ctrl+C可停止测试")

# 定义AI学习动作集
actions = ["pan_left", "pan_right", "tilt_up", "tilt_down", "stop"]
speeds = [10, 30, 50, 70, 90]  # 不同速度

try:
    # 学习时长：10分钟
    learning_duration = 600  # 10分钟
    start_time = time.time()
    
    # 学习轮次
    learning_rounds = 0
    
    while time.time() - start_time < learning_duration:
        learning_rounds += 1
        
        # 随机选择动作和速度
        action = random.choice(actions)
        speed = random.choice(speeds)
        
        print(f"\n学习轮次 {learning_rounds}：")
        print(f"  执行动作: {action}")
        print(f"  执行速度: {speed}")
        
        # 发送PTZ命令
        ptz_request = {
            "action": action,
            "speed": speed
        }
        response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=ptz_request)
        result = response.json()
        
        if result.get("success"):
            print(f"  ✓ 动作执行成功")
            # 打印当前状态
            status_response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
            status = status_response.json()
            if status.get("success"):
                position = status.get("data", {}).get("position", {})
                print(f"  当前位置: pan={position.get('pan'):.1f}, tilt={position.get('tilt'):.1f}, zoom={position.get('zoom'):.1f}")
        else:
            print(f"  ✗ 动作执行失败: {result.get('message')}")
        
        # 随机等待时间（0.5-2秒）
        wait_time = random.uniform(0.5, 2.0)
        time.sleep(wait_time)
        
except KeyboardInterrupt:
    print("\n\n=== AI学习测试已停止 ===")
except Exception as e:
    print(f"\n\n=== 测试发生错误: {e} ===")
finally:
    # 4. 停止所有动作
    print("\n4. 停止所有动作...")
    requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
    
    # 5. 断开PTZ连接
    print("5. 断开PTZ连接...")
    requests.post(f"{BASE_URL}/api/camera/ptz/disconnect")
    
    # 6. 关闭摄像头
    print("6. 关闭摄像头...")
    requests.post(f"{BASE_URL}/api/camera/close")
    
    print("\n=== 测试完成 ===")
    print(f"AI学习轮次: {learning_rounds}")
    print("\nAI学习建议:")
    print("1. 观察PTZ动作与状态变化的对应关系")
    print("2. 学习不同速度下的运动特性")
    print("3. 理解动作间的过渡关系")
    print("4. 建立位置预测模型")
    print("5. 学习最佳控制策略")
