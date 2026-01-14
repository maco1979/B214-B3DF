import requests
import time
import random

# 测试配置
BASE_URL = "http://localhost:8001"

# 长时间AI学习PTZ控制测试
print("=== 长时间AI学习PTZ控制测试 ===")
print("   真实摄像头打开，持续学习云台运动")
print("   不关闭摄像头和云台连接")
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

# 3. 检查PTZ状态
print("\n3. 检查PTZ状态...")
response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
ptz_status = response.json()
print(f"当前PTZ状态: {ptz_status}")

# 如果PTZ未连接，连接PTZ
if not ptz_status.get("success") or not ptz_status.get("data", {}).get("connected"):
    print("\n4. 连接PTZ...")
    ptz_config = {
        "protocol": "http",
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
        exit(1)

# 4. 长时间AI学习阶段
print("\n5. 开始长时间AI学习...")
print("   学习模式: 多样化动作+自适应探索")
print("   持续时间: 无限期（按Ctrl+C停止）")
print("   学习目标: 掌握云台运动规律，实现精准控制")

# 定义AI学习动作集
actions = [
    "pan_left", "pan_right", "tilt_up", "tilt_down", 
    "zoom_in", "zoom_out", "stop"
]

# 学习策略配置
speed_levels = [
    ("slow", 10, 30),     # 慢速
    ("medium", 40, 60),   # 中速
    ("fast", 70, 90)      # 快速
]

# 位置探索范围
pan_range = (-10.0, 10.0)  # 水平探索范围
tilt_range = (-5.0, 5.0)   # 垂直探索范围

# 学习统计
learning_stats = {
    "total_rounds": 0,
    "success_count": 0,
    "failed_count": 0,
    "action_distribution": {action: 0 for action in actions},
    "speed_distribution": {"slow": 0, "medium": 0, "fast": 0}
}

try:
    # 获取初始位置
    response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
    initial_status = response.json()
    if initial_status.get("success"):
        initial_pos = initial_status.get("data", {}).get("position", {})
        current_pan = initial_pos.get("pan", 0.0)
        current_tilt = initial_pos.get("tilt", 0.0)
        current_zoom = initial_pos.get("zoom", 1.0)
    else:
        current_pan, current_tilt, current_zoom = 0.0, 0.0, 1.0
    
    start_time = time.time()
    
    while True:
        learning_stats["total_rounds"] += 1
        round_num = learning_stats["total_rounds"]
        
        # 1. 自适应探索策略
        # 根据当前位置，决定下一步动作
        explore_action = None
        
        # 边界探索：如果接近边界，向相反方向移动
        if current_pan < pan_range[0] + 1.0:
            explore_action = "pan_right"
        elif current_pan > pan_range[1] - 1.0:
            explore_action = "pan_left"
        elif current_tilt < tilt_range[0] + 1.0:
            explore_action = "tilt_up"
        elif current_tilt > tilt_range[1] - 1.0:
            explore_action = "tilt_down"
        
        # 随机选择动作（如果没有边界探索需求）
        if not explore_action:
            explore_action = random.choice(actions)
        
        # 2. 随机选择速度级别
        speed_info = random.choice(speed_levels)
        speed_name, speed_min, speed_max = speed_info
        speed = random.randint(speed_min, speed_max)
        
        # 3. 执行动作
        print(f"\n学习轮次 {round_num}：")
        print(f"  执行动作: {explore_action} ({speed_name})")
        print(f"  执行速度: {speed}")
        
        ptz_request = {
            "action": explore_action,
            "speed": speed
        }
        
        response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=ptz_request)
        result = response.json()
        
        # 4. 更新统计
        learning_stats["action_distribution"][explore_action] += 1
        learning_stats["speed_distribution"][speed_name] += 1
        
        if result.get("success"):
            learning_stats["success_count"] += 1
            print(f"  ✓ 动作执行成功")
            
            # 5. 获取并更新当前位置
            status_response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
            status = status_response.json()
            if status.get("success"):
                position = status.get("data", {}).get("position", {})
                current_pan = position.get("pan", current_pan)
                current_tilt = position.get("tilt", current_tilt)
                current_zoom = position.get("zoom", current_zoom)
                print(f"  当前位置: pan={current_pan:.2f}, tilt={current_tilt:.2f}, zoom={current_zoom:.2f}")
        else:
            learning_stats["failed_count"] += 1
            print(f"  ✗ 动作执行失败: {result.get('message')}")
        
        # 6. 定期输出学习统计
        if round_num % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"\n=== 学习统计 ({round_num}轮次，{elapsed_time:.1f}秒) ===")
            print(f"  成功率: {learning_stats['success_count'] / round_num:.2%}")
            print(f"  动作分布: {learning_stats['action_distribution']}")
            print(f"  速度分布: {learning_stats['speed_distribution']}")
            print(f"  当前学习位置: pan={current_pan:.2f}, tilt={current_tilt:.2f}")
        
        # 7. 随机等待时间（0.3-1.5秒，平衡学习效率和设备负载）
        wait_time = random.uniform(0.3, 1.5)
        time.sleep(wait_time)
        
except KeyboardInterrupt:
    print("\n\n=== AI学习测试已停止 ===")
except Exception as e:
    print(f"\n\n=== 测试发生错误: {e} ===")
finally:
    # 停止所有动作
    print("\n1. 停止所有动作...")
    requests.post(f"{BASE_URL}/api/camera/ptz/action", json={"action": "stop"})
    
    # 不关闭摄像头和云台连接，保持学习状态
    print("\n2. 学习完成，保持摄像头和云台连接")
    print("   摄像头和云台保持打开状态，可继续学习")
    
    # 输出最终学习统计
    elapsed_time = time.time() - start_time
    print(f"\n=== 最终学习统计 ===")
    print(f"  总轮次: {learning_stats['total_rounds']}")
    print(f"  总时长: {elapsed_time:.1f}秒")
    print(f"  成功率: {learning_stats['success_count'] / learning_stats['total_rounds']:.2%}")
    print(f"  动作分布: {learning_stats['action_distribution']}")
    print(f"  速度分布: {learning_stats['speed_distribution']}")
    print(f"  最终学习位置: pan={current_pan:.2f}, tilt={current_tilt:.2f}")
    
    print("\n=== AI学习建议 ===")
    print("1. 继续保持摄像头和云台连接，AI可持续学习")
    print("2. 观察动作执行效果，调整学习策略")
    print("3. 尝试增加更复杂的动作组合")
    print("4. 结合视觉反馈优化控制精度")
