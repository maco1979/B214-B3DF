import requests
import time
import random
import threading

# AI控制PTZ脚本
print("=== AI自动控制PTZ云台 ===")
print("AI将根据预设算法自动控制云台运动")
print("按Ctrl+C可停止AI控制")

# 配置
BASE_URL = "http://localhost:8001"

# AI控制参数
AI_CONTROL_CONFIG = {
    "mode": "adaptive_exploration",  # AI控制模式
    "speed_range": (50, 100),         # 速度范围（提高速度）
    "action_duration": (3.0, 8.0),    # 每个动作持续时间（增加持续时间以实现更大角度变化）
    "learning_rate": 0.1,            # AI学习率
    "exploration_rate": 0.3,         # 探索率（降低探索率，增加持续同一方向运动的概率）
    "max_steps": 10000               # 最大控制步数
}

# AI控制状态
ai_control_state = {
    "current_step": 0,
    "last_position": {"pan": 0.0, "tilt": 0.0, "zoom": 1.0},
    "success_count": 0,
    "failed_count": 0,
    "action_history": [],
    "is_running": False
}

# 动作列表
AI_ACTIONS = ["pan_left", "pan_right", "tilt_up", "tilt_down", "stop"]

# 获取当前PTZ状态
def get_ptz_status():
    response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
    return response.json()

# 发送PTZ命令
def send_ptz_command(action, speed):
    request_data = {
        "action": action,
        "speed": speed
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=request_data)
    return response.json()

# AI控制算法：大角度运动优化

def ai_decision(current_position):
    """AI决策算法：优化大角度运动，持续向一个方向运动"""
    # 优先持续同一方向运动，减少切换
    if ai_control_state["action_history"]:
        # 最近5个动作
        recent_actions = [a["action"] for a in ai_control_state["action_history"][-5:]]
        
        # 如果最近有连续相同方向的动作，继续同一方向
        # 增加连续动作的要求，从2次增加到3次
        for action in AI_ACTIONS[:-1]:  # 排除stop动作
            if recent_actions.count(action) >= 3:
                # 继续同一方向
                return action
        
        # 如果最近动作中有占比超过60%的方向，继续该方向
        if len(recent_actions) >= 4:
            for action in AI_ACTIONS[:-1]:
                if recent_actions.count(action) >= 3:
                    return action
    
    # 大幅降低探索率，增加利用
    exploration_rate = AI_CONTROL_CONFIG["exploration_rate"] * 0.5  # 进一步降低探索率
    
    # 探索：随机选择动作
    if random.random() < exploration_rate:
        # 不随机选择stop，只选择移动动作
        return random.choice(AI_ACTIONS[:-1])  # 排除stop动作
    
    # 利用：选择一个主要方向进行持续运动
    # 基于当前位置，选择距离边界最远的方向
    pan = current_position["pan"]
    tilt = current_position["tilt"]
    
    # 计算距离边界的距离
    pan_distance_left = pan - (-180.0)  # 距离左边界
    pan_distance_right = 180.0 - pan    # 距离右边界
    tilt_distance_down = tilt - (-90.0)  # 距离下边界
    tilt_distance_up = 90.0 - tilt       # 距离上边界
    
    # 创建方向距离字典，只考虑pan方向（更容易实现大角度变化）
    direction_distances = {
        "pan_left": pan_distance_left * 1.5,  # 增加pan方向的权重
        "pan_right": pan_distance_right * 1.5,  # 增加pan方向的权重
        "tilt_up": tilt_distance_up,
        "tilt_down": tilt_distance_down
    }
    
    # 选择距离边界最远的方向，这样可以有最大的运动空间
    best_direction = max(direction_distances, key=direction_distances.get)
    
    return best_direction

# AI控制主循环
def ai_control_loop():
    print("\n=== AI控制启动 ===")
    print(f"控制模式: {AI_CONTROL_CONFIG['mode']}")
    print(f"速度范围: {AI_CONTROL_CONFIG['speed_range']}")
    print(f"动作持续时间: {AI_CONTROL_CONFIG['action_duration']}")
    
    ai_control_state["is_running"] = True
    
    # 获取初始位置
    status = get_ptz_status()
    if status.get("success"):
        ai_control_state["last_position"] = status.get("data", {}).get("position", {})
    
    try:
        while ai_control_state["is_running"] and ai_control_state["current_step"] < AI_CONTROL_CONFIG["max_steps"]:
            ai_control_state["current_step"] += 1
            step = ai_control_state["current_step"]
            
            # 1. AI决策
            current_action = ai_decision(ai_control_state["last_position"])
            
            # 2. 随机速度
            min_speed, max_speed = AI_CONTROL_CONFIG["speed_range"]
            speed = random.randint(min_speed, max_speed)
            
            # 3. 执行动作
            print(f"\nAI控制步骤 {step}：")
            print(f"  动作: {current_action}")
            print(f"  速度: {speed}")
            
            result = send_ptz_command(current_action, speed)
            
            # 4. 更新状态
            if result.get("success"):
                ai_control_state["success_count"] += 1
                print(f"  ✓ 执行成功")
                
                # 更新位置
                if "data" in result:
                    ai_control_state["last_position"] = result["data"]
                
                # 记录动作
                ai_control_state["action_history"].append({
                    "step": step,
                    "action": current_action,
                    "speed": speed,
                    "success": True,
                    "position": ai_control_state["last_position"]
                })
            else:
                ai_control_state["failed_count"] += 1
                print(f"  ✗ 执行失败: {result.get('message')}")
                
            # 5. 等待动作完成
            min_duration, max_duration = AI_CONTROL_CONFIG["action_duration"]
            wait_time = random.uniform(min_duration, max_duration)
            time.sleep(wait_time)
            
            # 6. 定期输出统计
            if step % 50 == 0:
                print(f"\n=== AI控制统计 (步骤 {step}) ===")
                print(f"  成功率: {ai_control_state['success_count'] / step:.2%}")
                print(f"  当前位置: pan={ai_control_state['last_position']['pan']:.1f}, tilt={ai_control_state['last_position']['tilt']:.1f}")
                print(f"  探索率: {AI_CONTROL_CONFIG['exploration_rate']:.1f}")
                
    except KeyboardInterrupt:
        print("\n\n=== AI控制已停止 ===")
    except Exception as e:
        print(f"\n\n=== AI控制发生错误: {e} ===")
    finally:
        # 停止所有动作
        send_ptz_command("stop", 0)
        ai_control_state["is_running"] = False
        
        # 输出最终统计
        print("\n=== AI控制最终统计 ===")
        total_steps = ai_control_state["current_step"]
        success_rate = ai_control_state['success_count'] / total_steps if total_steps > 0 else 0
        print(f"  总步骤: {total_steps}")
        print(f"  成功次数: {ai_control_state['success_count']}")
        print(f"  失败次数: {ai_control_state['failed_count']}")
        print(f"  成功率: {success_rate:.2%}")
        print(f"  最终位置: pan={ai_control_state['last_position']['pan']:.1f}, tilt={ai_control_state['last_position']['tilt']:.1f}")
        print("\n=== AI控制结束 ===")

# 主程序
if __name__ == "__main__":
    try:
        # 检查PTZ状态
        ptz_status = get_ptz_status()
        print(f"PTZ当前状态: {ptz_status}")
        
        # 如果PTZ未连接，尝试连接
        if not ptz_status.get("success") or not ptz_status.get("data", {}).get("connected"):
            print("\n正在连接PTZ...")
            ptz_config = {
                "protocol": "http",
                "connection_type": "http",
                "base_url": "http://localhost:8001",
                "username": "admin",
                "password": "admin"
            }
            response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
            print(f"PTZ连接结果: {response.json()}")
        
        # 检查摄像头状态
        camera_status = requests.get(f"{BASE_URL}/api/camera/status").json()
        print(f"摄像头当前状态: {camera_status}")
        
        # 如果摄像头未打开，尝试打开
        if not camera_status.get("success") or not camera_status.get("data", {}).get("is_open"):
            print("\n正在打开摄像头...")
            response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0})
            print(f"摄像头打开结果: {response.json()}")
        
        # 启动AI控制
        print("\n=== 启动AI控制 ===")
        ai_control_loop()
        
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"发生错误: {e}")
