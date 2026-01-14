import requests
import time
import json

# AI控制云台移动+拍照测试脚本
print("=== AI控制云台移动+拍照测试 ===")
print("AI将控制云台移动到不同位置，在每个位置拍照")
print("通过照片结果判断云台是否真的在动")

# 配置
BASE_URL = "http://localhost:8001"

# 测试位置列表
TEST_POSITIONS = [
    ("home", "初始位置", 0.0, 0.0),
    ("left", "左侧位置", -15.0, 0.0),
    ("right", "右侧位置", 15.0, 0.0),
    ("up", "上方位置", 0.0, 10.0),
    ("down", "下方位置", 0.0, -10.0),
    ("diagonal", "对角线位置", 10.0, 10.0)
]

# 连接PTZ
def connect_ptz():
    print("\n1. 连接PTZ...")
    ptz_config = {
        "protocol": "http",
        "connection_type": "http",
        "base_url": "http://localhost:8001",
        "username": "admin",
        "password": "admin"
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=ptz_config)
    result = response.json()
    print(f"PTZ连接结果: {result}")
    return result.get("success", False)

# 打开摄像头
def open_camera():
    print("\n2. 打开摄像头...")
    response = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0})
    result = response.json()
    print(f"摄像头打开结果: {result}")
    # 摄像头已打开也视为成功
    if result.get("message") == "摄像头已打开":
        return True
    return result.get("success", False)

# 关闭视觉跟踪
def stop_visual_tracking():
    print("\n3. 关闭视觉跟踪...")
    response = requests.post(f"{BASE_URL}/api/camera/tracking/stop")
    result = response.json()
    print(f"关闭视觉跟踪结果: {result}")
    # 无论结果如何都视为成功，因为可能本来就没开启
    return True

# 关闭视觉识别
def stop_visual_recognition():
    print("\n4. 关闭视觉识别...")
    response = requests.post(f"{BASE_URL}/api/camera/recognition/stop")
    result = response.json()
    print(f"关闭视觉识别结果: {result}")
    # 无论结果如何都视为成功，因为可能本来就没开启
    return True

# 发送PTZ命令
def send_ptz_command(action, speed):
    request_data = {
        "action": action,
        "speed": speed
    }
    response = requests.post(f"{BASE_URL}/api/camera/ptz/action", json=request_data)
    return response.json()

# 移动到指定位置
def move_to_position(pan, tilt, speed=30):
    print(f"\n移动到位置: pan={pan}, tilt={tilt}")
    
    # 构建位置请求
    position_request = {
        "pan": pan,
        "tilt": tilt,
        "zoom": 1.0,
        "speed": speed
    }
    
    # 发送移动命令
    response = requests.post(f"{BASE_URL}/api/camera/ptz/move", json=position_request)
    result = response.json()
    print(f"移动结果: {result}")
    return result.get("success", False)

# 拍照
def take_photo():
    response = requests.post(f"{BASE_URL}/api/camera/take_photo")
    result = response.json()
    return result

# 获取当前位置
def get_current_position():
    response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
    result = response.json()
    if result.get("success"):
        return result.get("data", {}).get("position", {})
    return {}

# 停止所有动作
def stop_all():
    send_ptz_command("stop", 0)

# 主测试流程
def main_test():
    # 1. 初始化设备
    if not open_camera():
        print("摄像头打开失败，退出测试")
        return
    
    if not connect_ptz():
        print("PTZ连接失败，退出测试")
        return
    
    # 关闭视觉跟踪和识别，避免自动调整干扰测试
    stop_visual_tracking()
    stop_visual_recognition()
    
    # 2. 测试流程
    test_results = []
    
    for name, desc, target_pan, target_tilt in TEST_POSITIONS:
        print(f"\n=== 测试位置: {desc} ===")
        
        # 移动到目标位置
        move_success = move_to_position(target_pan, target_tilt)
        
        # 等待移动完成
        time.sleep(2)
        
        # 获取当前位置
        current_pos = get_current_position()
        actual_pan = current_pos.get("pan", 0.0)
        actual_tilt = current_pos.get("tilt", 0.0)
        
        # 拍照
        photo_result = take_photo()
        photo_success = photo_result.get("success", False)
        
        # 记录结果
        test_results.append({
            "position_name": name,
            "description": desc,
            "target_pan": target_pan,
            "target_tilt": target_tilt,
            "actual_pan": actual_pan,
            "actual_tilt": actual_tilt,
            "move_success": move_success,
            "photo_success": photo_success,
            "photo_result": photo_result,
            "timestamp": time.time()
        })
        
        print(f"目标位置: pan={target_pan}, tilt={target_tilt}")
        print(f"实际位置: pan={actual_pan}, tilt={actual_tilt}")
        print(f"移动成功: {move_success}")
        print(f"拍照成功: {photo_success}")
        
        # 停止动作
        stop_all()
        time.sleep(1)
    
    # 3. 返回初始位置
    print("\n=== 返回初始位置 ===")
    move_to_position(0.0, 0.0)
    time.sleep(2)
    stop_all()
    
    # 4. 生成测试报告
    print("\n=== 测试报告 ===")
    print(f"测试位置数量: {len(TEST_POSITIONS)}")
    
    # 统计结果
    success_count = 0
    for result in test_results:
        # 计算位置误差
        pan_error = abs(result["target_pan"] - result["actual_pan"])
        tilt_error = abs(result["target_tilt"] - result["actual_tilt"])
        
        # 判断是否成功（误差小于5度）
        if pan_error < 5.0 and tilt_error < 5.0 and result["move_success"] and result["photo_success"]:
            success_count += 1
        
        print(f"\n{result['description']}:")
        print(f"  目标: pan={result['target_pan']}, tilt={result['target_tilt']}")
        print(f"  实际: pan={result['actual_pan']}, tilt={result['actual_tilt']}")
        print(f"  误差: pan={pan_error:.2f}°, tilt={tilt_error:.2f}°")
        print(f"  移动: {'✓' if result['move_success'] else '✗'}")
        print(f"  拍照: {'✓' if result['photo_success'] else '✗'}")
        
    # 总体成功率
    success_rate = (success_count / len(TEST_POSITIONS)) * 100
    print(f"\n=== 总体结果 ===")
    print(f"成功位置数: {success_count}/{len(TEST_POSITIONS)}")
    print(f"成功率: {success_rate:.1f}%")
    
    # 分析结果
    if success_rate == 0:
        print("\n❌ 测试失败: 所有位置都没有成功移动")
        print("可能原因:")
        print("1. PTZ连接问题")
        print("2. 命令没有发送到硬件")
        print("3. 硬件不支持HTTP API")
        print("4. 硬件IP或端口配置错误")
    elif success_rate < 50:
        print("\n⚠️  测试部分失败: 部分位置没有成功移动")
        print("可能原因:")
        print("1. 部分命令执行失败")
        print("2. 硬件响应延迟")
        print("3. 位置误差较大")
    else:
        print("\n✅ 测试成功: 大部分位置成功移动")
        print("云台正在按照AI指令移动")
    
    # 保存测试结果
    with open("ptz_photo_test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print(f"\n测试结果已保存到: ptz_photo_test_results.json")

# 执行测试
if __name__ == "__main__":
    try:
        main_test()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        stop_all()
    except Exception as e:
        print(f"\n\n测试发生错误: {e}")
        import traceback
        traceback.print_exc()
        stop_all()
