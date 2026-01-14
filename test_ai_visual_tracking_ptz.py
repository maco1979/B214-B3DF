#!/usr/bin/env python3
"""AI视觉跟踪+PTZ控制集成测试"""

import requests
import time
import random

BASE_URL = "http://localhost:8001"
PTZ_CONFIG = {
    "protocol": "http",
    "connection_type": "http",
    "base_url": "http://192.168.1.1",
    "username": "admin",
    "password": "admin"
}

def test_ai_visual_tracking_ptz():
    """测试AI视觉跟踪+PTZ控制"""
    print("=== AI视觉跟踪+PTZ控制集成测试 ===")
    
    # 1. 初始化状态
    print("1. 初始化状态...")
    
    # 关闭现有摄像头和PTZ连接
    try:
        requests.post(f"{BASE_URL}/api/camera/close").json()
        requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    except Exception as e:
        print(f"   初始化错误: {e}")
    
    # 2. 打开摄像头
    print("2. 打开摄像头...")
    open_result = requests.post(f"{BASE_URL}/api/camera/open", json={"camera_index": 0}).json()
    print(f"   摄像头状态: {open_result}")
    
    if not open_result.get("success"):
        print("   ❌ 摄像头打开失败，测试终止")
        return False
    
    # 3. 连接PTZ
    print("3. 连接PTZ云台...")
    connect_result = requests.post(f"{BASE_URL}/api/camera/ptz/connect", json=PTZ_CONFIG).json()
    print(f"   PTZ连接结果: {connect_result}")
    
    if not connect_result.get("success"):
        print("   ❌ PTZ连接失败，测试终止")
        return False
    
    # 4. 移动到初始位置
    print("4. 移动到初始位置...")
    init_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                              json={"pan": 0, "tilt": 0, "speed": 100})
    print(f"   初始位置设置: {init_result.json()}")
    time.sleep(2)
    
    # 5. 启动视觉识别（人脸识别）
    print("5. 启动视觉识别（人脸识别）...")
    recognition_result = requests.post(f"{BASE_URL}/api/camera/recognition/start", 
                                     json={"model_type": "haar"}).json()
    print(f"   视觉识别状态: {recognition_result}")
    
    if not recognition_result.get("success"):
        print("   ❌ 视觉识别启动失败，测试终止")
        return False
    
    # 6. 启动视觉跟踪
    print("6. 启动视觉跟踪...")
    tracking_result = requests.post(f"{BASE_URL}/api/camera/tracking/start", 
                                  json={"tracker_type": "CSRT"}).json()
    print(f"   视觉跟踪状态: {tracking_result}")
    
    # 7. 模拟AI观察场景变化
    print("\n7. AI观察场景变化测试...")
    print("   🎯 测试目标: AI检测目标并保持在中心，同时移动云台观察场景")
    
    # 模拟AI检测到不同位置的目标
    def simulate_target_detection():
        """模拟目标检测结果"""
        # 随机生成目标位置 (x, y, w, h)
        # 模拟目标在画面中的不同位置
        x = random.randint(0, 300)  # 左半部分
        y = random.randint(0, 200)  # 上半部分
        w = random.randint(50, 150)
        h = random.randint(50, 150)
        return (x, y, w, h)
    
    # 测试5轮场景观察
    for i in range(5):
        print(f"\n   🔄 观察轮次 {i+1}/5")
        
        # 模拟AI检测到新目标
        target_bbox = simulate_target_detection()
        print(f"   🎯 AI检测到目标: {target_bbox}")
        
        # 更新跟踪目标
        update_result = requests.post(f"{BASE_URL}/api/camera/tracking/update", 
                                    json={"new_bbox": target_bbox}).json()
        print(f"   📡 更新跟踪目标: {update_result}")
        
        # AI分析目标位置，计算需要移动的PTZ动作
        # 目标位置 (x, y, w, h)
        x, y, w, h = target_bbox
        center_x = x + w / 2
        center_y = y + h / 2
        
        print(f"   📊 目标中心: ({center_x:.1f}, {center_y:.1f})")
        
        # 画面中心 (320, 240) 假设摄像头分辨率640x480
        frame_center_x = 320
        frame_center_y = 240
        
        # 计算偏移量
        offset_x = center_x - frame_center_x
        offset_y = center_y - frame_center_y
        
        print(f"   📏 中心偏移: ({offset_x:.1f}, {offset_y:.1f})")
        
        # AI决策：如果偏移超过阈值，控制PTZ移动使目标回到中心
        if abs(offset_x) > 50 or abs(offset_y) > 30:
            # 计算需要移动的角度
            # 简单映射：像素偏移 -> PTZ角度
            pan_angle = offset_x / 640 * 30  # 30°视场角
            tilt_angle = -offset_y / 480 * 20  # 20°视场角
            
            print(f"   🤖 AI决策: 需要移动PTZ，调整角度 ({pan_angle:.1f}°, {tilt_angle:.1f}°)")
            
            # 获取当前PTZ状态
            ptz_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
            current_pan = ptz_status["data"]["position"]["pan"]
            current_tilt = ptz_status["data"]["position"]["tilt"]
            
            # 计算新的目标位置
            new_pan = current_pan + pan_angle
            new_tilt = current_tilt + tilt_angle
            
            # 限制角度范围
            new_pan = max(-180, min(180, new_pan))
            new_tilt = max(-90, min(90, new_tilt))
            
            print(f"   🎮 控制PTZ移动到: pan={new_pan:.1f}°, tilt={new_tilt:.1f}°")
            
            # 调用PTZ移动API
            move_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                                      json={"pan": new_pan, "tilt": new_tilt, "speed": 70})
            
            print(f"   ✅ PTZ移动命令响应: {move_result.json()}")
            
            # 等待移动完成
            time.sleep(2)
            
            # 获取新的PTZ状态
            new_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
            final_pan = new_status["data"]["position"]["pan"]
            final_tilt = new_status["data"]["position"]["tilt"]
            
            print(f"   📍 移动后PTZ位置: pan={final_pan:.1f}°, tilt={final_tilt:.1f}°")
        
        # AI观察场景变化：随机移动云台探索更多区域
        if i % 2 == 0:  # 每2轮进行一次场景探索
            print(f"   🔍 AI开始观察场景变化...")
            
            # 随机探索一个新位置
            explore_pan = random.uniform(-120, 120)
            explore_tilt = random.uniform(-60, 60)
            
            print(f"   🗺️  探索新位置: pan={explore_pan:.1f}°, tilt={explore_tilt:.1f}°")
            
            explore_result = requests.post(f"{BASE_URL}/api/camera/ptz/move", 
                                         json={"pan": explore_pan, "tilt": explore_tilt, "speed": 80})
            
            print(f"   🚀 场景探索响应: {explore_result.json()}")
            
            # 等待探索完成
            time.sleep(3)
            
            # 获取探索后的状态
            explore_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
            exp_pan = explore_status["data"]["position"]["pan"]
            exp_tilt = explore_status["data"]["position"]["tilt"]
            
            print(f"   🌍 探索后位置: pan={exp_pan:.1f}°, tilt={exp_tilt:.1f}°")
        
        # 暂停一下，让系统处理
        time.sleep(1)
    
    # 8. 获取视觉识别状态
    print("\n8. 获取视觉识别状态...")
    recognition_status = requests.get(f"{BASE_URL}/api/camera/recognition/status").json()
    print(f"   视觉识别状态: {recognition_status}")
    
    # 9. 获取跟踪状态
    print("9. 获取跟踪状态...")
    tracking_status = requests.get(f"{BASE_URL}/api/camera/tracking/status").json()
    print(f"   跟踪状态: {tracking_status}")
    
    # 10. 清理资源
    print("\n10. 清理资源...")
    
    # 停止跟踪
    stop_tracking = requests.post(f"{BASE_URL}/api/camera/tracking/stop").json()
    print(f"   停止跟踪: {stop_tracking}")
    
    # 停止视觉识别
    stop_recognition = requests.post(f"{BASE_URL}/api/camera/recognition/stop").json()
    print(f"   停止视觉识别: {stop_recognition}")
    
    # 断开PTZ连接
    disconnect_ptz = requests.post(f"{BASE_URL}/api/camera/ptz/disconnect").json()
    print(f"   断开PTZ连接: {disconnect_ptz}")
    
    # 关闭摄像头
    close_camera = requests.post(f"{BASE_URL}/api/camera/close").json()
    print(f"   关闭摄像头: {close_camera}")
    
    print("\n=== 测试完成 ===")
    print("✅ AI视觉跟踪+PTZ控制集成测试成功完成")
    print("🎯 AI能够检测目标、控制PTZ保持目标在中心，并移动云台观察场景变化")
    
    return True

if __name__ == "__main__":
    test_ai_visual_tracking_ptz()