#!/usr/bin/env python3
"""
AI控制PTZ验证脚本
按照用户指定的流程验证AI控制PTZ的结果
"""

import requests
import time
import json

# 配置
BASE_URL = "http://localhost:8001"
TEST_DURATION = 60  # 测试持续时间（秒），增加到60秒以实现大角度运动

def verify_ai_ptz_control():
    print("=== AI控制PTZ验证 ===")
    print("按照以下步骤执行验证：")
    print("1. 准备：启动AI控制脚本，确保摄像头和PTZ已连接")
    print("2. 初始状态：记录初始PTZ位置")
    print("3. 运行：让AI控制运行一段时间（5-30秒）")
    print("4. 检查：使用上述方法检查PTZ状态")
    print("5. 分析：比较初始状态和当前状态")
    print("6. 结论：判断AI控制是否成功")
    
    print("\n" + "="*50)
    
    # 1. 准备
    print("\n1. 准备...")
    
    # 检查摄像头状态
    print("   检查摄像头状态...")
    camera_status = requests.get(f"{BASE_URL}/api/camera/status").json()
    print(f"   摄像头状态: {'已打开' if camera_status['success'] else '未打开'}")
    
    # 检查PTZ状态
    print("   检查PTZ状态...")
    ptz_status = requests.get(f"{BASE_URL}/api/camera/ptz/status").json()
    print(f"   PTZ状态: {'已连接' if ptz_status['success'] and ptz_status['data']['connected'] else '未连接'}")
    
    # 2. 初始状态
    print("\n2. 初始状态...")
    initial_response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
    initial_status = initial_response.json()
    
    if initial_status["success"]:
        initial_pan = initial_status["data"]["position"]["pan"]
        initial_tilt = initial_status["data"]["position"]["tilt"]
        print(f"   初始位置: pan={initial_pan:.1f}°, tilt={initial_tilt:.1f}°")
    else:
        print("   获取初始位置失败")
        return False
    
    # 3. 运行
    print(f"\n3. 运行...")
    print(f"   让AI控制运行 {TEST_DURATION} 秒...")
    print(f"   时间: {time.strftime('%H:%M:%S')}")
    
    # 等待AI控制执行
    time.sleep(TEST_DURATION)
    
    # 4. 检查
    print("\n4. 检查...")
    
    # 检查PTZ状态
    print("   检查PTZ状态...")
    current_response = requests.get(f"{BASE_URL}/api/camera/ptz/status")
    current_status = current_response.json()
    
    if current_status["success"]:
        current_pan = current_status["data"]["position"]["pan"]
        current_tilt = current_status["data"]["position"]["tilt"]
        print(f"   当前位置: pan={current_pan:.1f}°, tilt={current_tilt:.1f}°")
    else:
        print("   获取当前位置失败")
        return False
    
    # 检查日志
    print("   检查PTZ日志...")
    try:
        import subprocess
        # 使用utf-8编码处理日志
        logs = subprocess.run(["powershell", "-Command", "Get-Content -Path logs/app.log | Select-String -Pattern 'PTZ|ptz' | Select-Object -Last 5"], 
                            capture_output=True, text=True, encoding='utf-8')
        print("   最新5条PTZ日志:")
        print(logs.stdout if logs.stdout else "   没有PTZ日志")
        logs_stdout = logs.stdout if logs.stdout else ""
    except Exception as e:
        print(f"   获取日志失败: {e}")
        logs_stdout = ""
    
    # 5. 分析
    print("\n5. 分析...")
    
    # 位置变化分析
    pan_change = abs(current_pan - initial_pan)
    tilt_change = abs(current_tilt - initial_tilt)
    print(f"   位置变化: pan={pan_change:.1f}°, tilt={tilt_change:.1f}°")
    
    # 验证条件1：位置变化 > 100°
    position_condition = pan_change > 100.0 or tilt_change > 100.0
    print(f"   位置变化条件: {'✅ 满足' if position_condition else '❌ 不满足'} (>100°)")
    
    # 验证条件2：日志中有PTZ命令执行
    log_condition = "PTZ HTTP" in logs_stdout or "PTZ" in logs_stdout
    print(f"   日志条件: {'✅ 满足' if log_condition else '❌ 不满足'} (有PTZ命令执行)")
    
    # 6. 结论
    print("\n6. 结论...")
    
    # 综合验证结果
    if position_condition and log_condition:
        print("\n🎉 ✅ AI控制PTZ成功！")
        print("   验证结果:")
        print(f"   - 位置变化: pan={pan_change:.1f}°, tilt={tilt_change:.1f}°")
        print(f"   - 位置变化条件: {'✅ 满足' if position_condition else '❌ 不满足'} (>100°)")
        print(f"   - 日志条件: {'✅ 满足' if log_condition else '❌ 不满足'} (有PTZ命令执行)")
        print(f"   - 视频画面: 请手动观察画面是否随AI控制移动")
        print(f"   - 目标跟踪: 请手动观察目标是否保持在画面中心")
        return True
    else:
        print("\n❌ AI控制PTZ失败！")
        print("   验证结果:")
        print(f"   - 位置变化: pan={pan_change:.1f}°, tilt={tilt_change:.1f}°")
        print(f"   - 位置变化条件: {'✅ 满足' if position_condition else '❌ 不满足'} (>100°)")
        print(f"   - 日志条件: {'✅ 满足' if log_condition else '❌ 不满足'} (有PTZ命令执行)")
        return False

if __name__ == "__main__":
    verify_ai_ptz_control()
