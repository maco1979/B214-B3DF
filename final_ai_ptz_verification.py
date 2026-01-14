#!/usr/bin/env python3
"""最终AI PTZ控制验证脚本"""

import asyncio
import requests
import time
from backend.src.core.services.ptz_camera_controller import PTZCameraController, PTZProtocol, PTZAction

async def final_ai_ptz_verification():
    """最终AI PTZ控制验证"""
    print("=== 最终AI PTZ控制验证 ===\n")
    
    # 1. 检测真实摄像头IP
    print("1. 检测真实摄像头IP...")
    real_camera_ips = []
    
    # 扫描192.168.1.1到192.168.1.50
    for i in range(1, 51):
        ip = f"192.168.1.{i}"
        print(f"   检测: {ip}...", end="\r")
        
        try:
            # 检测海康威视ISAPI接口
            response = requests.get(f"http://{ip}/ISAPI", timeout=1)
            # 排除HTML响应（路由器）
            if "<html" not in response.text.lower() and "<title" not in response.text.lower():
                real_camera_ips.append(ip)
                print(f"   ✅ 发现可能的摄像头: {ip}")
        except:
            pass
    
    print(f"   扫描完成，发现 {len(real_camera_ips)} 个可能的摄像头IP\n")
    
    # 2. 配置PTZ控制器
    print("2. 配置PTZ控制器...")
    
    # 使用第一个发现的摄像头IP，如果没有则使用默认IP（192.168.1.64，海康威视默认）
    camera_ip = real_camera_ips[0] if real_camera_ips else "192.168.1.64"
    print(f"   使用摄像头IP: {camera_ip}")
    
    ptz = PTZCameraController(
        protocol=PTZProtocol.HTTP_API,
        connection_type="http",
        base_url=f"http://{camera_ip}",
        username="admin",
        password="admin"
    )
    
    # 3. 连接摄像头
    print("3. 连接摄像头...")
    result = await ptz.connect()
    print(f"   连接结果: {result}")
    
    if not result["success"]:
        print("   ❌ 连接失败，跳过后续测试")
        return
    
    # 4. 执行大角度移动测试（>100°）
    print("\n4. 执行大角度移动测试（>100°）...")
    
    # 获取初始位置
    initial_state = ptz.get_status()
    initial_pan = initial_state["position"]["pan"]
    initial_tilt = initial_state["position"]["tilt"]
    print(f"   初始位置: pan={initial_pan}, tilt={initial_tilt}")
    
    # 执行向右120°移动
    print("   执行向右120°移动...")
    move_result = await ptz.move_to_position(pan=initial_pan + 120, tilt=initial_tilt, speed=100)
    print(f"   移动结果: {move_result}")
    
    # 获取移动后位置
    after_state = ptz.get_status()
    after_pan = after_state["position"]["pan"]
    after_tilt = after_state["position"]["tilt"]
    print(f"   移动后位置: pan={after_pan}, tilt={after_tilt}")
    
    # 计算实际移动角度
    actual_movement = abs(after_pan - initial_pan)
    print(f"   实际移动角度: {actual_movement}°")
    
    # 5. 验证移动结果
    print("\n5. 验证移动结果...")
    
    if actual_movement >= 100:
        print("   ✅ 成功！移动角度超过100°，符合要求")
        print("   🎯 AI可以真正控制PTZ摄像头进行大角度移动")
    else:
        print("   ⚠️  移动角度不足100°，需要检查配置")
        print("   建议: 检查摄像头IP是否正确，检查摄像头是否支持HTTP控制")
    
    # 6. 执行AI自动跟踪模拟测试
    print("\n6. 执行AI自动跟踪模拟测试...")
    
    # 模拟AI视觉识别到目标在画面左侧
    print("   模拟目标在画面左侧，AI调整摄像头...")
    track_result = await ptz.auto_track_object((100, 200, 100, 100), (640, 480))
    print(f"   跟踪结果: {track_result}")
    
    # 再次获取位置
    final_state = ptz.get_status()
    final_pan = final_state["position"]["pan"]
    print(f"   跟踪后位置: pan={final_pan}")
    
    # 7. 结论
    print("\n=== 验证结论 ===")
    print(f"1. 真实摄像头IP: {'发现' if real_camera_ips else '未发现，使用默认'}")
    print(f"2. 摄像头连接: {'成功' if result['success'] else '失败'}")
    print(f"3. 大角度移动测试: {'通过' if actual_movement >= 100 else '未通过'}")
    print(f"4. AI自动跟踪: {'正常执行' if track_result['success'] else '执行失败'}")
    
    if result['success'] and actual_movement >= 100:
        print("\n🎉 验证成功！AI可以真正控制PTZ摄像头进行大角度移动")
        print("   系统已按照要求完成所有功能:")
        print("   - 真实硬件检测，避免假连接")
        print("   - 支持大角度移动（>100°）")
        print("   - AI自动控制PTZ动作")
        print("   - 完整的验证流程")
    else:
        print("\n⚠️  验证未完全通过，请检查以下问题:")
        print("   - 确保使用了正确的摄像头IP")
        print("   - 检查摄像头是否支持HTTP API控制")
        print("   - 确认摄像头登录凭证正确")
        print("   - 检查网络连接是否正常")

if __name__ == "__main__":
    asyncio.run(final_ai_ptz_verification())